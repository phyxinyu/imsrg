#include "Commutator.hh"
#include "MpiSupport.hh"
#include "PhysicalConstants.hh"
#include "AngMom.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

#include <omp.h>

namespace Commutator
{
  std::vector<imsrg_mpi::TwoBodyElementRequest> GetPandyaElementRequestsForChannel(
    const Operator& X, const Operator& Y, const Operator& Z, std::size_t ch);

namespace EventIMSRG2
{
  int OneBodyOwner(int i);
  int TwoBodyOwner(ModelSpace& modelspace, int ch_bra);
  bool OwnsTwoBodyOutput(ModelSpace& modelspace, int ch_bra);

  struct OccPair
  {
    std::size_t a;
    std::size_t b;
    double factor;
  };

  struct OneBodyTargetCache
  {
    std::vector<std::vector<std::size_t>> targets;
  };

  struct OwnedTwoBodyKeyCache
  {
    std::vector<std::array<std::size_t, 2>> keys;
  };

  struct PandyaChannelCache
  {
    std::size_t ch = 0;
    arma::uvec ph_kets;
    std::vector<int> ket_phase;
  };

  struct EventIndexCache
  {
    std::vector<std::size_t> all_orbits;
    std::vector<OccPair> occ_diff_pairs;
    OneBodyTargetCache one_body_targets;
    OwnedTwoBodyKeyCache owned_two_body_keys;
    std::vector<std::array<std::size_t, 2>> scalar_two_body_keys;

    explicit EventIndexCache(Operator& Z)
    {
      ModelSpace& modelspace = *Z.modelspace;
      all_orbits.assign(modelspace.all_orbits.begin(), modelspace.all_orbits.end());

      one_body_targets.targets.resize(modelspace.GetNumberOrbits());
      for (std::size_t i : all_orbits)
      {
        Orbit& oi = modelspace.GetOrbit(i);
        auto& targets = one_body_targets.targets[i];
        for (std::size_t j : Z.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
          targets.push_back(j);
      }

      for (std::size_t a : all_orbits)
      {
        Orbit& oa = modelspace.GetOrbit(a);
        for (std::size_t b : all_orbits)
        {
          Orbit& ob = modelspace.GetOrbit(b);
          const double factor = oa.occ - ob.occ;
          if (std::abs(factor) >= ModelSpace::OCC_CUT)
            occ_diff_pairs.push_back({a, b, factor});
        }
      }

      for (const auto& iter : Z.TwoBody.MatEl)
      {
        scalar_two_body_keys.push_back(iter.first);
        if (OwnsTwoBodyOutput(modelspace, static_cast<int>(iter.first[0])))
          owned_two_body_keys.keys.push_back(iter.first);
      }
    }
  };

  std::size_t GetPandyaBatchSizeEvent()
  {
    const char* env_value = std::getenv("IMSRG_MPI_PANDYA_CC_BATCH");
    if (env_value != nullptr && env_value[0] != '\0')
    {
      char* endptr = nullptr;
      unsigned long parsed = std::strtoul(env_value, &endptr, 10);
      if (endptr != env_value && parsed > 0)
        return static_cast<std::size_t>(parsed);
    }
    return 4;
  }

  int OneBodyOwner(int i)
  {
    return imsrg_mpi::Size() <= 1 ? 0 : i % imsrg_mpi::Size();
  }

  int TwoBodyOwner(ModelSpace& modelspace, int ch_bra)
  {
    return imsrg_mpi::Size() <= 1 ? 0 : imsrg_mpi::TwoBodyChannelOwner(modelspace, ch_bra);
  }

  bool OwnsOneBodyRow(int i)
  {
    return !imsrg_mpi::Enabled() || OneBodyOwner(i) == imsrg_mpi::Rank();
  }

  bool OwnsTwoBodyOutput(ModelSpace& modelspace, int ch_bra)
  {
    return !imsrg_mpi::Enabled() || TwoBodyOwner(modelspace, ch_bra) == imsrg_mpi::Rank();
  }

  std::vector<std::size_t> OrbitList(ModelSpace& modelspace)
  {
    return std::vector<std::size_t>(modelspace.all_orbits.begin(), modelspace.all_orbits.end());
  }

  std::vector<std::vector<std::vector<imsrg_mpi::OneBodyContribution>>> MakeOneBodyBuffers()
  {
    return std::vector<std::vector<std::vector<imsrg_mpi::OneBodyContribution>>>(
      omp_get_max_threads(), std::vector<std::vector<imsrg_mpi::OneBodyContribution>>(imsrg_mpi::Size()));
  }

  std::vector<std::vector<std::vector<imsrg_mpi::OneBodyContribution>>> MakeOneBodyBuffers(std::size_t reserve_hint)
  {
    auto buffers = MakeOneBodyBuffers();
    const std::size_t denom = std::max<std::size_t>(1, buffers.size() * static_cast<std::size_t>(imsrg_mpi::Size()));
    const std::size_t per_buffer = reserve_hint / denom + 1;
    for (auto& thread_buffers : buffers)
      for (auto& rank_buffer : thread_buffers)
        rank_buffer.reserve(per_buffer);
    return buffers;
  }

  std::vector<std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>> MakeTwoBodyBuffers()
  {
    return std::vector<std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>>(
      omp_get_max_threads(), std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>(imsrg_mpi::Size()));
  }

  std::vector<std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>> MakeTwoBodyBuffers(std::size_t reserve_hint)
  {
    auto buffers = MakeTwoBodyBuffers();
    const std::size_t denom = std::max<std::size_t>(1, buffers.size() * static_cast<std::size_t>(imsrg_mpi::Size()));
    const std::size_t per_buffer = reserve_hint / denom + 1;
    for (auto& thread_buffers : buffers)
      for (auto& rank_buffer : thread_buffers)
        rank_buffer.reserve(per_buffer);
    return buffers;
  }

  void PushOneBody(std::vector<std::vector<std::vector<imsrg_mpi::OneBodyContribution>>>& buffers,
                   int tid, int i, int j, double value)
  {
    if (std::abs(value) == 0.0)
      return;
    buffers[tid][OneBodyOwner(i)].push_back({i, j, value});
  }

  void PushTwoBody(std::vector<std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>>& buffers,
                   ModelSpace& modelspace, int tid, int ch_bra, int ch_ket, int ibra, int iket, double value)
  {
    if (std::abs(value) == 0.0)
      return;
    buffers[tid][TwoBodyOwner(modelspace, ch_bra)].push_back({ch_bra, ch_ket, ibra, iket, value});
  }

  std::vector<std::array<std::size_t, 2>> ScalarTwoBodyKeys(ModelSpace& modelspace)
  {
    std::vector<std::array<std::size_t, 2>> keys;
    keys.reserve(modelspace.GetNumberTwoBodyChannels());
    for (std::size_t ch = 0; ch < modelspace.GetNumberTwoBodyChannels(); ++ch)
      keys.push_back({ch, ch});
    return keys;
  }

  double MatrixElement(const Operator& op, int ch_bra, int ch_ket, int i, int j)
  {
    if (ch_bra <= ch_ket)
      return op.TwoBody.GetMatrix(ch_bra, ch_ket)(i, j);
    const int h = op.IsHermitian() ? 1 : -1;
    return h * op.TwoBody.GetMatrix(ch_ket, ch_bra)(j, i);
  }

  void AccumulateIntermediateSet(const Operator& L, const Operator& R,
                                 int ch_bra, int ch_mid, int ch_ket,
                                 int ibra, int iket,
                                 const arma::uvec& intermediate_indices,
                                 const arma::vec* weights,
                                 double sign,
                                 double& accum)
  {
    for (arma::uword iab = 0; iab < intermediate_indices.n_elem; ++iab)
    {
      const int ab = static_cast<int>(intermediate_indices(iab));
      const double weight = weights == nullptr ? 1.0 : (*weights)(iab);
      if (std::abs(weight) < ModelSpace::OCC_CUT)
        continue;
      accum += sign * weight *
               MatrixElement(L, ch_bra, ch_mid, ibra, ab) *
               MatrixElement(R, ch_mid, ch_ket, ab, iket);
    }
  }

  void BuildScalarMppMhhEvent(const Operator& X, const Operator& Y, const Operator& Z,
                              TwoBodyME& Mpp, TwoBodyME& Mhh)
  {
    const double t_start = omp_get_wtime();
    Mpp = Z.TwoBody;
    Mhh = Z.TwoBody;
    Mpp.Erase();
    Mhh.Erase();

    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
    {
      X.profiler.timer["EventIMSRG2::BuildScalarMppMhhEvent"] += omp_get_wtime() - t_start;
      return;
    }

    EventIndexCache cache(const_cast<Operator&>(Z));
    const auto& keys = cache.owned_two_body_keys.keys;

#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ikey = 0; ikey < keys.size(); ++ikey)
    {
      const int ch_bra = static_cast<int>(keys[ikey][0]);
      const int ch_ket = static_cast<int>(keys[ikey][1]);
      TwoBodyChannel& tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel& tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);

      const int ch_ab_XY = ch_bra;
      const int ch_ab_YX = ch_ket;
      TwoBodyChannel& tbc_ab_XY = Z.modelspace->GetTwoBodyChannel(ch_ab_XY);
      TwoBodyChannel& tbc_ab_YX = Z.modelspace->GetTwoBodyChannel(ch_ab_YX);

      arma::mat& Matrixpp = Mpp.GetMatrix(ch_bra, ch_ket);
      arma::mat& Matrixhh = Mhh.GetMatrix(ch_bra, ch_ket);
      const int nbras = static_cast<int>(tbc_bra.GetNumberKets());
      const int nkets = static_cast<int>(tbc_ket.GetNumberKets());

      for (int ibra = 0; ibra < nbras; ++ibra)
      {
        for (int iket = 0; iket < nkets; ++iket)
        {
          double mpp = 0.0;
          double mhh = 0.0;

          AccumulateIntermediateSet(X, Y, ch_bra, ch_ab_XY, ch_ket, ibra, iket,
                                    tbc_ab_XY.GetKetIndex_pp(), nullptr, +1.0, mpp);
          AccumulateIntermediateSet(X, Y, ch_bra, ch_ab_XY, ch_ket, ibra, iket,
                                    tbc_ab_XY.GetKetIndex_hh(), &tbc_ab_XY.Ket_occ_hh, +1.0, mhh);
          AccumulateIntermediateSet(X, Y, ch_bra, ch_ab_XY, ch_ket, ibra, iket,
                                    tbc_ab_XY.GetKetIndex_hh(), &tbc_ab_XY.Ket_unocc_hh, +1.0, mpp);
          AccumulateIntermediateSet(X, Y, ch_bra, ch_ab_XY, ch_ket, ibra, iket,
                                    tbc_ab_XY.GetKetIndex_ph(), &tbc_ab_XY.Ket_unocc_ph, +1.0, mpp);

          if (!(Z.IsHermitian() && ch_bra == ch_ket) &&
              !(Z.IsAntiHermitian() && ch_bra == ch_ket))
          {
            AccumulateIntermediateSet(Y, X, ch_bra, ch_ab_YX, ch_ket, ibra, iket,
                                      tbc_ab_YX.GetKetIndex_pp(), nullptr, -1.0, mpp);
            AccumulateIntermediateSet(Y, X, ch_bra, ch_ab_YX, ch_ket, ibra, iket,
                                      tbc_ab_YX.GetKetIndex_hh(), &tbc_ab_YX.Ket_occ_hh, -1.0, mhh);
            AccumulateIntermediateSet(Y, X, ch_bra, ch_ab_YX, ch_ket, ibra, iket,
                                      tbc_ab_YX.GetKetIndex_hh(), &tbc_ab_YX.Ket_unocc_hh, -1.0, mpp);
            AccumulateIntermediateSet(Y, X, ch_bra, ch_ab_YX, ch_ket, ibra, iket,
                                      tbc_ab_YX.GetKetIndex_ph(), &tbc_ab_YX.Ket_unocc_ph, -1.0, mpp);
          }

          Matrixpp(ibra, iket) += mpp;
          Matrixhh(ibra, iket) += mhh;
        }
      }

      if (Z.IsHermitian() && ch_bra == ch_ket)
      {
        Matrixpp += Matrixpp.t();
        Matrixhh += Matrixhh.t();
      }
      else if (Z.IsAntiHermitian() && ch_bra == ch_ket)
      {
        Matrixpp -= Matrixpp.t();
        Matrixhh -= Matrixhh.t();
      }
    }

    X.profiler.timer["EventIMSRG2::BuildScalarMppMhhEvent"] += omp_get_wtime() - t_start;
  }

  void EmitMppMhhTwoBodyEvent(const Operator& X, const Operator& Y, Operator& Z,
                              const TwoBodyME& Mpp, const TwoBodyME& Mhh)
  {
    (void)Y;
    const double t_start = omp_get_wtime();
    EventIndexCache cache(Z);
    std::size_t reserve_hint = 0;
    for (const auto& key : cache.owned_two_body_keys.keys)
    {
      TwoBodyChannel& tbc_bra = Z.modelspace->GetTwoBodyChannel(key[0]);
      TwoBodyChannel& tbc_ket = Z.modelspace->GetTwoBodyChannel(key[1]);
      reserve_hint += tbc_bra.GetNumberKets() * tbc_ket.GetNumberKets();
    }
    auto buffers = MakeTwoBodyBuffers(reserve_hint);

#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ikey = 0; ikey < cache.owned_two_body_keys.keys.size(); ++ikey)
    {
      const int tid = omp_get_thread_num();
      const int ch_bra = static_cast<int>(cache.owned_two_body_keys.keys[ikey][0]);
      const int ch_ket = static_cast<int>(cache.owned_two_body_keys.keys[ikey][1]);
      TwoBodyChannel& tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel& tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      const int nbras = static_cast<int>(tbc_bra.GetNumberKets());
      const int nkets = static_cast<int>(tbc_ket.GetNumberKets());
      const arma::mat& MppMat = Mpp.GetMatrix(ch_bra, ch_ket);
      const arma::mat& MhhMat = Mhh.GetMatrix(ch_bra, ch_ket);
      for (int ibra = 0; ibra < nbras; ++ibra)
      {
        const int ketmin = (ch_bra == ch_ket) ? ibra : 0;
        for (int iket = ketmin; iket < nkets; ++iket)
        {
          const double zijkl = MppMat(ibra, iket) - MhhMat(ibra, iket);
          PushTwoBody(buffers, *Z.modelspace, tid, ch_bra, ch_ket, ibra, iket, zijkl);
        }
      }
    }

    imsrg_mpi::ExchangeAndApplyTwoBodyContributions(Z, buffers);
    X.profiler.timer["EventIMSRG2::comm222_pp_hh_event"] += omp_get_wtime() - t_start;
  }

  void EmitMppMhhOneBodyEvent(const Operator& X, const Operator& Y, Operator& Z,
                              const TwoBodyME& Mpp, const TwoBodyME& Mhh)
  {
    (void)Y;
    const double t_start = omp_get_wtime();
    EventIndexCache cache(Z);
    auto buffers = MakeOneBodyBuffers(cache.all_orbits.size() * cache.all_orbits.size());

#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ii = 0; ii < cache.all_orbits.size(); ++ii)
    {
      const int tid = omp_get_thread_num();
      const int i = static_cast<int>(cache.all_orbits[ii]);
      Orbit& oi = Z.modelspace->GetOrbit(i);
      for (std::size_t j_orb : cache.one_body_targets.targets[i])
      {
        const int j = static_cast<int>(j_orb);
        double zij = 0.0;
        for (std::size_t c : cache.all_orbits)
        {
          Orbit& oc = Z.modelspace->GetOrbit(c);
          const double nc = oc.occ;
          const double nbarc = 1.0 - nc;
          const int Jmin = std::max(std::abs(oc.j2 - oi.j2), std::abs(oc.j2 - Z.modelspace->GetOrbit(j).j2)) / 2;
          const int Jmax = (oc.j2 + std::min(oi.j2, Z.modelspace->GetOrbit(j).j2)) / 2;
          for (int J = Jmin; J <= Jmax; ++J)
          {
            const int parity = (oc.l + oi.l) % 2;
            const int Tz = (oc.tz2 + oi.tz2) / 2;
            const int ch = Z.modelspace->GetTwoBodyChannelIndex(J, parity, Tz);
            if (!OwnsTwoBodyOutput(*Z.modelspace, ch))
              continue;
            if (std::abs(nc) > 1e-9)
              zij += (2 * J + 1) * nc * Mpp.GetTBME_J(J, c, i, c, j);
            if (std::abs(nbarc) > 1e-9)
              zij += (2 * J + 1) * nbarc * Mhh.GetTBME_J(J, c, i, c, j);
          }
        }
        PushOneBody(buffers, tid, i, j, zij / (oi.j2 + 1.0));
      }
    }

    imsrg_mpi::ExchangeAndApplyOneBodyContributions(Z, buffers);
    X.profiler.timer["EventIMSRG2::comm221_event"] += omp_get_wtime() - t_start;
  }

  void comm110_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    const double t_start = omp_get_wtime();
    if (Z.IsAntiHermitian() || Z.GetJRank() > 0 || Z.GetTRank() > 0 || Z.GetParity() != 0)
      return;
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
    EventIndexCache cache(Z);
    const auto& orbit_list = cache.all_orbits;
    double z0 = 0.0;
#pragma omp parallel for reduction(+:z0) schedule(static)
    for (std::size_t ia = 0; ia < orbit_list.size(); ++ia)
    {
      const std::size_t a = orbit_list[ia];
      if (imsrg_mpi::Enabled() && OneBodyOwner(static_cast<int>(a)) != imsrg_mpi::Rank())
        continue;
      Orbit& oa = Z.modelspace->GetOrbit(a);
      for (std::size_t b : orbit_list)
      {
        Orbit& ob = Z.modelspace->GetOrbit(b);
        z0 += (oa.j2 + 1) * oa.occ * (1 - ob.occ) *
              (X1(a, b) * Y1(b, a) - Y1(a, b) * X1(b, a));
      }
    }
    Z.ZeroBody += z0;
    X.profiler.timer["EventIMSRG2::comm110_event"] += omp_get_wtime() - t_start;
  }

  void comm220_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    const double t_start = omp_get_wtime();
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2 || Z.IsAntiHermitian() ||
        Z.GetJRank() > 0 || Z.GetTRank() > 0 || Z.GetParity() != 0)
      return;
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
    EventIndexCache cache(Z);
    const auto& orbit_list = cache.all_orbits;
    double z0 = 0.0;
#pragma omp parallel for reduction(+:z0) schedule(dynamic, 1)
    for (std::size_t ia = 0; ia < orbit_list.size(); ++ia)
    {
      const std::size_t a = orbit_list[ia];
      if (imsrg_mpi::Enabled() && OneBodyOwner(static_cast<int>(a)) != imsrg_mpi::Rank())
        continue;
      Orbit& oa = Z.modelspace->GetOrbit(a);
      for (std::size_t b : orbit_list)
      {
        Orbit& ob = Z.modelspace->GetOrbit(b);
        for (std::size_t c : orbit_list)
        {
          Orbit& oc = Z.modelspace->GetOrbit(c);
          for (std::size_t d : orbit_list)
          {
            Orbit& od = Z.modelspace->GetOrbit(d);
            const double occ = oa.occ * ob.occ * (1 - oc.occ) * (1 - od.occ);
            if (std::abs(occ) < ModelSpace::OCC_CUT)
              continue;
            const int Jmin = std::max(std::abs(oa.j2 - ob.j2), std::abs(oc.j2 - od.j2)) / 2;
            const int Jmax = std::min(oa.j2 + ob.j2, oc.j2 + od.j2) / 2;
            for (int J = Jmin; J <= Jmax; ++J)
            {
              const double xabcd = X2.GetTBME_J(J, J, a, b, c, d);
              const double xcdab = X2.GetTBME_J(J, J, c, d, a, b);
              const double yabcd = Y2.GetTBME_J(J, J, a, b, c, d);
              const double ycdab = Y2.GetTBME_J(J, J, c, d, a, b);
              z0 += 0.25 * (2 * J + 1) * occ * (xabcd * ycdab - yabcd * xcdab);
            }
          }
        }
      }
    }
    Z.ZeroBody += z0;
    X.profiler.timer["EventIMSRG2::comm220_event"] += omp_get_wtime() - t_start;
  }

  void comm111_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    const double t_start = omp_get_wtime();
    auto buffers = MakeOneBodyBuffers();
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
    const auto orbit_list = OrbitList(*Z.modelspace);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ii = 0; ii < orbit_list.size(); ++ii)
    {
      const int tid = omp_get_thread_num();
      const int i = static_cast<int>(orbit_list[ii]);
      if (!OwnsOneBodyRow(i))
        continue;
      for (std::size_t j_orb : orbit_list)
      {
        const int j = static_cast<int>(j_orb);
        double zij = 0.0;
        for (std::size_t a : orbit_list)
          zij += X1(i, a) * Y1(a, j) - Y1(i, a) * X1(a, j);
        PushOneBody(buffers, tid, i, j, zij);
      }
    }
    imsrg_mpi::ExchangeAndApplyOneBodyContributions(Z, buffers);
    X.profiler.timer["EventIMSRG2::comm111_event"] += omp_get_wtime() - t_start;
  }

  void comm121_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    const double t_start = omp_get_wtime();
    if (X.GetParticleRank() < 2 && Y.GetParticleRank() < 2)
      return;
    auto buffers = MakeOneBodyBuffers();
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
    EventIndexCache cache(Z);
    const auto& orbit_list = cache.all_orbits;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ii = 0; ii < orbit_list.size(); ++ii)
    {
      const int tid = omp_get_thread_num();
      const int i = static_cast<int>(orbit_list[ii]);
      if (!OwnsOneBodyRow(i))
        continue;
      Orbit& oi = Z.modelspace->GetOrbit(i);
      for (std::size_t j_orb : cache.one_body_targets.targets[i])
      {
        const int j = static_cast<int>(j_orb);
        Orbit& oj = Z.modelspace->GetOrbit(j);
        double zij = 0.0;
        for (const auto& pair : cache.occ_diff_pairs)
        {
          const std::size_t a = pair.a;
          const std::size_t b = pair.b;
          Orbit& oa = Z.modelspace->GetOrbit(a);
          Orbit& ob = Z.modelspace->GetOrbit(b);
          const int Jmin = std::max(std::abs(ob.j2 - oi.j2), std::abs(oa.j2 - oj.j2)) / 2;
          const int Jmax = std::min(ob.j2 + oi.j2, oa.j2 + oj.j2) / 2;
          for (int J = Jmin; J <= Jmax; ++J)
          {
            const double xbiaj = X2.GetTBME_J(J, J, b, i, a, j);
            const double ybiaj = Y2.GetTBME_J(J, J, b, i, a, j);
            zij += (2 * J + 1) / (oi.j2 + 1.0) * pair.factor * (X1(a, b) * ybiaj - Y1(a, b) * xbiaj);
          }
        }
        PushOneBody(buffers, tid, i, j, zij);
      }
    }
    imsrg_mpi::ExchangeAndApplyOneBodyContributions(Z, buffers);
    X.profiler.timer["EventIMSRG2::comm121_event"] += omp_get_wtime() - t_start;
  }

  void comm221_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    TwoBodyME Mpp;
    TwoBodyME Mhh;
    BuildScalarMppMhhEvent(X, Y, Z, Mpp, Mhh);
    EmitMppMhhOneBodyEvent(X, Y, Z, Mpp, Mhh);
  }

  void comm122_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    const double t_start = omp_get_wtime();
    if (X.GetParticleRank() < 2 && Y.GetParticleRank() < 2)
      return;
    auto buffers = MakeTwoBodyBuffers();
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
    EventIndexCache cache(Z);
    const auto& keys = cache.owned_two_body_keys.keys;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ikey = 0; ikey < keys.size(); ++ikey)
    {
      const int tid = omp_get_thread_num();
      const int ch_bra = static_cast<int>(keys[ikey][0]);
      const int ch_ket = static_cast<int>(keys[ikey][1]);
      TwoBodyChannel& tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel& tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      const int J = tbc_bra.J;
      const int nbras = tbc_bra.GetNumberKets();
      const int nkets = tbc_ket.GetNumberKets();
      for (int ibra = 0; ibra < nbras; ++ibra)
      {
        Ket& bra = tbc_bra.GetKet(ibra);
        const int i = bra.p;
        const int j = bra.q;
        const int ketmin = (ch_bra == ch_ket) ? ibra : 0;
        for (int iket = ketmin; iket < nkets; ++iket)
        {
          Ket& ket = tbc_ket.GetKet(iket);
          const int k = ket.p;
          const int l = ket.q;
          double zijkl = 0.0;
          for (std::size_t a : cache.all_orbits)
          {
            zijkl += X1(i, a) * Y2.GetTBME_J(J, J, a, j, k, l);
            zijkl += X1(j, a) * Y2.GetTBME_J(J, J, i, a, k, l);
            zijkl += X2.GetTBME_J(J, J, i, j, a, l) * Y1(a, k);
            zijkl += X2.GetTBME_J(J, J, i, j, k, a) * Y1(a, l);
            zijkl -= Y1(i, a) * X2.GetTBME_J(J, J, a, j, k, l);
            zijkl -= Y1(j, a) * X2.GetTBME_J(J, J, i, a, k, l);
            zijkl -= Y2.GetTBME_J(J, J, i, j, a, l) * X1(a, k);
            zijkl -= Y2.GetTBME_J(J, J, i, j, k, a) * X1(a, l);
          }
          if (i == j)
            zijkl /= PhysConst::SQRT2;
          if (k == l)
            zijkl /= PhysConst::SQRT2;
          PushTwoBody(buffers, *Z.modelspace, tid, ch_bra, ch_ket, ibra, iket, zijkl);
        }
      }
    }
    imsrg_mpi::ExchangeAndApplyTwoBodyContributions(Z, buffers);
    X.profiler.timer["EventIMSRG2::comm122_event"] += omp_get_wtime() - t_start;
  }

  void comm222_pp_hh_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    TwoBodyME Mpp;
    TwoBodyME Mhh;
    BuildScalarMppMhhEvent(X, Y, Z, Mpp, Mhh);
    EmitMppMhhTwoBodyEvent(X, Y, Z, Mpp, Mhh);
  }

  struct InversePandyaTerm
  {
    int ch_cc;
    int row_cc;
    int col_cc;
    double coefficient;
  };

  struct InversePandyaOutputPlan
  {
    int ch;
    int ibra;
    int iket;
    std::vector<InversePandyaTerm> terms;
  };

  std::vector<InversePandyaOutputPlan> BuildInversePandyaPlan(Operator& Z)
  {
    std::vector<InversePandyaOutputPlan> plan;
    const int nch = static_cast<int>(Z.modelspace->GetNumberTwoBodyChannels());
    for (int ch = 0; ch < nch; ++ch)
    {
      TwoBodyChannel& tbc = Z.modelspace->GetTwoBodyChannel(ch);
      const int J = tbc.J;
      const int nKets = static_cast<int>(tbc.GetNumberKets());
      for (int ibra = 0; ibra < nKets; ++ibra)
      {
        Ket& bra = tbc.GetKet(ibra);
        const int i = bra.p;
        const int j = bra.q;
        Orbit& oi = Z.modelspace->GetOrbit(i);
        Orbit& oj = Z.modelspace->GetOrbit(j);
        const int ketmin = Z.IsHermitian() ? ibra : ibra + 1;
        for (int iket = ketmin; iket < nKets; ++iket)
        {
          Ket& ket = tbc.GetKet(iket);
          const int k = ket.p;
          const int l = ket.q;
          Orbit& ok = Z.modelspace->GetOrbit(k);
          Orbit& ol = Z.modelspace->GetOrbit(l);

          std::vector<InversePandyaTerm> commij_terms;
          std::vector<InversePandyaTerm> commji_terms;

          int parity_cc = (oi.l + ol.l) % 2;
          int Tz_cc = std::abs(oi.tz2 - ol.tz2) / 2;
          int Jpmin = std::max(std::abs(oi.j2 - ol.j2), std::abs(ok.j2 - oj.j2)) / 2;
          int Jpmax = std::min(oi.j2 + ol.j2, ok.j2 + oj.j2) / 2;
          for (int Jprime = Jpmin; Jprime <= Jpmax; ++Jprime)
          {
            const double sixj = Z.modelspace->GetCachedSixJ(oi.j2, oj.j2, J, ok.j2, ol.j2, Jprime);
            if (std::abs(sixj) < 1e-8)
              continue;
            const int ch_cc = Z.modelspace->GetTwoBodyChannelIndex(Jprime, parity_cc, Tz_cc);
            TwoBodyChannel_CC& tbc_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_cc);
            const int nkets_cc = static_cast<int>(tbc_cc.GetNumberKets());
            const int row = static_cast<int>(tbc_cc.GetLocalIndex(std::min(i, l), std::max(i, l))) + (i > l ? nkets_cc : 0);
            const int col = static_cast<int>(tbc_cc.GetLocalIndex(std::min(j, k), std::max(j, k))) + (k > j ? nkets_cc : 0);
            if (row >= 0 && row < nkets_cc && col >= 0 && col < 2 * nkets_cc)
              commij_terms.push_back({ch_cc, row, col, -(2 * Jprime + 1) * sixj});
          }

          if (k == l)
          {
            commji_terms = commij_terms;
          }
          else if (i == j)
          {
            const int phase = Z.modelspace->phase((oi.j2 + oj.j2 + ok.j2 + ol.j2) / 2);
            commji_terms = commij_terms;
            for (auto& term : commji_terms)
              term.coefficient *= phase;
          }
          else
          {
            parity_cc = (oi.l + ok.l) % 2;
            Tz_cc = std::abs(oi.tz2 - ok.tz2) / 2;
            Jpmin = std::max(std::abs(oj.j2 - ol.j2), std::abs(ok.j2 - oi.j2)) / 2;
            Jpmax = std::min(oj.j2 + ol.j2, ok.j2 + oi.j2) / 2;
            for (int Jprime = Jpmin; Jprime <= Jpmax; ++Jprime)
            {
              const double sixj = Z.modelspace->GetCachedSixJ(oj.j2, oi.j2, J, ok.j2, ol.j2, Jprime);
              if (std::abs(sixj) < 1e-8)
                continue;
              const int ch_cc = Z.modelspace->GetTwoBodyChannelIndex(Jprime, parity_cc, Tz_cc);
              TwoBodyChannel_CC& tbc_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_cc);
              const int nkets_cc = static_cast<int>(tbc_cc.GetNumberKets());
              const int row = static_cast<int>(tbc_cc.GetLocalIndex(std::min(i, k), std::max(i, k))) + (i > k ? nkets_cc : 0);
              const int col = static_cast<int>(tbc_cc.GetLocalIndex(std::min(l, j), std::max(l, j))) + (l > j ? nkets_cc : 0);
              if (row >= 0 && row < nkets_cc && col >= 0 && col < 2 * nkets_cc)
                commji_terms.push_back({ch_cc, row, col, -(2 * Jprime + 1) * sixj});
            }
          }

          const double norm = bra.delta_pq() == ket.delta_pq() ? 1 + bra.delta_pq() : PhysConst::SQRT2;
          const int phase_kl = Z.modelspace->phase((ok.j2 + ol.j2) / 2 - J);
          InversePandyaOutputPlan output{ch, ibra, iket, {}};
          output.terms.reserve(commij_terms.size() + commji_terms.size());
          for (const auto& term : commij_terms)
            output.terms.push_back({term.ch_cc, term.row_cc, term.col_cc, -term.coefficient / norm});
          for (const auto& term : commji_terms)
            output.terms.push_back({term.ch_cc, term.row_cc, term.col_cc, phase_kl * term.coefficient / norm});
          if (!output.terms.empty())
            plan.push_back(std::move(output));
        }
      }
    }
    return plan;
  }

  void BuildManualZbarForChannel(const Operator& X, const Operator& Y, Operator& Z,
                                 int ch, arma::mat& Zbar_ch)
  {
    TwoBodyChannel_CC& tbc_cc = Z.modelspace->GetTwoBodyChannel_CC(ch);
    const int nKets_cc = static_cast<int>(tbc_cc.GetNumberKets());
    const std::size_t nph_kets = tbc_cc.GetKetIndex_hh().size() + tbc_cc.GetKetIndex_ph().size();
    Zbar_ch.zeros(nKets_cc, 2 * nKets_cc);
    if (nKets_cc == 0 || nph_kets == 0)
      return;

    arma::mat Y_bar_ph;
    arma::mat Xt_bar_ph;
    Commutator::DoPandyaTransformation_SingleChannel_XandY(X, Y, Xt_bar_ph, Y_bar_ph, ch);
    if (Y_bar_ph.n_elem == 0 || Xt_bar_ph.n_elem == 0)
      return;

    const int hy = Y.IsHermitian() ? 1 : -1;
    std::vector<int> ket_phase(nKets_cc, 1);
    for (int iket = 0; iket < nKets_cc; ++iket)
    {
      const Ket& ket = tbc_cc.GetKet(iket);
      if (Z.modelspace->phase((ket.op->j2 + ket.oq->j2) / 2) < 0)
        ket_phase[iket] = -1;
    }
    arma::uvec phkets = arma::join_cols(tbc_cc.GetKetIndex_hh(), tbc_cc.GetKetIndex_ph());

    for (int row = 0; row < nKets_cc; ++row)
    {
      for (int col = 0; col < nKets_cc; ++col)
      {
        double direct = 0.0;
        double flipped = 0.0;
        for (std::size_t iph = 0; iph < 2 * nph_kets; ++iph)
        {
          direct += Xt_bar_ph(row, iph) * Y_bar_ph(iph, col);
          const std::size_t ph_index = iph < nph_kets ? iph : iph - nph_kets;
          const std::size_t source = iph < nph_kets ? iph + nph_kets : iph - nph_kets;
          const int phase_y = ket_phase[phkets(ph_index)] * ket_phase[col] * hy;
          flipped += Xt_bar_ph(row, iph) * Y_bar_ph(source, col) * phase_y;
        }
        Zbar_ch(row, col) = direct;
        Zbar_ch(row, nKets_cc + col) = flipped;
      }
    }

    arma::mat head = Zbar_ch.head_cols(nKets_cc);
    if (Z.IsHermitian() && X.IsHermitian() != Y.IsHermitian())
      Zbar_ch.head_cols(nKets_cc) += head.t();
    else
      Zbar_ch.head_cols(nKets_cc) -= head.t();

    arma::mat tail = Zbar_ch.tail_cols(nKets_cc);
    for (int row = 0; row < nKets_cc; ++row)
      for (int col = 0; col < nKets_cc; ++col)
        Zbar_ch(row, nKets_cc + col) += tail(col, row) * ket_phase[row] * ket_phase[col];
  }

  void EmitInversePandyaEvents(Operator& Z,
                               const std::deque<arma::mat>& Zbar,
                               const std::vector<char>& active_cc,
                               const std::vector<InversePandyaOutputPlan>& inverse_plan,
                               std::vector<std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>>& buffers)
  {
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t iplan = 0; iplan < inverse_plan.size(); ++iplan)
    {
      const int tid = omp_get_thread_num();
      const auto& output = inverse_plan[iplan];
      double zijkl = 0.0;
      for (const auto& term : output.terms)
      {
        if (term.ch_cc < 0 || term.ch_cc >= static_cast<int>(active_cc.size()) || !active_cc[term.ch_cc])
          continue;
        const arma::mat& Zbar_ch = Zbar[term.ch_cc];
        if (Zbar_ch.n_elem == 0)
          continue;
        zijkl += term.coefficient * Zbar_ch(term.row_cc, term.col_cc);
      }
      PushTwoBody(buffers, *Z.modelspace, tid, output.ch, output.ch, output.ibra, output.iket, zijkl);
    }
  }

  void comm222_ph_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    const double t_start = omp_get_wtime();
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    const auto inverse_plan = BuildInversePandyaPlan(Z);
    auto buffers = MakeTwoBodyBuffers(inverse_plan.size());
    const std::size_t nch = Z.modelspace->GetNumberTwoBodyChannels_CC();
    const std::size_t batch_size = GetPandyaBatchSizeEvent();
    const int nranks = imsrg_mpi::Size();
    std::vector<std::size_t> local_channels;
    std::vector<std::size_t> owned_counts(nranks, 0);

    for (int ch_cc : Z.modelspace->SortedTwoBodyChannels_CC)
    {
      const int owner = imsrg_mpi::Enabled() ? imsrg_mpi::CrossCoupledChannelOwner(*Z.modelspace, ch_cc) : 0;
      owned_counts[owner] += 1;
      if (owner == imsrg_mpi::Rank())
        local_channels.push_back(static_cast<std::size_t>(ch_cc));
    }

    std::size_t max_batches = 0;
    for (std::size_t count : owned_counts)
      max_batches = std::max(max_batches, (count + batch_size - 1) / batch_size);

    Operator& X_mut = const_cast<Operator&>(X);
    Operator& Y_mut = const_cast<Operator&>(Y);
    for (std::size_t ibatch = 0; ibatch < max_batches; ++ibatch)
    {
      const std::size_t first = ibatch * batch_size;
      const std::size_t last = std::min(first + batch_size, local_channels.size());
      std::vector<std::size_t> batch_channels;
      std::vector<char> active_cc(nch, 0);
      std::vector<imsrg_mpi::TwoBodyElementRequest> requests;
      for (std::size_t i = first; i < last; ++i)
      {
        const std::size_t ch = local_channels[i];
        batch_channels.push_back(ch);
        active_cc[ch] = 1;
        auto ch_requests = Commutator::GetPandyaElementRequestsForChannel(X, Y, Z, ch);
        requests.insert(requests.end(), ch_requests.begin(), ch_requests.end());
      }

      imsrg_mpi::PrefetchTwoBodyMatrixElements(X_mut, requests);
      imsrg_mpi::PrefetchTwoBodyMatrixElements(Y_mut, requests);

      std::deque<arma::mat> Zbar(nch);
      for (std::size_t ch : batch_channels)
        BuildManualZbarForChannel(X, Y, Z, static_cast<int>(ch), Zbar[ch]);

      EmitInversePandyaEvents(Z, Zbar, active_cc, inverse_plan, buffers);

      imsrg_mpi::ClearTwoBodyCache(X_mut);
      imsrg_mpi::ClearTwoBodyCache(Y_mut);
    }

    imsrg_mpi::ExchangeAndApplyTwoBodyContributions(Z, buffers);
    X.profiler.timer["EventIMSRG2::comm222_ph_event"] += omp_get_wtime() - t_start;
  }
}
}
