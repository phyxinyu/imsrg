#include "Commutator.hh"
#include "MpiSupport.hh"
#include "PhysicalConstants.hh"
#include "AngMom.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <omp.h>

namespace Commutator
{
namespace EventIMSRG2
{
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

  std::vector<std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>> MakeTwoBodyBuffers()
  {
    return std::vector<std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>>(
      omp_get_max_threads(), std::vector<std::vector<imsrg_mpi::TwoBodyContribution>>(imsrg_mpi::Size()));
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

  void comm110_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    const double t_start = omp_get_wtime();
    if (Z.IsAntiHermitian() || Z.GetJRank() > 0 || Z.GetTRank() > 0 || Z.GetParity() != 0)
      return;
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
    const auto orbit_list = OrbitList(*Z.modelspace);
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
    const auto orbit_list = OrbitList(*Z.modelspace);
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
    const auto orbit_list = OrbitList(*Z.modelspace);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ii = 0; ii < orbit_list.size(); ++ii)
    {
      const int tid = omp_get_thread_num();
      const int i = static_cast<int>(orbit_list[ii]);
      if (!OwnsOneBodyRow(i))
        continue;
      Orbit& oi = Z.modelspace->GetOrbit(i);
      for (std::size_t j_orb : Z.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
      {
        const int j = static_cast<int>(j_orb);
        Orbit& oj = Z.modelspace->GetOrbit(j);
        double zij = 0.0;
        for (std::size_t a : orbit_list)
        {
          Orbit& oa = Z.modelspace->GetOrbit(a);
          for (std::size_t b : orbit_list)
          {
            Orbit& ob = Z.modelspace->GetOrbit(b);
            const double occ = oa.occ - ob.occ;
            if (std::abs(occ) < ModelSpace::OCC_CUT)
              continue;
            const int Jmin = std::max(std::abs(ob.j2 - oi.j2), std::abs(oa.j2 - oj.j2)) / 2;
            const int Jmax = std::min(ob.j2 + oi.j2, oa.j2 + oj.j2) / 2;
            for (int J = Jmin; J <= Jmax; ++J)
            {
              const double xbiaj = X2.GetTBME_J(J, J, b, i, a, j);
              const double ybiaj = Y2.GetTBME_J(J, J, b, i, a, j);
              zij += (2 * J + 1) / (oi.j2 + 1.0) * occ * (X1(a, b) * ybiaj - Y1(a, b) * xbiaj);
            }
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
    const double t_start = omp_get_wtime();
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    auto buffers = MakeOneBodyBuffers();
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
    const auto orbit_list = OrbitList(*Z.modelspace);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ii = 0; ii < orbit_list.size(); ++ii)
    {
      const int tid = omp_get_thread_num();
      const int i = static_cast<int>(orbit_list[ii]);
      if (!OwnsOneBodyRow(i))
        continue;
      Orbit& oi = Z.modelspace->GetOrbit(i);
      for (std::size_t j_orb : Z.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
      {
        const int j = static_cast<int>(j_orb);
        double zij = 0.0;
        for (std::size_t a : orbit_list)
        {
          Orbit& oa = Z.modelspace->GetOrbit(a);
          for (std::size_t b : orbit_list)
          {
            Orbit& ob = Z.modelspace->GetOrbit(b);
            for (std::size_t c : orbit_list)
            {
              Orbit& oc = Z.modelspace->GetOrbit(c);
              const double occ = oa.occ * ob.occ * (1 - oc.occ) +
                                 (1 - oa.occ) * (1 - ob.occ) * oc.occ;
              if (std::abs(occ) < ModelSpace::OCC_CUT)
                continue;
              const int Jmin = std::max(std::abs(ob.j2 - oa.j2), std::abs(oc.j2 - oi.j2)) / 2;
              const int Jmax = std::min(ob.j2 + oa.j2, oc.j2 + oi.j2) / 2;
              for (int J = Jmin; J <= Jmax; ++J)
              {
                const double xciab = X2.GetTBME_J(J, J, c, i, a, b);
                const double yciab = Y2.GetTBME_J(J, J, c, i, a, b);
                const double xabcj = X2.GetTBME_J(J, J, a, b, c, j);
                const double yabcj = Y2.GetTBME_J(J, J, a, b, c, j);
                zij += 0.5 * (2 * J + 1) / (oi.j2 + 1.0) * occ *
                       (xciab * yabcj - yciab * xabcj);
              }
            }
          }
        }
        PushOneBody(buffers, tid, i, j, zij);
      }
    }
    imsrg_mpi::ExchangeAndApplyOneBodyContributions(Z, buffers);
    X.profiler.timer["EventIMSRG2::comm221_event"] += omp_get_wtime() - t_start;
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
    std::vector<std::array<std::size_t, 2>> keys;
    for (const auto& iter : Z.TwoBody.MatEl)
      keys.push_back(iter.first);
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
          for (std::size_t a : Z.modelspace->all_orbits)
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
    const double t_start = omp_get_wtime();
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    auto buffers = MakeTwoBodyBuffers();
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
    std::vector<std::array<std::size_t, 2>> keys;
    for (const auto& iter : Z.TwoBody.MatEl)
      keys.push_back(iter.first);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ikey = 0; ikey < keys.size(); ++ikey)
    {
      const int tid = omp_get_thread_num();
      const int ch_bra = static_cast<int>(keys[ikey][0]);
      const int ch_ket = static_cast<int>(keys[ikey][1]);
      if (!OwnsTwoBodyOutput(*Z.modelspace, ch_bra))
        continue;
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
          for (std::size_t a : Z.modelspace->all_orbits)
          {
            for (std::size_t b : Z.modelspace->all_orbits)
            {
              if (b < a)
                continue;
              Orbit& oa = Z.modelspace->GetOrbit(a);
              Orbit& ob = Z.modelspace->GetOrbit(b);
              const double occ = (1 - oa.occ) * (1 - ob.occ) - oa.occ * ob.occ;
              if (std::abs(occ) < ModelSpace::OCC_CUT)
                continue;
              const double xijab = X2.GetTBME_J(J, J, i, j, a, b);
              const double yijab = Y2.GetTBME_J(J, J, i, j, a, b);
              const double xabkl = X2.GetTBME_J(J, J, a, b, k, l);
              const double yabkl = Y2.GetTBME_J(J, J, a, b, k, l);
              const double flip_factor = (a == b) ? 1.0 : 2.0;
              zijkl += 0.5 * flip_factor * occ * (xijab * yabkl - yijab * xabkl);
            }
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
    X.profiler.timer["EventIMSRG2::comm222_pp_hh_event"] += omp_get_wtime() - t_start;
  }

  void comm222_ph_event(const Operator& X, const Operator& Y, Operator& Z)
  {
    const double t_start = omp_get_wtime();
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    auto buffers = MakeTwoBodyBuffers();
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
    std::vector<std::array<std::size_t, 2>> keys;
    for (const auto& iter : Z.TwoBody.MatEl)
      keys.push_back(iter.first);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ikey = 0; ikey < keys.size(); ++ikey)
    {
      const int tid = omp_get_thread_num();
      const int ch_bra = static_cast<int>(keys[ikey][0]);
      const int ch_ket = static_cast<int>(keys[ikey][1]);
      if (!OwnsTwoBodyOutput(*Z.modelspace, ch_bra))
        continue;
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
        Orbit& oi = Z.modelspace->GetOrbit(i);
        Orbit& oj = Z.modelspace->GetOrbit(j);
        const int ketmin = (ch_bra == ch_ket) ? ibra : 0;
        for (int iket = ketmin; iket < nkets; ++iket)
        {
          Ket& ket = tbc_ket.GetKet(iket);
          const int k = ket.p;
          const int l = ket.q;
          Orbit& ok = Z.modelspace->GetOrbit(k);
          Orbit& ol = Z.modelspace->GetOrbit(l);
          double zijkl = 0.0;
          const int Jpmin = std::min(std::max(std::abs(oi.j2 - ol.j2), std::abs(oj.j2 - ok.j2)),
                                     std::max(std::abs(oj.j2 - ol.j2), std::abs(oi.j2 - ok.j2))) / 2;
          const int Jpmax = std::max(std::min(oi.j2 + ol.j2, oj.j2 + ok.j2),
                                     std::min(oj.j2 + ol.j2, oi.j2 + ok.j2)) / 2;
          for (int Jp = Jpmin; Jp <= Jpmax; ++Jp)
          {
            const int ch_cc_direct = Z.modelspace->GetTwoBodyChannelIndex(
              Jp, (oi.l + ol.l) % 2, std::abs(oi.tz2 - ol.tz2) / 2);
            const int ch_cc_exchange = Z.modelspace->GetTwoBodyChannelIndex(
              Jp, (oj.l + ol.l) % 2, std::abs(oj.tz2 - ol.tz2) / 2);
            const bool do_direct = !imsrg_mpi::Enabled() ||
                                   imsrg_mpi::OwnsCrossCoupledChannel(*Z.modelspace, ch_cc_direct);
            const bool do_exchange = !imsrg_mpi::Enabled() ||
                                     imsrg_mpi::OwnsCrossCoupledChannel(*Z.modelspace, ch_cc_exchange);
            if (!do_direct && !do_exchange)
              continue;

            double zbar_ilkj = 0.0;
            double zbar_jlki = 0.0;
            for (std::size_t a : Z.modelspace->all_orbits)
            {
              Orbit& oa = Z.modelspace->GetOrbit(a);
              for (std::size_t b : Z.modelspace->all_orbits)
              {
                Orbit& ob = Z.modelspace->GetOrbit(b);
                const double occ = oa.occ - ob.occ;
                if (std::abs(occ) < ModelSpace::OCC_CUT)
                  continue;

                if (do_direct)
                {
                  double xbar_ilab = 0.0;
                  double ybar_ilab = 0.0;
                  int Jppmin = std::max(std::abs(oi.j2 - ob.j2), std::abs(oa.j2 - ol.j2)) / 2;
                  int Jppmax = std::min(oi.j2 + ob.j2, oa.j2 + ol.j2) / 2;
                  for (int Jpp = Jppmin; Jpp <= Jppmax; ++Jpp)
                  {
                    const double sixj = AngMom::SixJ(oi.j2 * 0.5, ol.j2 * 0.5, Jp, oa.j2 * 0.5, ob.j2 * 0.5, Jpp);
                    xbar_ilab -= (2 * Jpp + 1) * sixj * X2.GetTBME_J(Jpp, i, b, a, l);
                    ybar_ilab -= (2 * Jpp + 1) * sixj * Y2.GetTBME_J(Jpp, i, b, a, l);
                  }

                  double xbar_abkj = 0.0;
                  double ybar_abkj = 0.0;
                  Jppmin = std::max(std::abs(oa.j2 - oj.j2), std::abs(ok.j2 - ob.j2)) / 2;
                  Jppmax = std::min(oa.j2 + oj.j2, ok.j2 + ob.j2) / 2;
                  for (int Jpp = Jppmin; Jpp <= Jppmax; ++Jpp)
                  {
                    const double sixj = AngMom::SixJ(oa.j2 * 0.5, ob.j2 * 0.5, Jp, ok.j2 * 0.5, oj.j2 * 0.5, Jpp);
                    xbar_abkj -= (2 * Jpp + 1) * sixj * X2.GetTBME_J(Jpp, a, j, k, b);
                    ybar_abkj -= (2 * Jpp + 1) * sixj * Y2.GetTBME_J(Jpp, a, j, k, b);
                  }
                  zbar_ilkj += occ * (xbar_ilab * ybar_abkj - ybar_ilab * xbar_abkj);
                }

                if (do_exchange)
                {
                  double xbar_jlab = 0.0;
                  double ybar_jlab = 0.0;
                  int Jppmin = std::max(std::abs(oj.j2 - ob.j2), std::abs(oa.j2 - ol.j2)) / 2;
                  int Jppmax = std::min(oj.j2 + ob.j2, oa.j2 + ol.j2) / 2;
                  for (int Jpp = Jppmin; Jpp <= Jppmax; ++Jpp)
                  {
                    const double sixj = AngMom::SixJ(oj.j2 * 0.5, ol.j2 * 0.5, Jp, oa.j2 * 0.5, ob.j2 * 0.5, Jpp);
                    xbar_jlab -= (2 * Jpp + 1) * sixj * X2.GetTBME_J(Jpp, j, b, a, l);
                    ybar_jlab -= (2 * Jpp + 1) * sixj * Y2.GetTBME_J(Jpp, j, b, a, l);
                  }

                  double xbar_abki = 0.0;
                  double ybar_abki = 0.0;
                  Jppmin = std::max(std::abs(oa.j2 - oi.j2), std::abs(ok.j2 - ob.j2)) / 2;
                  Jppmax = std::min(oa.j2 + oi.j2, ok.j2 + ob.j2) / 2;
                  for (int Jpp = Jppmin; Jpp <= Jppmax; ++Jpp)
                  {
                    const double sixj = AngMom::SixJ(oa.j2 * 0.5, ob.j2 * 0.5, Jp, ok.j2 * 0.5, oi.j2 * 0.5, Jpp);
                    xbar_abki -= (2 * Jpp + 1) * sixj * X2.GetTBME_J(Jpp, a, i, k, b);
                    ybar_abki -= (2 * Jpp + 1) * sixj * Y2.GetTBME_J(Jpp, a, i, k, b);
                  }
                  zbar_jlki += occ * (xbar_jlab * ybar_abki - ybar_jlab * xbar_abki);
                }
              }
            }
            const double sixj_ijkl = AngMom::SixJ(oi.j2 * 0.5, oj.j2 * 0.5, J, ok.j2 * 0.5, ol.j2 * 0.5, Jp);
            const double sixj_jikl = AngMom::SixJ(oj.j2 * 0.5, oi.j2 * 0.5, J, ok.j2 * 0.5, ol.j2 * 0.5, Jp);
            const int phase_ij = AngMom::phase((oi.j2 + oj.j2 - 2 * J) / 2);
            if (do_direct)
              zijkl += (2 * Jp + 1) * sixj_ijkl * zbar_ilkj;
            if (do_exchange)
              zijkl -= (2 * Jp + 1) * sixj_jikl * zbar_jlki * phase_ij;
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
    X.profiler.timer["EventIMSRG2::comm222_ph_event"] += omp_get_wtime() - t_start;
  }
}
}
