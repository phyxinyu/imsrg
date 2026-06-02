#include "StochasticEventIMSRG2.hh"
#include "Commutator.hh"
#include "IMSRGProfiler.hh"
#include "MpiSupport.hh"
#include "PhysicalConstants.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace Commutator
{
  std::vector<imsrg_mpi::TwoBodyElementRequest> GetPandyaElementRequestsForChannel(
    const Operator& X, const Operator& Y, const Operator& Z, std::size_t ch);
}

namespace
{
  enum TermId
  {
    TERM_COMM111 = 111,
    TERM_COMM121 = 121,
    TERM_COMM122 = 122,
    TERM_COMM221 = 221,
    TERM_COMM222_PPHH = 2221,
    TERM_COMM222_PH = 2222
  };

  struct OccPair
  {
    std::size_t a;
    std::size_t b;
    double factor;
  };

  struct IndexCache
  {
    std::vector<std::size_t> all_orbits;
    std::vector<OccPair> occ_diff_pairs;
    std::vector<std::vector<std::size_t>> one_body_targets;
    std::vector<std::array<std::size_t, 2>> owned_two_body_keys;
    std::vector<std::array<std::size_t, 2>> scalar_two_body_keys;

    explicit IndexCache(const Operator& H)
    {
      ModelSpace& modelspace = *H.GetModelSpace();
      all_orbits.assign(modelspace.all_orbits.begin(), modelspace.all_orbits.end());

      one_body_targets.resize(modelspace.GetNumberOrbits());
      for (std::size_t i : all_orbits)
      {
        Orbit& oi = modelspace.GetOrbit(i);
        auto& targets = one_body_targets[i];
        for (std::size_t j : H.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
          if (i <= j)
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

      for (const auto& iter : H.TwoBody.MatEl)
      {
        scalar_two_body_keys.push_back(iter.first);
        if (!imsrg_mpi::OwnerOnlyStorageEnabled() ||
            imsrg_mpi::TwoBodyChannelOwner(modelspace, iter.first[0]) == imsrg_mpi::Rank())
        {
          owned_two_body_keys.push_back(iter.first);
        }
      }
    }
  };

  std::uint64_t SplitMix64(std::uint64_t x)
  {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  void HashCombine(std::uint64_t& seed, std::uint64_t value)
  {
    seed = SplitMix64(seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2)));
  }

  double UnitRandom(std::uint64_t key)
  {
    constexpr double inv_two_to_53 = 1.0 / 9007199254740992.0;
    return static_cast<double>(SplitMix64(key) >> 11) * inv_two_to_53;
  }

  std::uint64_t MixSigned(std::int64_t value)
  {
    return static_cast<std::uint64_t>(value) ^ 0x8000000000000000ULL;
  }

  std::int64_t StochasticRoundExpected(double expected_count, std::uint64_t key)
  {
    if (expected_count == 0.0)
      return 0;
    const double scaled = std::abs(expected_count);
    const double base_as_double = std::floor(scaled);
    if (base_as_double > static_cast<double>(std::numeric_limits<std::int64_t>::max() - 1))
      throw std::runtime_error("stochastic source spawn delta count overflow.");
    std::int64_t magnitude = static_cast<std::int64_t>(base_as_double);
    if (UnitRandom(key) < scaled - base_as_double)
      ++magnitude;
    return expected_count < 0.0 ? -magnitude : magnitude;
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
    return !imsrg_mpi::OwnerOnlyStorageEnabled() ||
           imsrg_mpi::TwoBodyChannelOwner(modelspace, ch_bra) == imsrg_mpi::Rank();
  }

  std::size_t GetPandyaBatchSize()
  {
    const char* env_value = std::getenv("IMSRG_MPI_PANDYA_CC_BATCH");
    if (env_value != nullptr && env_value[0] != '\0')
    {
      char* endptr = nullptr;
      const unsigned long parsed = std::strtoul(env_value, &endptr, 10);
      if (endptr != env_value && parsed > 0)
        return static_cast<std::size_t>(parsed);
    }
    return 4;
  }

  typedef std::vector<std::vector<std::vector<imsrg_mpi::OneBodyDeltaContribution>>> OneBodyDeltaBuffers;
  typedef std::vector<std::vector<std::vector<imsrg_mpi::TwoBodyDeltaContribution>>> TwoBodyDeltaBuffers;

  OneBodyDeltaBuffers MakeOneBodyDeltaBuffers()
  {
    return OneBodyDeltaBuffers(
      omp_get_max_threads(), std::vector<std::vector<imsrg_mpi::OneBodyDeltaContribution>>(imsrg_mpi::Size()));
  }

  TwoBodyDeltaBuffers MakeTwoBodyDeltaBuffers()
  {
    return TwoBodyDeltaBuffers(
      omp_get_max_threads(), std::vector<std::vector<imsrg_mpi::TwoBodyDeltaContribution>>(imsrg_mpi::Size()));
  }

  struct SpawnContext
  {
    const stochastic_imsrg::HamiltonianWalkerState& state;
    double ds;
    int step;
    OneBodyDeltaBuffers& one_body_buffers;
    TwoBodyDeltaBuffers& two_body_buffers;
  };

  std::uint64_t SpawnKey(const SpawnContext& ctx, int term_id, int body_id, int piece_id,
                         std::initializer_list<std::int64_t> target,
                         std::initializer_list<std::int64_t> source)
  {
    std::uint64_t key = ctx.state.seed;
    HashCombine(key, static_cast<std::uint64_t>(ctx.step));
    HashCombine(key, static_cast<std::uint64_t>(term_id));
    HashCombine(key, static_cast<std::uint64_t>(body_id));
    HashCombine(key, static_cast<std::uint64_t>(piece_id));
    for (auto value : target)
      HashCombine(key, MixSigned(value));
    HashCombine(key, 0x9ddfea08eb382d69ULL);
    for (auto value : source)
      HashCombine(key, MixSigned(value));
    return key;
  }

  void SpawnOneBodyDelta(SpawnContext& ctx, int tid, int term_id, int piece_id,
                         int i, int j, double value,
                         std::initializer_list<std::int64_t> source)
  {
    if (i > j || value == 0.0)
      return;
    const double expected = ctx.ds * value / ctx.state.quantum;
    const std::uint64_t key = SpawnKey(ctx, term_id, 1, piece_id, {i, j}, source);
    const std::int64_t delta = StochasticRoundExpected(expected, key);
    if (delta != 0)
      ctx.one_body_buffers[tid][OneBodyOwner(i)].push_back({i, j, delta});
  }

  void SpawnTwoBodyDelta(SpawnContext& ctx, ModelSpace& modelspace, int tid, int term_id, int piece_id,
                         int ch_bra, int ch_ket, int ibra, int iket, double value,
                         std::initializer_list<std::int64_t> source)
  {
    if (ch_bra == ch_ket && ibra > iket)
      return;
    if (value == 0.0)
      return;
    const double expected = ctx.ds * value / ctx.state.quantum;
    const std::uint64_t key = SpawnKey(ctx, term_id, 2, piece_id,
                                       {ch_bra, ch_ket, ibra, iket}, source);
    const std::int64_t delta = StochasticRoundExpected(expected, key);
    if (delta != 0)
      ctx.two_body_buffers[tid][TwoBodyOwner(modelspace, ch_bra)].push_back(
        {ch_bra, ch_ket, ibra, iket, delta});
  }

  double MatrixElement(const Operator& op, int ch_bra, int ch_ket, int i, int j)
  {
    if (ch_bra <= ch_ket)
      return op.TwoBody.GetMatrix(ch_bra, ch_ket)(i, j);
    const int h = op.IsHermitian() ? 1 : -1;
    return h * op.TwoBody.GetMatrix(ch_ket, ch_bra)(j, i);
  }

  double comm110_zero(const Operator& X, const Operator& Y, const IndexCache& cache)
  {
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
    double z0 = 0.0;
#pragma omp parallel for reduction(+:z0) schedule(static)
    for (std::size_t ia = 0; ia < cache.all_orbits.size(); ++ia)
    {
      const std::size_t a = cache.all_orbits[ia];
      if (imsrg_mpi::Enabled() && OneBodyOwner(static_cast<int>(a)) != imsrg_mpi::Rank())
        continue;
      Orbit& oa = X.GetModelSpace()->GetOrbit(a);
      for (std::size_t b : cache.all_orbits)
      {
        Orbit& ob = X.GetModelSpace()->GetOrbit(b);
        z0 += (oa.j2 + 1) * oa.occ * (1 - ob.occ) *
              (X1(a, b) * Y1(b, a) - Y1(a, b) * X1(b, a));
      }
    }
    return z0;
  }

  double comm220_zero(const Operator& X, const Operator& Y, const IndexCache& cache)
  {
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return 0.0;
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
    double z0 = 0.0;
#pragma omp parallel for reduction(+:z0) schedule(dynamic, 1)
    for (std::size_t ia = 0; ia < cache.all_orbits.size(); ++ia)
    {
      const std::size_t a = cache.all_orbits[ia];
      if (imsrg_mpi::Enabled() && OneBodyOwner(static_cast<int>(a)) != imsrg_mpi::Rank())
        continue;
      Orbit& oa = X.GetModelSpace()->GetOrbit(a);
      for (std::size_t b : cache.all_orbits)
      {
        Orbit& ob = X.GetModelSpace()->GetOrbit(b);
        for (std::size_t c : cache.all_orbits)
        {
          Orbit& oc = X.GetModelSpace()->GetOrbit(c);
          for (std::size_t d : cache.all_orbits)
          {
            Orbit& od = X.GetModelSpace()->GetOrbit(d);
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
    return z0;
  }

  void comm111_spawn(const Operator& X, const Operator& Y, const IndexCache& cache, SpawnContext& ctx)
  {
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ii = 0; ii < cache.all_orbits.size(); ++ii)
    {
      const int tid = omp_get_thread_num();
      const int i = static_cast<int>(cache.all_orbits[ii]);
      if (!OwnsOneBodyRow(i))
        continue;
      for (std::size_t j_orb : cache.one_body_targets[i])
      {
        const int j = static_cast<int>(j_orb);
        for (std::size_t a : cache.all_orbits)
        {
          const double value = X1(i, a) * Y1(a, j) - Y1(i, a) * X1(a, j);
          SpawnOneBodyDelta(ctx, tid, TERM_COMM111, 0, i, j, value, {static_cast<std::int64_t>(a)});
        }
      }
    }
  }

  void comm121_spawn(const Operator& X, const Operator& Y, const IndexCache& cache, SpawnContext& ctx)
  {
    if (X.GetParticleRank() < 2 && Y.GetParticleRank() < 2)
      return;
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ii = 0; ii < cache.all_orbits.size(); ++ii)
    {
      const int tid = omp_get_thread_num();
      const int i = static_cast<int>(cache.all_orbits[ii]);
      if (!OwnsOneBodyRow(i))
        continue;
      Orbit& oi = X.GetModelSpace()->GetOrbit(i);
      for (std::size_t j_orb : cache.one_body_targets[i])
      {
        const int j = static_cast<int>(j_orb);
        Orbit& oj = X.GetModelSpace()->GetOrbit(j);
        for (const auto& pair : cache.occ_diff_pairs)
        {
          const std::size_t a = pair.a;
          const std::size_t b = pair.b;
          Orbit& oa = X.GetModelSpace()->GetOrbit(a);
          Orbit& ob = X.GetModelSpace()->GetOrbit(b);
          const int Jmin = std::max(std::abs(ob.j2 - oi.j2), std::abs(oa.j2 - oj.j2)) / 2;
          const int Jmax = std::min(ob.j2 + oi.j2, oa.j2 + oj.j2) / 2;
          for (int J = Jmin; J <= Jmax; ++J)
          {
            const double xbiaj = X2.GetTBME_J(J, J, b, i, a, j);
            const double ybiaj = Y2.GetTBME_J(J, J, b, i, a, j);
            const double value = (2 * J + 1) / (oi.j2 + 1.0) *
                                 pair.factor * (X1(a, b) * ybiaj - Y1(a, b) * xbiaj);
            SpawnOneBodyDelta(ctx, tid, TERM_COMM121, 0, i, j, value,
                              {static_cast<std::int64_t>(a), static_cast<std::int64_t>(b), J});
          }
        }
      }
    }
  }

  void comm122_spawn(const Operator& X, const Operator& Y, const IndexCache& cache, SpawnContext& ctx)
  {
    if (X.GetParticleRank() < 2 && Y.GetParticleRank() < 2)
      return;
    const auto& X1 = X.OneBody;
    const auto& Y1 = Y.OneBody;
    const auto& X2 = X.TwoBody;
    const auto& Y2 = Y.TwoBody;
    ModelSpace& modelspace = *X.GetModelSpace();
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ikey = 0; ikey < cache.owned_two_body_keys.size(); ++ikey)
    {
      const int tid = omp_get_thread_num();
      const int ch_bra = static_cast<int>(cache.owned_two_body_keys[ikey][0]);
      const int ch_ket = static_cast<int>(cache.owned_two_body_keys[ikey][1]);
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(ch_bra);
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(ch_ket);
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
          for (std::size_t a : cache.all_orbits)
          {
            double value = 0.0;
            value += X1(i, a) * Y2.GetTBME_J(J, J, a, j, k, l);
            value += X1(j, a) * Y2.GetTBME_J(J, J, i, a, k, l);
            value += X2.GetTBME_J(J, J, i, j, a, l) * Y1(a, k);
            value += X2.GetTBME_J(J, J, i, j, k, a) * Y1(a, l);
            value -= Y1(i, a) * X2.GetTBME_J(J, J, a, j, k, l);
            value -= Y1(j, a) * X2.GetTBME_J(J, J, i, a, k, l);
            value -= Y2.GetTBME_J(J, J, i, j, a, l) * X1(a, k);
            value -= Y2.GetTBME_J(J, J, i, j, k, a) * X1(a, l);
            if (i == j)
              value /= PhysConst::SQRT2;
            if (k == l)
              value /= PhysConst::SQRT2;
            SpawnTwoBodyDelta(ctx, modelspace, tid, TERM_COMM122, 0,
                              ch_bra, ch_ket, ibra, iket, value, {static_cast<std::int64_t>(a)});
          }
        }
      }
    }
  }

  void SpawnIntermediateSet222(const Operator& L, const Operator& R, SpawnContext& ctx,
                               ModelSpace& modelspace, int tid, int piece_id,
                               int ch_bra, int ch_mid, int ch_ket, int ibra, int iket,
                               const arma::uvec& intermediate_indices, const arma::vec* weights,
                               double build_sign, double final_sign,
                               std::initializer_list<std::int64_t> source_prefix)
  {
    for (arma::uword iab = 0; iab < intermediate_indices.n_elem; ++iab)
    {
      const int ab = static_cast<int>(intermediate_indices(iab));
      const double weight = weights == nullptr ? 1.0 : (*weights)(iab);
      if (std::abs(weight) < ModelSpace::OCC_CUT)
        continue;
      const double value = final_sign * build_sign * weight *
                           MatrixElement(L, ch_bra, ch_mid, ibra, ab) *
                           MatrixElement(R, ch_mid, ch_ket, ab, iket);
      std::vector<std::int64_t> source(source_prefix.begin(), source_prefix.end());
      source.push_back(ab);
      SpawnTwoBodyDelta(ctx, modelspace, tid, TERM_COMM222_PPHH, piece_id,
                        ch_bra, ch_ket, ibra, iket, value,
                        {source[0], source[1], source[2], source[3], source[4]});
    }
  }

  void Emit222ForMatrixElement(const Operator& X, const Operator& Y, SpawnContext& ctx,
                               ModelSpace& modelspace, int tid,
                               int ch_bra, int ch_ket, int ibra, int iket,
                               int row, int col, int sym_piece)
  {
    const int ch_ab_XY = ch_bra;
    const int ch_ab_YX = ch_ket;
    TwoBodyChannel& tbc_ab_XY = modelspace.GetTwoBodyChannel(ch_ab_XY);
    TwoBodyChannel& tbc_ab_YX = modelspace.GetTwoBodyChannel(ch_ab_YX);
    SpawnIntermediateSet222(X, Y, ctx, modelspace, tid, 10 + sym_piece,
                            ch_bra, ch_ab_XY, ch_ket, row, col,
                            tbc_ab_XY.GetKetIndex_pp(), nullptr, +1.0, +1.0,
                            {row, col, 0, 0});
    SpawnIntermediateSet222(X, Y, ctx, modelspace, tid, 20 + sym_piece,
                            ch_bra, ch_ab_XY, ch_ket, row, col,
                            tbc_ab_XY.GetKetIndex_hh(), &tbc_ab_XY.Ket_occ_hh, +1.0, -1.0,
                            {row, col, 1, 0});
    SpawnIntermediateSet222(X, Y, ctx, modelspace, tid, 30 + sym_piece,
                            ch_bra, ch_ab_XY, ch_ket, row, col,
                            tbc_ab_XY.GetKetIndex_hh(), &tbc_ab_XY.Ket_unocc_hh, +1.0, +1.0,
                            {row, col, 2, 0});
    SpawnIntermediateSet222(X, Y, ctx, modelspace, tid, 40 + sym_piece,
                            ch_bra, ch_ab_XY, ch_ket, row, col,
                            tbc_ab_XY.GetKetIndex_ph(), &tbc_ab_XY.Ket_unocc_ph, +1.0, +1.0,
                            {row, col, 3, 0});

    if (!(ch_bra == ch_ket))
    {
      SpawnIntermediateSet222(Y, X, ctx, modelspace, tid, 50 + sym_piece,
                              ch_bra, ch_ab_YX, ch_ket, row, col,
                              tbc_ab_YX.GetKetIndex_pp(), nullptr, -1.0, +1.0,
                              {row, col, 0, 1});
      SpawnIntermediateSet222(Y, X, ctx, modelspace, tid, 60 + sym_piece,
                              ch_bra, ch_ab_YX, ch_ket, row, col,
                              tbc_ab_YX.GetKetIndex_hh(), &tbc_ab_YX.Ket_occ_hh, -1.0, -1.0,
                              {row, col, 1, 1});
      SpawnIntermediateSet222(Y, X, ctx, modelspace, tid, 70 + sym_piece,
                              ch_bra, ch_ab_YX, ch_ket, row, col,
                              tbc_ab_YX.GetKetIndex_hh(), &tbc_ab_YX.Ket_unocc_hh, -1.0, +1.0,
                              {row, col, 2, 1});
      SpawnIntermediateSet222(Y, X, ctx, modelspace, tid, 80 + sym_piece,
                              ch_bra, ch_ab_YX, ch_ket, row, col,
                              tbc_ab_YX.GetKetIndex_ph(), &tbc_ab_YX.Ket_unocc_ph, -1.0, +1.0,
                              {row, col, 3, 1});
    }
  }

  void comm222_pp_hh_spawn(const Operator& X, const Operator& Y, const IndexCache& cache, SpawnContext& ctx)
  {
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    ModelSpace& modelspace = *X.GetModelSpace();
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ikey = 0; ikey < cache.owned_two_body_keys.size(); ++ikey)
    {
      const int tid = omp_get_thread_num();
      const int ch_bra = static_cast<int>(cache.owned_two_body_keys[ikey][0]);
      const int ch_ket = static_cast<int>(cache.owned_two_body_keys[ikey][1]);
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(ch_bra);
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(ch_ket);
      const int nbras = static_cast<int>(tbc_bra.GetNumberKets());
      const int nkets = static_cast<int>(tbc_ket.GetNumberKets());
      for (int ibra = 0; ibra < nbras; ++ibra)
      {
        const int ketmin = (ch_bra == ch_ket) ? ibra : 0;
        for (int iket = ketmin; iket < nkets; ++iket)
        {
          Emit222ForMatrixElement(X, Y, ctx, modelspace, tid, ch_bra, ch_ket, ibra, iket, ibra, iket, 0);
          if (ch_bra == ch_ket)
            Emit222ForMatrixElement(X, Y, ctx, modelspace, tid, ch_bra, ch_ket, ibra, iket, iket, ibra, 1);
        }
      }
    }
  }

  void SpawnIntermediateSet221(const Operator& L, const Operator& R, SpawnContext& ctx,
                               int tid, int piece_id, int i, int j, int c, int J,
                               const arma::uvec& intermediate_indices, const arma::vec* weights,
                               double build_sign, double final_sign, double prefactor,
                               std::initializer_list<std::int64_t> source_prefix)
  {
    TwoBodyChannel& tbc = L.GetModelSpace()->GetTwoBodyChannel(
      L.GetModelSpace()->GetTwoBodyChannelIndex(J,
        (L.GetModelSpace()->GetOrbit(c).l + L.GetModelSpace()->GetOrbit(i).l) % 2,
        (L.GetModelSpace()->GetOrbit(c).tz2 + L.GetModelSpace()->GetOrbit(i).tz2) / 2));
    for (arma::uword iab = 0; iab < intermediate_indices.n_elem; ++iab)
    {
      const int ab = static_cast<int>(intermediate_indices(iab));
      const double weight = weights == nullptr ? 1.0 : (*weights)(iab);
      if (std::abs(weight) < ModelSpace::OCC_CUT)
        continue;
      Ket& ket_ab = tbc.GetKet(ab);
      const int a = ket_ab.p;
      const int b = ket_ab.q;
      const double value = prefactor * final_sign * build_sign * weight *
                           L.TwoBody.GetTBME_J(J, J, c, i, a, b) *
                           R.TwoBody.GetTBME_J(J, J, a, b, c, j);
      std::vector<std::int64_t> source(source_prefix.begin(), source_prefix.end());
      source.push_back(ab);
      SpawnOneBodyDelta(ctx, tid, TERM_COMM221, piece_id, i, j, value,
                        {source[0], source[1], source[2], source[3], source[4]});
    }
  }

  void comm221_spawn(const Operator& X, const Operator& Y, const IndexCache& cache, SpawnContext& ctx)
  {
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    ModelSpace& modelspace = *X.GetModelSpace();
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t ii = 0; ii < cache.all_orbits.size(); ++ii)
    {
      const int tid = omp_get_thread_num();
      const int i = static_cast<int>(cache.all_orbits[ii]);
      if (!OwnsOneBodyRow(i))
        continue;
      Orbit& oi = modelspace.GetOrbit(i);
      for (std::size_t j_orb : cache.one_body_targets[i])
      {
        const int j = static_cast<int>(j_orb);
        Orbit& oj = modelspace.GetOrbit(j);
        for (std::size_t c_orb : cache.all_orbits)
        {
          const int c = static_cast<int>(c_orb);
          Orbit& oc = modelspace.GetOrbit(c);
          const double nc = oc.occ;
          const double nbarc = 1.0 - nc;
          const int Jmin = std::max(std::abs(oc.j2 - oi.j2), std::abs(oc.j2 - oj.j2)) / 2;
          const int Jmax = (oc.j2 + std::min(oi.j2, oj.j2)) / 2;
          for (int J = Jmin; J <= Jmax; ++J)
          {
            const int parity = (oc.l + oi.l) % 2;
            const int Tz = (oc.tz2 + oi.tz2) / 2;
            const int ch = modelspace.GetTwoBodyChannelIndex(J, parity, Tz);
            if (!OwnsTwoBodyOutput(modelspace, ch))
              continue;
            TwoBodyChannel& tbc = modelspace.GetTwoBodyChannel(ch);
            const double pref_mpp = (2 * J + 1) * nc / (oi.j2 + 1.0);
            const double pref_mhh = (2 * J + 1) * nbarc / (oi.j2 + 1.0);
            if (std::abs(pref_mpp) > 0.0)
            {
              SpawnIntermediateSet221(X, Y, ctx, tid, 10, i, j, c, J,
                                      tbc.GetKetIndex_pp(), nullptr, +1.0, +1.0, pref_mpp, {c, J, 0, 0});
              SpawnIntermediateSet221(Y, X, ctx, tid, 20, i, j, c, J,
                                      tbc.GetKetIndex_pp(), nullptr, -1.0, +1.0, pref_mpp, {c, J, 0, 1});
              SpawnIntermediateSet221(X, Y, ctx, tid, 30, i, j, c, J,
                                      tbc.GetKetIndex_hh(), &tbc.Ket_unocc_hh, +1.0, +1.0, pref_mpp, {c, J, 1, 0});
              SpawnIntermediateSet221(Y, X, ctx, tid, 40, i, j, c, J,
                                      tbc.GetKetIndex_hh(), &tbc.Ket_unocc_hh, -1.0, +1.0, pref_mpp, {c, J, 1, 1});
              SpawnIntermediateSet221(X, Y, ctx, tid, 50, i, j, c, J,
                                      tbc.GetKetIndex_ph(), &tbc.Ket_unocc_ph, +1.0, +1.0, pref_mpp, {c, J, 2, 0});
              SpawnIntermediateSet221(Y, X, ctx, tid, 60, i, j, c, J,
                                      tbc.GetKetIndex_ph(), &tbc.Ket_unocc_ph, -1.0, +1.0, pref_mpp, {c, J, 2, 1});
            }
            if (std::abs(pref_mhh) > 0.0)
            {
              SpawnIntermediateSet221(X, Y, ctx, tid, 70, i, j, c, J,
                                      tbc.GetKetIndex_hh(), &tbc.Ket_occ_hh, +1.0, +1.0, pref_mhh, {c, J, 3, 0});
              SpawnIntermediateSet221(Y, X, ctx, tid, 80, i, j, c, J,
                                      tbc.GetKetIndex_hh(), &tbc.Ket_occ_hh, -1.0, +1.0, pref_mhh, {c, J, 3, 1});
            }
          }
        }
      }
    }
  }

  struct ScatterTarget
  {
    int ch;
    int ibra;
    int iket;
    double coefficient;
  };

  typedef std::vector<std::vector<std::vector<std::vector<ScatterTarget>>>> ReversePandyaPlan;

  void AddReversePandyaTerm(ReversePandyaPlan& reverse_plan,
                            int ch_cc, int row, int col,
                            int ch, int ibra, int iket, double coefficient)
  {
    if (ch_cc < 0 || ch_cc >= static_cast<int>(reverse_plan.size()))
      return;
    if (row < 0 || row >= static_cast<int>(reverse_plan[ch_cc].size()))
      return;
    if (col < 0 || col >= static_cast<int>(reverse_plan[ch_cc][row].size()))
      return;
    reverse_plan[ch_cc][row][col].push_back({ch, ibra, iket, coefficient});
  }

  ReversePandyaPlan BuildReversePandyaPlan(const Operator& H)
  {
    ModelSpace& modelspace = *H.GetModelSpace();
    const int nch_cc = static_cast<int>(modelspace.GetNumberTwoBodyChannels_CC());
    ReversePandyaPlan reverse_plan(nch_cc);
    for (int ch_cc = 0; ch_cc < nch_cc; ++ch_cc)
    {
      TwoBodyChannel_CC& tbc_cc = modelspace.GetTwoBodyChannel_CC(ch_cc);
      const int n = static_cast<int>(tbc_cc.GetNumberKets());
      reverse_plan[ch_cc].resize(n);
      for (int row = 0; row < n; ++row)
        reverse_plan[ch_cc][row].resize(2 * n);
    }

    const int nch = static_cast<int>(modelspace.GetNumberTwoBodyChannels());
    for (int ch = 0; ch < nch; ++ch)
    {
      TwoBodyChannel& tbc = modelspace.GetTwoBodyChannel(ch);
      const int J = tbc.J;
      const int nKets = static_cast<int>(tbc.GetNumberKets());
      for (int ibra = 0; ibra < nKets; ++ibra)
      {
        Ket& bra = tbc.GetKet(ibra);
        const int i = bra.p;
        const int j = bra.q;
        Orbit& oi = modelspace.GetOrbit(i);
        Orbit& oj = modelspace.GetOrbit(j);
        for (int iket = ibra; iket < nKets; ++iket)
        {
          Ket& ket = tbc.GetKet(iket);
          const int k = ket.p;
          const int l = ket.q;
          Orbit& ok = modelspace.GetOrbit(k);
          Orbit& ol = modelspace.GetOrbit(l);
          std::vector<std::array<double, 4>> commij;
          std::vector<std::array<double, 4>> commji;

          int parity_cc = (oi.l + ol.l) % 2;
          int Tz_cc = std::abs(oi.tz2 - ol.tz2) / 2;
          int Jpmin = std::max(std::abs(oi.j2 - ol.j2), std::abs(ok.j2 - oj.j2)) / 2;
          int Jpmax = std::min(oi.j2 + ol.j2, ok.j2 + oj.j2) / 2;
          for (int Jprime = Jpmin; Jprime <= Jpmax; ++Jprime)
          {
            const double sixj = modelspace.GetCachedSixJ(oi.j2, oj.j2, J, ok.j2, ol.j2, Jprime);
            if (std::abs(sixj) < 1e-8)
              continue;
            const int ch_cc = modelspace.GetTwoBodyChannelIndex(Jprime, parity_cc, Tz_cc);
            TwoBodyChannel_CC& tbc_cc = modelspace.GetTwoBodyChannel_CC(ch_cc);
            const int nkets_cc = static_cast<int>(tbc_cc.GetNumberKets());
            const int row = static_cast<int>(tbc_cc.GetLocalIndex(std::min(i, l), std::max(i, l))) + (i > l ? nkets_cc : 0);
            const int col = static_cast<int>(tbc_cc.GetLocalIndex(std::min(j, k), std::max(j, k))) + (k > j ? nkets_cc : 0);
            if (row >= 0 && row < nkets_cc && col >= 0 && col < 2 * nkets_cc)
              commij.push_back({static_cast<double>(ch_cc), static_cast<double>(row), static_cast<double>(col), -(2 * Jprime + 1) * sixj});
          }

          if (k == l)
          {
            commji = commij;
          }
          else if (i == j)
          {
            const int phase = modelspace.phase((oi.j2 + oj.j2 + ok.j2 + ol.j2) / 2);
            commji = commij;
            for (auto& term : commji)
              term[3] *= phase;
          }
          else
          {
            parity_cc = (oi.l + ok.l) % 2;
            Tz_cc = std::abs(oi.tz2 - ok.tz2) / 2;
            Jpmin = std::max(std::abs(oj.j2 - ol.j2), std::abs(ok.j2 - oi.j2)) / 2;
            Jpmax = std::min(oj.j2 + ol.j2, ok.j2 + oi.j2) / 2;
            for (int Jprime = Jpmin; Jprime <= Jpmax; ++Jprime)
            {
              const double sixj = modelspace.GetCachedSixJ(oj.j2, oi.j2, J, ok.j2, ol.j2, Jprime);
              if (std::abs(sixj) < 1e-8)
                continue;
              const int ch_cc = modelspace.GetTwoBodyChannelIndex(Jprime, parity_cc, Tz_cc);
              TwoBodyChannel_CC& tbc_cc = modelspace.GetTwoBodyChannel_CC(ch_cc);
              const int nkets_cc = static_cast<int>(tbc_cc.GetNumberKets());
              const int row = static_cast<int>(tbc_cc.GetLocalIndex(std::min(i, k), std::max(i, k))) + (i > k ? nkets_cc : 0);
              const int col = static_cast<int>(tbc_cc.GetLocalIndex(std::min(l, j), std::max(l, j))) + (l > j ? nkets_cc : 0);
              if (row >= 0 && row < nkets_cc && col >= 0 && col < 2 * nkets_cc)
                commji.push_back({static_cast<double>(ch_cc), static_cast<double>(row), static_cast<double>(col), -(2 * Jprime + 1) * sixj});
            }
          }

          const double norm = bra.delta_pq() == ket.delta_pq() ? 1 + bra.delta_pq() : PhysConst::SQRT2;
          const int phase_kl = modelspace.phase((ok.j2 + ol.j2) / 2 - J);
          for (const auto& term : commij)
            AddReversePandyaTerm(reverse_plan, static_cast<int>(term[0]), static_cast<int>(term[1]), static_cast<int>(term[2]),
                                 ch, ibra, iket, -term[3] / norm);
          for (const auto& term : commji)
            AddReversePandyaTerm(reverse_plan, static_cast<int>(term[0]), static_cast<int>(term[1]), static_cast<int>(term[2]),
                                 ch, ibra, iket, phase_kl * term[3] / norm);
        }
      }
    }
    return reverse_plan;
  }

  void ScatterZbarPrimitive(SpawnContext& ctx, ModelSpace& modelspace, int tid,
                            const ReversePandyaPlan& reverse_plan, int ch_cc, int row, int col,
                            double primitive, int piece_id,
                            std::initializer_list<std::int64_t> source)
  {
    if (primitive == 0.0)
      return;
    if (ch_cc < 0 || ch_cc >= static_cast<int>(reverse_plan.size()))
      return;
    if (row < 0 || row >= static_cast<int>(reverse_plan[ch_cc].size()))
      return;
    if (col < 0 || col >= static_cast<int>(reverse_plan[ch_cc][row].size()))
      return;
    for (const auto& target : reverse_plan[ch_cc][row][col])
      SpawnTwoBodyDelta(ctx, modelspace, tid, TERM_COMM222_PH, piece_id,
                        target.ch, target.ch, target.ibra, target.iket,
                        target.coefficient * primitive, source);
  }

  void comm222_ph_spawn(const Operator& X, const Operator& Y, const IndexCache& cache,
                        const ReversePandyaPlan& reverse_plan, SpawnContext& ctx)
  {
    if (X.GetParticleRank() < 2 || Y.GetParticleRank() < 2)
      return;
    ModelSpace& modelspace = *X.GetModelSpace();
    const std::size_t nch = modelspace.GetNumberTwoBodyChannels_CC();
    const int nranks = imsrg_mpi::Size();
    std::vector<std::size_t> local_channels;
    std::vector<std::size_t> owned_counts(nranks, 0);
    for (int ch_cc : modelspace.SortedTwoBodyChannels_CC)
    {
      const int owner = imsrg_mpi::Enabled() ? imsrg_mpi::CrossCoupledChannelOwner(modelspace, ch_cc) : 0;
      owned_counts[owner] += 1;
      if (owner == imsrg_mpi::Rank())
        local_channels.push_back(static_cast<std::size_t>(ch_cc));
    }

    const std::size_t batch_size = GetPandyaBatchSize();
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
      std::vector<imsrg_mpi::TwoBodyElementRequest> requests;
      for (std::size_t i = first; i < last; ++i)
      {
        const std::size_t ch = local_channels[i];
        batch_channels.push_back(ch);
        auto ch_requests = Commutator::GetPandyaElementRequestsForChannel(X, Y, X, ch);
        requests.insert(requests.end(), ch_requests.begin(), ch_requests.end());
      }
      imsrg_mpi::PrefetchTwoBodyMatrixElements(X_mut, requests);
      imsrg_mpi::PrefetchTwoBodyMatrixElements(Y_mut, requests);

      for (std::size_t ch_size : batch_channels)
      {
        const int ch = static_cast<int>(ch_size);
        TwoBodyChannel_CC& tbc_cc = modelspace.GetTwoBodyChannel_CC(ch);
        const int nKets_cc = static_cast<int>(tbc_cc.GetNumberKets());
        const std::size_t nph_kets = tbc_cc.GetKetIndex_hh().size() + tbc_cc.GetKetIndex_ph().size();
        if (nKets_cc == 0 || nph_kets == 0)
          continue;
        arma::mat Y_bar_ph;
        arma::mat Xt_bar_ph;
        Commutator::DoPandyaTransformation_SingleChannel_XandY(X, Y, Xt_bar_ph, Y_bar_ph, ch);
        if (Y_bar_ph.n_elem == 0 || Xt_bar_ph.n_elem == 0)
          continue;

        const int hy = Y.IsHermitian() ? 1 : -1;
        std::vector<int> ket_phase(nKets_cc, 1);
        for (int iket = 0; iket < nKets_cc; ++iket)
        {
          const Ket& ket = tbc_cc.GetKet(iket);
          if (modelspace.phase((ket.op->j2 + ket.oq->j2) / 2) < 0)
            ket_phase[iket] = -1;
        }
        arma::uvec phkets = arma::join_cols(tbc_cc.GetKetIndex_hh(), tbc_cc.GetKetIndex_ph());
        const int head_sym = (X.IsHermitian() != Y.IsHermitian()) ? +1 : -1;

#pragma omp parallel for schedule(dynamic, 1)
        for (int row = 0; row < nKets_cc; ++row)
        {
          const int tid = omp_get_thread_num();
          for (int col = 0; col < nKets_cc; ++col)
          {
            for (std::size_t iph = 0; iph < 2 * nph_kets; ++iph)
            {
              const double direct = Xt_bar_ph(row, iph) * Y_bar_ph(iph, col);
              ScatterZbarPrimitive(ctx, modelspace, tid, reverse_plan, ch, row, col, direct, 0,
                                   {ch, row, col, static_cast<std::int64_t>(iph), 0});
              ScatterZbarPrimitive(ctx, modelspace, tid, reverse_plan, ch, col, row, head_sym * direct, 1,
                                   {ch, row, col, static_cast<std::int64_t>(iph), 1});

              const std::size_t ph_index = iph < nph_kets ? iph : iph - nph_kets;
              const std::size_t source = iph < nph_kets ? iph + nph_kets : iph - nph_kets;
              const int phase_y = ket_phase[phkets(ph_index)] * ket_phase[col] * hy;
              const double flipped = Xt_bar_ph(row, iph) * Y_bar_ph(source, col) * phase_y;
              ScatterZbarPrimitive(ctx, modelspace, tid, reverse_plan, ch, row, nKets_cc + col, flipped, 2,
                                   {ch, row, col, static_cast<std::int64_t>(iph), 2});
              ScatterZbarPrimitive(ctx, modelspace, tid, reverse_plan, ch, col, nKets_cc + row,
                                   ket_phase[row] * ket_phase[col] * flipped, 3,
                                   {ch, row, col, static_cast<std::int64_t>(iph), 3});
            }
          }
        }
      }
      imsrg_mpi::ClearTwoBodyCache(X_mut);
      imsrg_mpi::ClearTwoBodyCache(Y_mut);
    }
    (void)cache;
  }
}

namespace StochasticEventIMSRG2
{
  double SpawnIMSRG2Delta(const Operator& Eta,
                          const Operator& H,
                          stochastic_imsrg::HamiltonianWalkerState& state,
                          double ds,
                          int istep)
  {
    const double t_start = omp_get_wtime();
    if (!Eta.IsAntiHermitian() || !H.IsHermitian() ||
        !Eta.IsNumberConserving() || !H.IsNumberConserving() ||
        Eta.GetParticleRank() > 2 || H.GetParticleRank() > 2 ||
        Eta.GetJRank() != 0 || H.GetJRank() != 0 ||
        Eta.GetTRank() != 0 || H.GetTRank() != 0 ||
        Eta.GetParity() != 0 || H.GetParity() != 0)
    {
      if (imsrg_mpi::Enabled())
        imsrg_mpi::Abort("stochastic source spawn currently supports scalar number-conserving IMSRG(2) Hamiltonian flow only.");
      throw std::runtime_error("stochastic source spawn currently supports scalar number-conserving IMSRG(2) Hamiltonian flow only.");
    }

    Operator& Eta_mut = const_cast<Operator&>(Eta);
    Operator& H_mut = const_cast<Operator&>(H);
    IndexCache cache(H);
    if (imsrg_mpi::OwnerOnlyStorageEnabled())
    {
      imsrg_mpi::PrefetchTwoBodyMatrices(Eta_mut, cache.scalar_two_body_keys);
      imsrg_mpi::PrefetchTwoBodyMatrices(H_mut, cache.scalar_two_body_keys);
    }

    OneBodyDeltaBuffers one_body_buffers = MakeOneBodyDeltaBuffers();
    TwoBodyDeltaBuffers two_body_buffers = MakeTwoBodyDeltaBuffers();
    SpawnContext ctx{state, ds, istep, one_body_buffers, two_body_buffers};

    double dEds = 0.0;
    if (Commutator::comm_term_on["comm110ss"])
      dEds += comm110_zero(Eta, H, cache);
    if (Commutator::comm_term_on["comm220ss"])
      dEds += comm220_zero(Eta, H, cache);
    imsrg_mpi::AllreduceInPlace(dEds);

    if (Commutator::comm_term_on["comm111ss"])
      comm111_spawn(Eta, H, cache, ctx);
    if (Commutator::comm_term_on["comm121ss"])
      comm121_spawn(Eta, H, cache, ctx);
    if (Commutator::comm_term_on["comm122ss"])
      comm122_spawn(Eta, H, cache, ctx);
    if (Commutator::comm_term_on["comm222_pp_hhss"])
      comm222_pp_hh_spawn(Eta, H, cache, ctx);
    if (Commutator::comm_term_on["comm221ss"])
      comm221_spawn(Eta, H, cache, ctx);
    if (Commutator::comm_term_on["comm222_phss"])
    {
      const ReversePandyaPlan reverse_plan = BuildReversePandyaPlan(H);
      comm222_ph_spawn(Eta, H, cache, reverse_plan, ctx);
    }

    if (imsrg_mpi::OwnerOnlyStorageEnabled())
    {
      imsrg_mpi::ClearTwoBodyCache(Eta_mut);
      imsrg_mpi::ClearTwoBodyCache(H_mut);
    }

    std::vector<imsrg_mpi::OneBodyDeltaContribution> received_one =
      imsrg_mpi::ExchangeOneBodyDeltaContributions(one_body_buffers);
    std::vector<imsrg_mpi::TwoBodyDeltaContribution> received_two =
      imsrg_mpi::ExchangeTwoBodyDeltaContributions(two_body_buffers);

    std::vector<stochastic_imsrg::OneBodyDeltaWalker> one_body_deltas;
    std::vector<stochastic_imsrg::TwoBodyDeltaWalker> two_body_deltas;
    one_body_deltas.reserve(received_one.size());
    two_body_deltas.reserve(received_two.size());
    for (const auto& delta : received_one)
      one_body_deltas.push_back({{delta.i, delta.j}, delta.delta_count});
    for (const auto& delta : received_two)
      two_body_deltas.push_back({{delta.ch_bra, delta.ch_ket, delta.ibra, delta.iket}, delta.delta_count});
    state.ApplyDeltaWalkers(one_body_deltas, two_body_deltas);

    IMSRGProfiler::timer["StochasticEventIMSRG2::SpawnIMSRG2Delta"] += omp_get_wtime() - t_start;
    return dEds;
  }
}
