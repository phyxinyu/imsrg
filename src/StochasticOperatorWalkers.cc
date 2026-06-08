///////////////////////////////////////////////////////////////////////////////////
// Deterministic-key stochastic walkers for IMSRG Hamiltonian matrix elements.
///////////////////////////////////////////////////////////////////////////////////

#include "StochasticOperatorWalkers.hh"
#include "IMSRGProfiler.hh"
#include "MpiSupport.hh"
#include "Operator.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <omp.h>

namespace
{
  void Fail(const std::string& message)
  {
    if (imsrg_mpi::Enabled())
      imsrg_mpi::Abort(message);
    throw std::runtime_error(message);
  }

  void ValidateHamiltonianLike(const Operator& op)
  {
    if (!op.IsNumberConserving() || op.GetJRank() != 0 || op.GetTRank() != 0 ||
        op.GetParity() != 0 || op.GetParticleRank() > 2)
    {
      Fail("stochastic Hamiltonian walkers currently support scalar, number-conserving IMSRG(2) Hamiltonians only.");
    }
    if (!op.IsHermitian())
      Fail("stochastic Hamiltonian walkers currently require a Hermitian Hamiltonian.");
  }

  int OneBodyOwner(int i)
  {
    const int nranks = std::max(1, imsrg_mpi::Size());
    return i % nranks;
  }

  bool OwnsOneBodyTarget(int i)
  {
    return !imsrg_mpi::Enabled() || OneBodyOwner(i) == imsrg_mpi::Rank();
  }

  bool OwnsTwoBodyTarget(const Operator& op, std::size_t ch_bra)
  {
    return !imsrg_mpi::OwnerOnlyStorageEnabled() ||
           imsrg_mpi::TwoBodyChannelOwner(*op.GetModelSpace(), ch_bra) == imsrg_mpi::Rank();
  }

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

  std::int64_t StochasticRoundSigned(double value, double quantum, std::uint64_t key)
  {
    if (value == 0.0)
      return 0;
    if (!(quantum > 0.0))
      Fail("stochastic Hamiltonian walker quantum must be positive.");

    const double scaled = std::abs(value) / quantum;
    const double base_as_double = std::floor(scaled);
    if (base_as_double > static_cast<double>(std::numeric_limits<std::int64_t>::max() - 1))
      Fail("stochastic Hamiltonian walker count overflow.");

    std::int64_t magnitude = static_cast<std::int64_t>(base_as_double);
    const double frac = scaled - base_as_double;
    if (UnitRandom(key) < frac)
      ++magnitude;

    return value < 0.0 ? -magnitude : magnitude;
  }

  std::uint64_t OneBodyKey(std::uint64_t seed, int step, int i, int j)
  {
    std::uint64_t key = seed;
    HashCombine(key, static_cast<std::uint64_t>(step));
    HashCombine(key, 1);
    HashCombine(key, static_cast<std::uint64_t>(i));
    HashCombine(key, static_cast<std::uint64_t>(j));
    return key;
  }

  std::uint64_t OneBodyDeltaKey(std::uint64_t seed, int step, int i, int j)
  {
    std::uint64_t key = seed;
    HashCombine(key, static_cast<std::uint64_t>(step));
    HashCombine(key, 101);
    HashCombine(key, static_cast<std::uint64_t>(i));
    HashCombine(key, static_cast<std::uint64_t>(j));
    return key;
  }

  std::uint64_t TwoBodyKey(std::uint64_t seed, int step, int ch_bra, int ch_ket, int ibra, int iket)
  {
    std::uint64_t key = seed;
    HashCombine(key, static_cast<std::uint64_t>(step));
    HashCombine(key, 2);
    HashCombine(key, static_cast<std::uint64_t>(ch_bra));
    HashCombine(key, static_cast<std::uint64_t>(ch_ket));
    HashCombine(key, static_cast<std::uint64_t>(ibra));
    HashCombine(key, static_cast<std::uint64_t>(iket));
    return key;
  }

  std::uint64_t TwoBodyDeltaKey(std::uint64_t seed, int step, int ch_bra, int ch_ket, int ibra, int iket)
  {
    std::uint64_t key = seed;
    HashCombine(key, static_cast<std::uint64_t>(step));
    HashCombine(key, 102);
    HashCombine(key, static_cast<std::uint64_t>(ch_bra));
    HashCombine(key, static_cast<std::uint64_t>(ch_ket));
    HashCombine(key, static_cast<std::uint64_t>(ibra));
    HashCombine(key, static_cast<std::uint64_t>(iket));
    return key;
  }

  bool IsCanonicalOneBody(int i, int j)
  {
    return i <= j;
  }

  bool IsCanonicalTwoBody(std::size_t ch_bra, std::size_t ch_ket, std::size_t ibra, std::size_t iket)
  {
    return ch_bra != ch_ket || ibra <= iket;
  }

  bool OneBodyTargetLess(const stochastic_imsrg::OneBodyMatrixWalkerTarget& lhs,
                         const stochastic_imsrg::OneBodyMatrixWalkerTarget& rhs)
  {
    if (lhs.i != rhs.i)
      return lhs.i < rhs.i;
    return lhs.j < rhs.j;
  }

  bool OneBodyTargetEqual(const stochastic_imsrg::OneBodyMatrixWalkerTarget& lhs,
                          const stochastic_imsrg::OneBodyMatrixWalkerTarget& rhs)
  {
    return lhs.i == rhs.i && lhs.j == rhs.j;
  }

  bool TwoBodyTargetLess(const stochastic_imsrg::TwoBodyMatrixWalkerTarget& lhs,
                         const stochastic_imsrg::TwoBodyMatrixWalkerTarget& rhs)
  {
    if (lhs.ch_bra != rhs.ch_bra)
      return lhs.ch_bra < rhs.ch_bra;
    if (lhs.ch_ket != rhs.ch_ket)
      return lhs.ch_ket < rhs.ch_ket;
    if (lhs.ibra != rhs.ibra)
      return lhs.ibra < rhs.ibra;
    return lhs.iket < rhs.iket;
  }

  bool TwoBodyTargetEqual(const stochastic_imsrg::TwoBodyMatrixWalkerTarget& lhs,
                          const stochastic_imsrg::TwoBodyMatrixWalkerTarget& rhs)
  {
    return lhs.ch_bra == rhs.ch_bra && lhs.ch_ket == rhs.ch_ket &&
           lhs.ibra == rhs.ibra && lhs.iket == rhs.iket;
  }

  void ApplyOneBodyDeltas(std::vector<stochastic_imsrg::OneBodyMatrixWalker>& population,
                          const std::vector<stochastic_imsrg::OneBodyDeltaWalker>& deltas)
  {
    if (deltas.empty())
      return;

    population.reserve(population.size() + deltas.size());
    for (const auto& delta : deltas)
      population.push_back({delta.target, delta.delta_count});

    std::sort(population.begin(), population.end(),
              [](const auto& lhs, const auto& rhs) {
                return OneBodyTargetLess(lhs.target, rhs.target);
              });

    std::vector<stochastic_imsrg::OneBodyMatrixWalker> merged;
    merged.reserve(population.size());
    for (const auto& walker : population)
    {
      if (!merged.empty() && OneBodyTargetEqual(merged.back().target, walker.target))
      {
        merged.back().count += walker.count;
        if (merged.back().count == 0)
          merged.pop_back();
      }
      else if (walker.count != 0)
      {
        merged.push_back(walker);
      }
    }
    population.swap(merged);
  }

  void ApplyTwoBodyDeltas(std::vector<stochastic_imsrg::TwoBodyMatrixWalker>& population,
                          const std::vector<stochastic_imsrg::TwoBodyDeltaWalker>& deltas)
  {
    if (deltas.empty())
      return;

    population.reserve(population.size() + deltas.size());
    for (const auto& delta : deltas)
      population.push_back({delta.target, delta.delta_count});

    std::sort(population.begin(), population.end(),
              [](const auto& lhs, const auto& rhs) {
                return TwoBodyTargetLess(lhs.target, rhs.target);
              });

    std::vector<stochastic_imsrg::TwoBodyMatrixWalker> merged;
    merged.reserve(population.size());
    for (const auto& walker : population)
    {
      if (!merged.empty() && TwoBodyTargetEqual(merged.back().target, walker.target))
      {
        merged.back().count += walker.count;
        if (merged.back().count == 0)
          merged.pop_back();
      }
      else if (walker.count != 0)
      {
        merged.push_back(walker);
      }
    }
    population.swap(merged);
  }

  double LocalIndependentOneTwoBodyL1(const Operator& op)
  {
    ValidateHamiltonianLike(op);

    double l1 = 0.0;
    const int norbits = static_cast<int>(op.GetModelSpace()->GetNumberOrbits());
    for (int i = 0; i < norbits; ++i)
    {
      if (!OwnsOneBodyTarget(i))
        continue;
      for (int j = i; j < norbits; ++j)
        l1 += std::abs(op.OneBody(i, j));
    }

    if (op.TwoBody.IsAllocated())
    {
      for (const auto& itmat : op.TwoBody.MatEl)
      {
        const std::size_t ch_bra = itmat.first[0];
        const std::size_t ch_ket = itmat.first[1];
        if (!OwnsTwoBodyTarget(op, ch_bra))
          continue;
        const arma::mat& matrix = itmat.second;
        for (std::size_t ibra = 0; ibra < matrix.n_rows; ++ibra)
        {
          for (std::size_t iket = 0; iket < matrix.n_cols; ++iket)
          {
            if (!IsCanonicalTwoBody(ch_bra, ch_ket, ibra, iket))
              continue;
            l1 += std::abs(matrix(ibra, iket));
          }
        }
      }
    }

    imsrg_mpi::AllreduceInPlace(l1);
    return l1;
  }
}

namespace stochastic_imsrg
{
  void HamiltonianWalkerState::InitializeFromOperator(const Operator& op,
                                                      std::uint64_t initial_walkers,
                                                      std::uint64_t rng_seed)
  {
    if (initial_walkers == 0)
      Fail("stochastic_imsrg_initial_walkers must be positive.");

    requested_walkers = initial_walkers;
    seed = rng_seed;
    refinement_count = 0;
    last_refine_step = 0;

    const double l1 = LocalIndependentOneTwoBodyL1(op);
    if (!(l1 > 0.0))
      Fail("cannot initialize stochastic Hamiltonian walkers: OneBody/TwoBody L1 norm is zero.");

    quantum = l1 / static_cast<double>(initial_walkers);
    initial_quantum = quantum;
    ProjectFromOperator(op, 0);
  }

  void HamiltonianWalkerState::ProjectFromOperator(const Operator& op, int step)
  {
    ValidateHamiltonianLike(op);
    if (!(quantum > 0.0))
      Fail("stochastic Hamiltonian walkers have not been initialized.");

    const double t_start = omp_get_wtime();
    one_body.clear();
    two_body.clear();

    const int norbits = static_cast<int>(op.GetModelSpace()->GetNumberOrbits());
    one_body.reserve(static_cast<std::size_t>(norbits * (norbits + 1) / 2));
    for (int i = 0; i < norbits; ++i)
    {
      if (!OwnsOneBodyTarget(i))
        continue;
      for (int j = i; j < norbits; ++j)
      {
        if (!IsCanonicalOneBody(i, j))
          continue;
        const double value = op.OneBody(i, j);
        const std::int64_t count = StochasticRoundSigned(value, quantum, OneBodyKey(seed, step, i, j));
        if (count != 0)
          one_body.push_back({{i, j}, count});
      }
    }

    if (op.TwoBody.IsAllocated())
    {
      for (const auto& itmat : op.TwoBody.MatEl)
      {
        const std::size_t ch_bra = itmat.first[0];
        const std::size_t ch_ket = itmat.first[1];
        if (!OwnsTwoBodyTarget(op, ch_bra))
          continue;
        const arma::mat& matrix = itmat.second;
        for (std::size_t ibra = 0; ibra < matrix.n_rows; ++ibra)
        {
          for (std::size_t iket = 0; iket < matrix.n_cols; ++iket)
          {
            if (!IsCanonicalTwoBody(ch_bra, ch_ket, ibra, iket))
              continue;
            const double value = matrix(ibra, iket);
            const std::int64_t count = StochasticRoundSigned(
                value, quantum,
                TwoBodyKey(seed, step, static_cast<int>(ch_bra), static_cast<int>(ch_ket),
                           static_cast<int>(ibra), static_cast<int>(iket)));
            if (count != 0)
            {
              two_body.push_back({{static_cast<int>(ch_bra), static_cast<int>(ch_ket),
                                   static_cast<int>(ibra), static_cast<int>(iket)},
                                  count});
            }
          }
        }
      }
    }

    IMSRGProfiler::counter["StochasticIMSRG_OneBodyWalkerTargets"] += static_cast<int>(one_body.size());
    IMSRGProfiler::counter["StochasticIMSRG_TwoBodyWalkerTargets"] += static_cast<int>(two_body.size());
    IMSRGProfiler::timer["StochasticIMSRG_ProjectHamiltonian"] += omp_get_wtime() - t_start;
  }

  void HamiltonianWalkerState::ReconstructInto(Operator& op, double zero_body) const
  {
    if (!(quantum > 0.0))
      Fail("stochastic Hamiltonian walkers have not been initialized.");

    const double t_start = omp_get_wtime();
    op.ZeroBody = zero_body;
    op.EraseOneBody();
    op.EraseTwoBody();

    for (const auto& walker : one_body)
    {
      const int i = walker.target.i;
      const int j = walker.target.j;
      const double value = quantum * static_cast<double>(walker.count);
      op.OneBody(i, j) = value;
      if (i != j)
        op.OneBody(j, i) = value;
    }
    imsrg_mpi::AllreduceInPlace(op.OneBody);

    for (const auto& walker : two_body)
    {
      const int ch_bra = walker.target.ch_bra;
      const int ch_ket = walker.target.ch_ket;
      const int ibra = walker.target.ibra;
      const int iket = walker.target.iket;
      const double value = quantum * static_cast<double>(walker.count);
      arma::mat& matrix = op.TwoBody.GetMatrix(ch_bra, ch_ket);
      matrix(ibra, iket) = value;
      if (ch_bra == ch_ket && ibra != iket)
        matrix(iket, ibra) = value;
    }

    imsrg_mpi::RestrictOperatorToOwnedChannels(op);
    IMSRGProfiler::timer["StochasticIMSRG_ReconstructHamiltonian"] += omp_get_wtime() - t_start;
  }

  void HamiltonianWalkerState::ApplyDerivativeSpawn(const Operator& dH, double ds, int step)
  {
    ValidateHamiltonianLike(dH);
    if (!(quantum > 0.0))
      Fail("stochastic Hamiltonian walkers have not been initialized.");

    const double t_start = omp_get_wtime();
    std::vector<OneBodyDeltaWalker> one_body_deltas;
    std::vector<TwoBodyDeltaWalker> two_body_deltas;

    const int norbits = static_cast<int>(dH.GetModelSpace()->GetNumberOrbits());
    one_body_deltas.reserve(static_cast<std::size_t>(norbits * (norbits + 1) / 2));
    for (int i = 0; i < norbits; ++i)
    {
      if (!OwnsOneBodyTarget(i))
        continue;
      for (int j = i; j < norbits; ++j)
      {
        const double increment = ds * dH.OneBody(i, j);
        const std::int64_t delta_count = StochasticRoundSigned(
            increment, quantum, OneBodyDeltaKey(seed, step, i, j));
        if (delta_count != 0)
          one_body_deltas.push_back({{i, j}, delta_count});
      }
    }

    if (dH.TwoBody.IsAllocated())
    {
      for (const auto& itmat : dH.TwoBody.MatEl)
      {
        const std::size_t ch_bra = itmat.first[0];
        const std::size_t ch_ket = itmat.first[1];
        if (!OwnsTwoBodyTarget(dH, ch_bra))
          continue;
        const arma::mat& matrix = itmat.second;
        for (std::size_t ibra = 0; ibra < matrix.n_rows; ++ibra)
        {
          for (std::size_t iket = 0; iket < matrix.n_cols; ++iket)
          {
            if (!IsCanonicalTwoBody(ch_bra, ch_ket, ibra, iket))
              continue;
            const double increment = ds * matrix(ibra, iket);
            const std::int64_t delta_count = StochasticRoundSigned(
                increment, quantum,
                TwoBodyDeltaKey(seed, step, static_cast<int>(ch_bra), static_cast<int>(ch_ket),
                                static_cast<int>(ibra), static_cast<int>(iket)));
            if (delta_count != 0)
            {
              two_body_deltas.push_back({{static_cast<int>(ch_bra), static_cast<int>(ch_ket),
                                          static_cast<int>(ibra), static_cast<int>(iket)},
                                         delta_count});
            }
          }
        }
      }
    }

    ApplyOneBodyDeltas(one_body, one_body_deltas);
    ApplyTwoBodyDeltas(two_body, two_body_deltas);

    IMSRGProfiler::counter["StochasticIMSRG_OneBodyDeltaTargets"] += static_cast<int>(one_body_deltas.size());
    IMSRGProfiler::counter["StochasticIMSRG_TwoBodyDeltaTargets"] += static_cast<int>(two_body_deltas.size());
    IMSRGProfiler::counter["StochasticIMSRG_OneBodyWalkerTargetsAfterDelta"] += static_cast<int>(one_body.size());
    IMSRGProfiler::counter["StochasticIMSRG_TwoBodyWalkerTargetsAfterDelta"] += static_cast<int>(two_body.size());
    IMSRGProfiler::timer["StochasticIMSRG_ApplyDerivativeSpawn"] += omp_get_wtime() - t_start;
  }

  void HamiltonianWalkerState::ApplyDeltaWalkers(const std::vector<OneBodyDeltaWalker>& one_body_deltas,
                                                 const std::vector<TwoBodyDeltaWalker>& two_body_deltas)
  {
    if (!(quantum > 0.0))
      Fail("stochastic Hamiltonian walkers have not been initialized.");

    const double t_start = omp_get_wtime();
    ApplyOneBodyDeltas(one_body, one_body_deltas);
    ApplyTwoBodyDeltas(two_body, two_body_deltas);

    IMSRGProfiler::counter["StochasticIMSRG_OneBodyDeltaTargets"] += static_cast<int>(one_body_deltas.size());
    IMSRGProfiler::counter["StochasticIMSRG_TwoBodyDeltaTargets"] += static_cast<int>(two_body_deltas.size());
    IMSRGProfiler::counter["StochasticIMSRG_OneBodyWalkerTargetsAfterDelta"] += static_cast<int>(one_body.size());
    IMSRGProfiler::counter["StochasticIMSRG_TwoBodyWalkerTargetsAfterDelta"] += static_cast<int>(two_body.size());
    IMSRGProfiler::timer["StochasticIMSRG_ApplyDeltaWalkers"] += omp_get_wtime() - t_start;
  }

  double HamiltonianWalkerState::CurrentIndependentOneTwoBodyL1(const Operator& op) const
  {
    return LocalIndependentOneTwoBodyL1(op);
  }

  void HamiltonianWalkerState::RefineQuantumFromOperator(const Operator& op,
                                                         double new_quantum,
                                                         int step)
  {
    if (!(new_quantum > 0.0))
      Fail("new stochastic Hamiltonian walker quantum must be positive.");
    if (quantum > 0.0 && !(new_quantum < quantum))
      return;

    const double t_start = omp_get_wtime();
    quantum = new_quantum;
    ProjectFromOperator(op, step);
    IMSRGProfiler::timer["StochasticIMSRG_QuantumRefinementProject"] += omp_get_wtime() - t_start;
  }

  std::uint64_t HamiltonianWalkerState::TotalAbsWalkerCount() const
  {
    std::uint64_t total = 0;
    for (const auto& walker : one_body)
      total += static_cast<std::uint64_t>(std::llabs(walker.count));
    for (const auto& walker : two_body)
      total += static_cast<std::uint64_t>(std::llabs(walker.count));

    double total_as_double = static_cast<double>(total);
    imsrg_mpi::AllreduceInPlace(total_as_double);
    return static_cast<std::uint64_t>(std::llround(total_as_double));
  }
}
