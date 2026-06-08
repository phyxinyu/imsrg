///////////////////////////////////////////////////////////////////////////////////
// Deterministic-key stochastic walkers for IMSRG Hamiltonian matrix elements.
///////////////////////////////////////////////////////////////////////////////////

#ifndef StochasticOperatorWalkers_hh
#define StochasticOperatorWalkers_hh 1

#include <cstdint>
#include <vector>

class Operator;

namespace stochastic_imsrg
{
  struct OneBodyMatrixWalkerTarget
  {
    int i;
    int j;
  };

  struct TwoBodyMatrixWalkerTarget
  {
    int ch_bra;
    int ch_ket;
    int ibra;
    int iket;
  };

  struct OneBodyMatrixWalker
  {
    OneBodyMatrixWalkerTarget target;
    std::int64_t count;
  };

  struct TwoBodyMatrixWalker
  {
    TwoBodyMatrixWalkerTarget target;
    std::int64_t count;
  };

  struct OneBodyDeltaWalker
  {
    OneBodyMatrixWalkerTarget target;
    std::int64_t delta_count;
  };

  struct TwoBodyDeltaWalker
  {
    TwoBodyMatrixWalkerTarget target;
    std::int64_t delta_count;
  };

  struct HamiltonianWalkerState
  {
    double quantum = 0.0;
    double initial_quantum = 0.0;
    std::uint64_t seed = 0;
    std::uint64_t requested_walkers = 0;
    int refinement_count = 0;
    int last_refine_step = 0;
    std::vector<OneBodyMatrixWalker> one_body;
    std::vector<TwoBodyMatrixWalker> two_body;

    void InitializeFromOperator(const Operator& op,
                                std::uint64_t initial_walkers,
                                std::uint64_t rng_seed);
    void ProjectFromOperator(const Operator& op, int step);
    void ReconstructInto(Operator& op, double zero_body) const;
    void ApplyDerivativeSpawn(const Operator& dH, double ds, int step);
    void ApplyDeltaWalkers(const std::vector<OneBodyDeltaWalker>& one_body_deltas,
                           const std::vector<TwoBodyDeltaWalker>& two_body_deltas);
    double CurrentIndependentOneTwoBodyL1(const Operator& op) const;
    void RefineQuantumFromOperator(const Operator& op, double new_quantum, int step);

    std::uint64_t TotalAbsWalkerCount() const;
  };
}

#endif
