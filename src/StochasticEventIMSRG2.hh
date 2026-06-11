///////////////////////////////////////////////////////////////////////////////////
// Source-level stochastic spawn kernels for scalar IMSRG(2) Hamiltonian flow.
///////////////////////////////////////////////////////////////////////////////////

#ifndef StochasticEventIMSRG2_hh
#define StochasticEventIMSRG2_hh 1

#include "Operator.hh"
#include "StochasticOperatorWalkers.hh"

#include <cstdint>

namespace StochasticEventIMSRG2
{
  struct ChannelSamplingOptions
  {
    bool enabled = false;
    bool diagnostics = false;
    bool coalesce_samples = true;
    std::uint64_t samples = 1000000;
    int min_samples = 4;
    int exact_threshold = 64;
    double uniform_mix = 0.05;
  };

  double SpawnIMSRG2Delta(const Operator& Eta,
                          const Operator& H,
                          stochastic_imsrg::HamiltonianWalkerState& state,
                          double ds,
                          int istep,
                          const ChannelSamplingOptions& channel_sampling = ChannelSamplingOptions{});
}

#endif
