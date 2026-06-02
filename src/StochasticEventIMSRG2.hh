///////////////////////////////////////////////////////////////////////////////////
// Source-level stochastic spawn kernels for scalar IMSRG(2) Hamiltonian flow.
///////////////////////////////////////////////////////////////////////////////////

#ifndef StochasticEventIMSRG2_hh
#define StochasticEventIMSRG2_hh 1

#include "Operator.hh"
#include "StochasticOperatorWalkers.hh"

namespace StochasticEventIMSRG2
{
  double SpawnIMSRG2Delta(const Operator& Eta,
                          const Operator& H,
                          stochastic_imsrg::HamiltonianWalkerState& state,
                          double ds,
                          int istep);
}

#endif
