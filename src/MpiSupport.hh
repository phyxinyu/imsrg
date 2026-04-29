///////////////////////////////////////////////////////////////////////////////////
// Optional MPI helpers for imsrg++.
///////////////////////////////////////////////////////////////////////////////////

#ifndef MpiSupport_hh
#define MpiSupport_hh 1

#include <armadillo>
#include <cstddef>
#include <string>

class ModelSpace;
class Operator;
class TwoBodyME;

namespace imsrg_mpi
{
  void Initialize(int& argc, char**& argv);
  void Finalize();

  bool IsCompiledWithMPI();
  void SetEnabled(bool enabled);
  bool Enabled();
  bool Initialized();

  int Rank();
  int Size();
  bool IsRoot();

  void Barrier();
  void Abort(const std::string& message, int error_code = 1);

  void EnsureChannelOwnership(ModelSpace& modelspace);
  bool OwnsTwoBodyChannel(ModelSpace& modelspace, std::size_t ch);
  bool OwnsCrossCoupledChannel(ModelSpace& modelspace, std::size_t ch);
  int TwoBodyChannelOwner(ModelSpace& modelspace, std::size_t ch);
  int CrossCoupledChannelOwner(ModelSpace& modelspace, std::size_t ch);

  void AllreduceInPlace(double& value);
  void AllreduceInPlace(arma::mat& matrix);
  void AllreduceTwoBodyInPlace(TwoBodyME& two_body);
  void AllreduceOperatorInPlace(Operator& op);
}

#endif
