///////////////////////////////////////////////////////////////////////////////////
// Optional MPI helpers for imsrg++.
///////////////////////////////////////////////////////////////////////////////////

#ifndef MpiSupport_hh
#define MpiSupport_hh 1

#include <armadillo>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

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

  void SetOwnerOnlyStorage(bool enabled);
  bool OwnerOnlyStorageEnabled();

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
  bool OwnsTwoBodyMatrix(ModelSpace& modelspace, const std::array<std::size_t, 2>& key);

  void AllreduceInPlace(double& value);
  void AllreduceInPlace(arma::mat& matrix);
  void BroadcastMatrixFromRank(arma::mat& matrix, int root);
  std::vector<double> AlltoallvDoubles(const std::vector<std::vector<double>>& send_buffers);
  void AllreduceTwoBodyInPlace(TwoBodyME& two_body);
  void AllreduceOperatorInPlace(Operator& op);

  void RestrictOperatorToOwnedChannels(Operator& op);
  void PrefetchTwoBodyMatrices(Operator& op);
  void PrefetchTwoBodyMatrices(Operator& op, const std::vector<std::array<std::size_t, 2>>& requested_keys);
  void ClearTwoBodyCache(Operator& op);
  void GatherOperatorToRoot(Operator& op);

  double TwoBodyNorm(const TwoBodyME& two_body);
  double OneBodyNorm(const Operator& op);
  double TwoBodyNorm(const Operator& op);
  double ThreeBodyNorm(const Operator& op);
  double Norm(const Operator& op);
  double MP2Energy(const Operator& op);
}

#endif
