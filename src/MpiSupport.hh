///////////////////////////////////////////////////////////////////////////////////
// Optional MPI helpers for imsrg++.
///////////////////////////////////////////////////////////////////////////////////

#ifndef MpiSupport_hh
#define MpiSupport_hh 1

#include <armadillo>
#include <array>
#include <cstddef>
#include <cstdint>
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

  struct OneBodyContribution
  {
    int i;
    int j;
    double value;
  };

  struct TwoBodyContribution
  {
    int ch_bra;
    int ch_ket;
    int ibra;
    int iket;
    double value;
  };

  struct OneBodyDeltaContribution
  {
    int i;
    int j;
    std::int64_t delta_count;
  };

  struct TwoBodyDeltaContribution
  {
    int ch_bra;
    int ch_ket;
    int ibra;
    int iket;
    std::int64_t delta_count;
  };

  void ExchangeAndApplyOneBodyContributions(
    Operator& op,
    std::vector<std::vector<std::vector<OneBodyContribution>>>& thread_send_buffers);
  void ExchangeAndApplyTwoBodyContributions(
    Operator& op,
    std::vector<std::vector<std::vector<TwoBodyContribution>>>& thread_send_buffers);

  std::vector<OneBodyDeltaContribution> ExchangeOneBodyDeltaContributions(
    std::vector<std::vector<std::vector<OneBodyDeltaContribution>>>& thread_send_buffers);
  std::vector<TwoBodyDeltaContribution> ExchangeTwoBodyDeltaContributions(
    std::vector<std::vector<std::vector<TwoBodyDeltaContribution>>>& thread_send_buffers);

  void AllreduceInPlace(double& value);
  void AllreduceInPlace(arma::mat& matrix);
  void BroadcastMatrixFromRank(arma::mat& matrix, int root);
  std::vector<double> AlltoallvDoubles(const std::vector<std::vector<double>>& send_buffers);
  void AllreduceTwoBodyInPlace(TwoBodyME& two_body);
  void AllreduceOperatorInPlace(Operator& op);

  void RestrictOperatorToOwnedChannels(Operator& op);
  void PrefetchTwoBodyMatrices(Operator& op);
  void PrefetchTwoBodyMatrices(Operator& op, const std::vector<std::array<std::size_t, 2>>& requested_keys);

  struct TwoBodyElementRequest
  {
    std::size_t ch_bra;
    std::size_t ch_ket;
    std::size_t bra_ind;
    std::size_t ket_ind;
  };

  void PrefetchTwoBodyMatrixElements(Operator& op, const std::vector<TwoBodyElementRequest>& requested_elements);
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
