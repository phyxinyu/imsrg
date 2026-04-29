///////////////////////////////////////////////////////////////////////////////////
// Optional MPI helpers for imsrg++.
///////////////////////////////////////////////////////////////////////////////////

#include "MpiSupport.hh"

#include "ModelSpace.hh"
#include "Operator.hh"
#include "TwoBodyME.hh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef IMSRG_USE_MPI
#include <mpi.h>
#endif

namespace
{
  bool mpi_enabled = false;
  bool mpi_initialized_by_us = false;
  int mpi_rank = 0;
  int mpi_size = 1;

  ModelSpace* ownership_modelspace = nullptr;
  int ownership_rank_count = 1;
  std::vector<int> two_body_owner;
  std::vector<int> cross_coupled_owner;

  void RefreshRankSize()
  {
#ifdef IMSRG_USE_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized)
    {
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
      return;
    }
#endif
    mpi_rank = 0;
    mpi_size = 1;
  }

  template <typename WeightFn>
  std::vector<int> BuildOwners(std::size_t nchannels, int nranks, WeightFn weight_fn)
  {
    std::vector<int> owner(nchannels, 0);
    if (nranks <= 1 || nchannels == 0)
      return owner;

    std::vector<std::pair<double, std::size_t>> weighted_channels;
    weighted_channels.reserve(nchannels);
    for (std::size_t ch = 0; ch < nchannels; ++ch)
      weighted_channels.push_back({weight_fn(ch), ch});

    std::sort(weighted_channels.begin(), weighted_channels.end(),
              [](const auto& lhs, const auto& rhs) {
                if (lhs.first == rhs.first)
                  return lhs.second < rhs.second;
                return lhs.first > rhs.first;
              });

    std::vector<double> rank_load(nranks, 0.0);
    for (const auto& entry : weighted_channels)
    {
      auto rank_it = std::min_element(rank_load.begin(), rank_load.end());
      int rank = static_cast<int>(std::distance(rank_load.begin(), rank_it));
      owner[entry.second] = rank;
      rank_load[rank] += entry.first;
    }
    return owner;
  }
}

namespace imsrg_mpi
{
  bool IsCompiledWithMPI()
  {
#ifdef IMSRG_USE_MPI
    return true;
#else
    return false;
#endif
  }

  void Initialize(int& argc, char**& argv)
  {
#ifdef IMSRG_USE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
      MPI_Init(&argc, &argv);
      mpi_initialized_by_us = true;
    }
    RefreshRankSize();
#else
    (void)argc;
    (void)argv;
    RefreshRankSize();
#endif
  }

  void Finalize()
  {
#ifdef IMSRG_USE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (mpi_initialized_by_us && !finalized)
      MPI_Finalize();
#endif
  }

  void SetEnabled(bool enabled)
  {
#ifdef IMSRG_USE_MPI
    RefreshRankSize();
    mpi_enabled = enabled;
    if (mpi_enabled && !Initialized())
      throw std::runtime_error("MPI IMSRG(2) requested before MPI_Init.");
#else
    RefreshRankSize();
    if (enabled)
      std::cout << "mpi_imsrg2=true was requested, but this build was not configured with IMSRG_USE_MPI=ON. Running serial code." << std::endl;
    mpi_enabled = false;
#endif
  }

  bool Enabled()
  {
    return mpi_enabled && IsCompiledWithMPI() && mpi_size > 1;
  }

  bool Initialized()
  {
#ifdef IMSRG_USE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    return initialized != 0;
#else
    return false;
#endif
  }

  int Rank()
  {
    return mpi_rank;
  }

  int Size()
  {
    return mpi_size;
  }

  bool IsRoot()
  {
    return Rank() == 0;
  }

  void Barrier()
  {
#ifdef IMSRG_USE_MPI
    if (Enabled())
      MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  void Abort(const std::string& message, int error_code)
  {
    if (IsRoot())
      std::cerr << message << std::endl;
#ifdef IMSRG_USE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (Initialized() && !finalized)
      MPI_Abort(MPI_COMM_WORLD, error_code);
#endif
    std::exit(error_code);
  }

  void EnsureChannelOwnership(ModelSpace& modelspace)
  {
    int nranks = Size();
    if (ownership_modelspace == &modelspace && ownership_rank_count == nranks &&
        two_body_owner.size() == modelspace.GetNumberTwoBodyChannels() &&
        cross_coupled_owner.size() == modelspace.GetNumberTwoBodyChannels_CC())
      return;

    ownership_modelspace = &modelspace;
    ownership_rank_count = nranks;

    two_body_owner = BuildOwners(modelspace.GetNumberTwoBodyChannels(), nranks,
      [&modelspace](std::size_t ch) {
        double nkets = static_cast<double>(modelspace.GetTwoBodyChannel(ch).GetNumberKets());
        return std::max(1.0, nkets * nkets);
      });

    cross_coupled_owner = BuildOwners(modelspace.GetNumberTwoBodyChannels_CC(), nranks,
      [&modelspace](std::size_t ch) {
        double nkets = static_cast<double>(modelspace.GetTwoBodyChannel_CC(ch).GetNumberKets());
        return std::max(1.0, 2.0 * nkets * nkets);
      });
  }

  int TwoBodyChannelOwner(ModelSpace& modelspace, std::size_t ch)
  {
    EnsureChannelOwnership(modelspace);
    return two_body_owner.at(ch);
  }

  int CrossCoupledChannelOwner(ModelSpace& modelspace, std::size_t ch)
  {
    EnsureChannelOwnership(modelspace);
    return cross_coupled_owner.at(ch);
  }

  bool OwnsTwoBodyChannel(ModelSpace& modelspace, std::size_t ch)
  {
    return !Enabled() || TwoBodyChannelOwner(modelspace, ch) == Rank();
  }

  bool OwnsCrossCoupledChannel(ModelSpace& modelspace, std::size_t ch)
  {
    return !Enabled() || CrossCoupledChannelOwner(modelspace, ch) == Rank();
  }

  void AllreduceInPlace(double& value)
  {
#ifdef IMSRG_USE_MPI
    if (Enabled())
      MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)value;
#endif
  }

  void AllreduceInPlace(arma::mat& matrix)
  {
#ifdef IMSRG_USE_MPI
    if (Enabled() && matrix.n_elem > 0)
    {
      arma::uword offset = 0;
      const arma::uword max_count = static_cast<arma::uword>(std::numeric_limits<int>::max());
      while (offset < matrix.n_elem)
      {
        arma::uword count = std::min(max_count, matrix.n_elem - offset);
        MPI_Allreduce(MPI_IN_PLACE, matrix.memptr() + offset, static_cast<int>(count), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        offset += count;
      }
    }
#else
    (void)matrix;
#endif
  }

  void AllreduceTwoBodyInPlace(TwoBodyME& two_body)
  {
    if (!Enabled() || !two_body.IsAllocated())
      return;
    for (auto& itmat : two_body.MatEl)
      AllreduceInPlace(itmat.second);
  }

  void AllreduceOperatorInPlace(Operator& op)
  {
    if (!Enabled())
      return;
    AllreduceInPlace(op.ZeroBody);
    AllreduceInPlace(op.OneBody);
    AllreduceTwoBodyInPlace(op.TwoBody);
  }
}
