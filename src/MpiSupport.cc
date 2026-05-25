///////////////////////////////////////////////////////////////////////////////////
// Optional MPI helpers for imsrg++.
///////////////////////////////////////////////////////////////////////////////////

#include "MpiSupport.hh"

#include "AngMom.hh"
#include "IMSRGProfiler.hh"
#include "ModelSpace.hh"
#include "Operator.hh"
#include "TwoBodyME.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#ifdef IMSRG_USE_MPI
#include <mpi.h>
#endif

#include <omp.h>

namespace
{
  bool mpi_enabled = false;
  bool owner_only_storage = false;
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

  std::vector<std::array<std::size_t, 2>> MatrixKeys(const TwoBodyME& two_body)
  {
    std::vector<std::array<std::size_t, 2>> keys;
    if (two_body.modelspace == nullptr)
      return keys;

    std::size_t nchannels = two_body.modelspace->GetNumberTwoBodyChannels();
    keys.reserve(nchannels);
    for (std::size_t ch_bra = 0; ch_bra < nchannels; ++ch_bra)
    {
      TwoBodyChannel& tbc_bra = two_body.modelspace->GetTwoBodyChannel(ch_bra);
      for (std::size_t ch_ket = ch_bra; ch_ket < nchannels; ++ch_ket)
      {
        TwoBodyChannel& tbc_ket = two_body.modelspace->GetTwoBodyChannel(ch_ket);
        if (!AngMom::Triangle(tbc_bra.J, tbc_ket.J, two_body.rank_J))
          continue;
        if (std::abs(tbc_bra.Tz - tbc_ket.Tz) != two_body.rank_T)
          continue;
        if ((tbc_bra.parity + tbc_ket.parity + two_body.parity) % 2 > 0)
          continue;
        keys.push_back({ch_bra, ch_ket});
      }
    }
    return keys;
  }

  std::array<std::size_t, 2> CanonicalKey(std::array<std::size_t, 2> key)
  {
    if (key[0] > key[1])
      std::swap(key[0], key[1]);
    return key;
  }

  template <typename ChunkFn>
  void ForEachMatrixChunk(arma::mat& matrix, ChunkFn&& chunk_fn)
  {
    arma::uword offset = 0;
    const arma::uword max_count = static_cast<arma::uword>(std::numeric_limits<int>::max());
    while (offset < matrix.n_elem)
    {
      arma::uword count = std::min(max_count, matrix.n_elem - offset);
      chunk_fn(matrix.memptr() + offset, static_cast<int>(count));
      offset += count;
    }
  }

  bool ContributionKeyLess(const imsrg_mpi::OneBodyContribution& lhs,
                           const imsrg_mpi::OneBodyContribution& rhs)
  {
    return std::tie(lhs.i, lhs.j) < std::tie(rhs.i, rhs.j);
  }

  bool ContributionKeyEqual(const imsrg_mpi::OneBodyContribution& lhs,
                            const imsrg_mpi::OneBodyContribution& rhs)
  {
    return lhs.i == rhs.i && lhs.j == rhs.j;
  }

  bool ContributionKeyLess(const imsrg_mpi::TwoBodyContribution& lhs,
                           const imsrg_mpi::TwoBodyContribution& rhs)
  {
    return std::tie(lhs.ch_bra, lhs.ch_ket, lhs.ibra, lhs.iket) <
           std::tie(rhs.ch_bra, rhs.ch_ket, rhs.ibra, rhs.iket);
  }

  bool ContributionKeyEqual(const imsrg_mpi::TwoBodyContribution& lhs,
                            const imsrg_mpi::TwoBodyContribution& rhs)
  {
    return lhs.ch_bra == rhs.ch_bra && lhs.ch_ket == rhs.ch_ket &&
           lhs.ibra == rhs.ibra && lhs.iket == rhs.iket;
  }

  template <typename Contribution>
  void SortAndMergeContributions(std::vector<Contribution>& contributions)
  {
    std::sort(contributions.begin(), contributions.end(),
              [](const Contribution& lhs, const Contribution& rhs) {
                return ContributionKeyLess(lhs, rhs);
              });
    std::vector<Contribution> merged;
    merged.reserve(contributions.size());
    for (std::size_t i = 0; i < contributions.size();)
    {
      Contribution accum = contributions[i];
      double sum = 0.0;
      double correction = 0.0;
      std::size_t j = i;
      while (j < contributions.size() && ContributionKeyEqual(contributions[i], contributions[j]))
      {
        const double y = contributions[j].value - correction;
        const double t = sum + y;
        correction = (t - sum) - y;
        sum = t;
        ++j;
      }
      accum.value = sum;
      if (std::abs(accum.value) > 0.0)
        merged.push_back(accum);
      i = j;
    }
    contributions.swap(merged);
  }

  template <typename Contribution>
  std::vector<std::vector<Contribution>> MergeThreadContributionBuffers(
    std::vector<std::vector<std::vector<Contribution>>>& thread_send_buffers,
    const std::string& timer_name)
  {
    const double t_start = omp_get_wtime();
    const int nranks = imsrg_mpi::Size();
    std::vector<std::vector<Contribution>> send_buffers(nranks);
    for (int rank = 0; rank < nranks; ++rank)
    {
      std::size_t total = 0;
      for (auto& thread_buffers : thread_send_buffers)
        if (rank < static_cast<int>(thread_buffers.size()))
          total += thread_buffers[rank].size();
      send_buffers[rank].reserve(total);
      for (auto& thread_buffers : thread_send_buffers)
      {
        if (rank >= static_cast<int>(thread_buffers.size()))
          continue;
        auto& src = thread_buffers[rank];
        send_buffers[rank].insert(send_buffers[rank].end(),
                                  std::make_move_iterator(src.begin()),
                                  std::make_move_iterator(src.end()));
        src.clear();
      }
      SortAndMergeContributions(send_buffers[rank]);
    }
    IMSRGProfiler::timer[timer_name] += omp_get_wtime() - t_start;
    return send_buffers;
  }

#ifdef IMSRG_USE_MPI
  MPI_Datatype GetOneBodyContributionType()
  {
    static MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    if (mpi_type == MPI_DATATYPE_NULL)
    {
      imsrg_mpi::OneBodyContribution dummy{};
      const int nblocks = 3;
      int blocklengths[nblocks] = {1, 1, 1};
      MPI_Datatype types[nblocks] = {MPI_INT, MPI_INT, MPI_DOUBLE};
      MPI_Aint offsets[nblocks];
      MPI_Aint base = 0;
      MPI_Get_address(&dummy, &base);
      MPI_Get_address(&dummy.i, &offsets[0]);
      MPI_Get_address(&dummy.j, &offsets[1]);
      MPI_Get_address(&dummy.value, &offsets[2]);
      for (int i = 0; i < nblocks; ++i)
        offsets[i] -= base;
      MPI_Type_create_struct(nblocks, blocklengths, offsets, types, &mpi_type);
      MPI_Type_commit(&mpi_type);
    }
    return mpi_type;
  }

  MPI_Datatype GetTwoBodyContributionType()
  {
    static MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    if (mpi_type == MPI_DATATYPE_NULL)
    {
      imsrg_mpi::TwoBodyContribution dummy{};
      const int nblocks = 5;
      int blocklengths[nblocks] = {1, 1, 1, 1, 1};
      MPI_Datatype types[nblocks] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
      MPI_Aint offsets[nblocks];
      MPI_Aint base = 0;
      MPI_Get_address(&dummy, &base);
      MPI_Get_address(&dummy.ch_bra, &offsets[0]);
      MPI_Get_address(&dummy.ch_ket, &offsets[1]);
      MPI_Get_address(&dummy.ibra, &offsets[2]);
      MPI_Get_address(&dummy.iket, &offsets[3]);
      MPI_Get_address(&dummy.value, &offsets[4]);
      for (int i = 0; i < nblocks; ++i)
        offsets[i] -= base;
      MPI_Type_create_struct(nblocks, blocklengths, offsets, types, &mpi_type);
      MPI_Type_commit(&mpi_type);
    }
    return mpi_type;
  }
#endif

  template <typename Contribution>
  std::vector<Contribution> AlltoallvContributions(
    const std::vector<std::vector<Contribution>>& send_buffers,
    const std::string& timer_name
#ifdef IMSRG_USE_MPI
    , MPI_Datatype mpi_type
#endif
    )
  {
    const double t_start = omp_get_wtime();
    std::vector<Contribution> received;
#ifdef IMSRG_USE_MPI
    if (imsrg_mpi::Enabled())
    {
      const int nranks = imsrg_mpi::Size();
      std::vector<int> send_counts(nranks, 0);
      std::vector<int> recv_counts(nranks, 0);
      for (int rank = 0; rank < nranks; ++rank)
      {
        if (rank < static_cast<int>(send_buffers.size()))
          send_counts[rank] = static_cast<int>(send_buffers[rank].size());
      }

      MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

      std::vector<int> send_displs(nranks, 0);
      std::vector<int> recv_displs(nranks, 0);
      for (int rank = 1; rank < nranks; ++rank)
      {
        send_displs[rank] = send_displs[rank - 1] + send_counts[rank - 1];
        recv_displs[rank] = recv_displs[rank - 1] + recv_counts[rank - 1];
      }

      const int total_send = send_displs.empty() ? 0 : send_displs.back() + send_counts.back();
      const int total_recv = recv_displs.empty() ? 0 : recv_displs.back() + recv_counts.back();
      std::vector<Contribution> send_flat(static_cast<std::size_t>(total_send));
      for (int rank = 0; rank < nranks; ++rank)
      {
        if (rank < static_cast<int>(send_buffers.size()) && !send_buffers[rank].empty())
          std::copy(send_buffers[rank].begin(), send_buffers[rank].end(),
                    send_flat.begin() + send_displs[rank]);
      }
      received.resize(static_cast<std::size_t>(total_recv));
      MPI_Alltoallv(send_flat.data(), send_counts.data(), send_displs.data(), mpi_type,
                    received.data(), recv_counts.data(), recv_displs.data(), mpi_type,
                    MPI_COMM_WORLD);

      IMSRGProfiler::counter[timer_name + "_SendTotal"] += total_send;
      IMSRGProfiler::counter[timer_name + "_RecvTotal"] += total_recv;
      IMSRGProfiler::counter[timer_name + "_SendMaxPeer"] += send_counts.empty() ? 0 : *std::max_element(send_counts.begin(), send_counts.end());
      IMSRGProfiler::counter[timer_name + "_RecvMaxPeer"] += recv_counts.empty() ? 0 : *std::max_element(recv_counts.begin(), recv_counts.end());
    }
    else
#endif
    {
      if (!send_buffers.empty())
        received = send_buffers.front();
    }
    IMSRGProfiler::timer[timer_name] += omp_get_wtime() - t_start;
    return received;
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
      int ierr = MPI_Init(&argc, &argv);
      if (ierr != MPI_SUCCESS)
        throw std::runtime_error("MPI_Init failed with error code " + std::to_string(ierr));
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
    return mpi_enabled && IsCompiledWithMPI();
  }

  void SetOwnerOnlyStorage(bool enabled)
  {
    owner_only_storage = enabled && Enabled();
  }

  bool OwnerOnlyStorageEnabled()
  {
    return owner_only_storage && Enabled();
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

  bool OwnsTwoBodyMatrix(ModelSpace& modelspace, const std::array<std::size_t, 2>& key)
  {
    return !OwnerOnlyStorageEnabled() || TwoBodyChannelOwner(modelspace, key[0]) == Rank();
  }

  void ExchangeAndApplyOneBodyContributions(
    Operator& op,
    std::vector<std::vector<std::vector<OneBodyContribution>>>& thread_send_buffers)
  {
    auto send_buffers = MergeThreadContributionBuffers(thread_send_buffers, "MPI_MergeOneBodyContributions_Send");
    std::vector<OneBodyContribution> received;
    if (!Enabled() || Size() <= 1)
    {
      received = send_buffers.empty() ? std::vector<OneBodyContribution>() : std::move(send_buffers.front());
    }
    else
    {
      received = AlltoallvContributions(
        send_buffers, "MPI_AlltoallvOneBodyContributions"
#ifdef IMSRG_USE_MPI
        , GetOneBodyContributionType()
#endif
        );
    }

    const double t_merge = omp_get_wtime();
    SortAndMergeContributions(received);
    for (const auto& c : received)
    {
      if (Enabled() && Size() > 1 && c.i % Size() != Rank())
        Abort("One-body contribution delivered to non-owner rank.");
      op.OneBody(c.i, c.j) += c.value;
    }
    IMSRGProfiler::timer["MPI_MergeOneBodyContributions_Recv"] += omp_get_wtime() - t_merge;
  }

  void ExchangeAndApplyTwoBodyContributions(
    Operator& op,
    std::vector<std::vector<std::vector<TwoBodyContribution>>>& thread_send_buffers)
  {
    auto send_buffers = MergeThreadContributionBuffers(thread_send_buffers, "MPI_MergeTwoBodyContributions_Send");
    std::vector<TwoBodyContribution> received;
    if (!Enabled() || Size() <= 1)
    {
      received = send_buffers.empty() ? std::vector<TwoBodyContribution>() : std::move(send_buffers.front());
    }
    else
    {
      received = AlltoallvContributions(
        send_buffers, "MPI_AlltoallvTwoBodyContributions"
#ifdef IMSRG_USE_MPI
        , GetTwoBodyContributionType()
#endif
        );
    }

    const double t_merge = omp_get_wtime();
    SortAndMergeContributions(received);
    for (const auto& c : received)
    {
      std::array<std::size_t, 2> key{static_cast<std::size_t>(c.ch_bra), static_cast<std::size_t>(c.ch_ket)};
      if (OwnerOnlyStorageEnabled() && !OwnsTwoBodyMatrix(*op.GetModelSpace(), key))
        Abort("Two-body contribution delivered to non-owner rank.");
      op.TwoBody.AddToTBME(c.ch_bra, c.ch_ket, c.ibra, c.iket, c.value);
    }
    IMSRGProfiler::timer["MPI_MergeTwoBodyContributions_Recv"] += omp_get_wtime() - t_merge;
  }

  void AllreduceInPlace(double& value)
  {
    double t_start = omp_get_wtime();
#ifdef IMSRG_USE_MPI
    if (Enabled())
      MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)value;
#endif
    double elapsed = omp_get_wtime() - t_start;
    IMSRGProfiler::timer["MPI_Allreduce"] += elapsed;
    IMSRGProfiler::timer["MPI_AllreduceScalar"] += elapsed;
  }

  void AllreduceInPlace(arma::mat& matrix)
  {
    double t_start = omp_get_wtime();
#ifdef IMSRG_USE_MPI
    if (Enabled() && matrix.n_elem > 0)
    {
      ForEachMatrixChunk(matrix, [](double* data, int count) {
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      });
    }
#else
    (void)matrix;
#endif
    double elapsed = omp_get_wtime() - t_start;
    IMSRGProfiler::timer["MPI_Allreduce"] += elapsed;
    IMSRGProfiler::timer["MPI_AllreduceMatrix"] += elapsed;
  }

  void BroadcastMatrixFromRank(arma::mat& matrix, int root)
  {
    double t_start = omp_get_wtime();
#ifdef IMSRG_USE_MPI
    if (Enabled() && matrix.n_elem > 0)
    {
      ForEachMatrixChunk(matrix, [root](double* data, int count) {
        MPI_Bcast(data, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      });
    }
#else
    (void)matrix;
    (void)root;
#endif
    IMSRGProfiler::timer["MPI_BcastTwoBodyMatrices"] += omp_get_wtime() - t_start;
  }

  void TransferMatrixFromOwnerToRequesters(arma::mat& matrix, int owner, const std::vector<int>& requesters)
  {
    double t_start = omp_get_wtime();
#ifdef IMSRG_USE_MPI
    if (Enabled() && matrix.n_elem > 0 && !requesters.empty())
    {
      const int tag = 39217;
      const int rank = Rank();
      const bool rank_requests_matrix = std::binary_search(requesters.begin(), requesters.end(), rank);
      if (rank == owner || rank_requests_matrix)
      {
        ForEachMatrixChunk(matrix, [&](double* data, int count) {
          if (rank == owner)
          {
            std::vector<MPI_Request> requests;
            requests.reserve(requesters.size());
            for (int requester : requesters)
            {
              if (requester == owner)
                continue;
              MPI_Request request;
              MPI_Isend(data, count, MPI_DOUBLE, requester, tag, MPI_COMM_WORLD, &request);
              requests.push_back(request);
            }
            if (!requests.empty())
              MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
          }
          else
          {
            MPI_Recv(data, count, MPI_DOUBLE, owner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
      }
    }
#else
    (void)matrix;
    (void)owner;
    (void)requesters;
#endif
    double elapsed = omp_get_wtime() - t_start;
    IMSRGProfiler::timer["MPI_BcastTwoBodyMatrices"] += elapsed;
    IMSRGProfiler::timer["MPI_RequesterTransferTwoBodyMatrices"] += elapsed;
  }

  std::vector<double> AlltoallvDoubles(const std::vector<std::vector<double>>& send_buffers)
  {
    double t_start = omp_get_wtime();
    std::vector<double> received;
#ifdef IMSRG_USE_MPI
    if (!Enabled())
    {
      received = send_buffers.empty() ? received : send_buffers.front();
    }
    else
    {
      int nranks = Size();
      std::vector<int> send_counts(nranks, 0);
      std::vector<int> recv_counts(nranks, 0);
      for (int rank = 0; rank < nranks; ++rank)
      {
        if (rank < static_cast<int>(send_buffers.size()))
          send_counts[rank] = static_cast<int>(send_buffers[rank].size());
      }

      MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

      std::vector<int> send_displs(nranks, 0);
      std::vector<int> recv_displs(nranks, 0);
      for (int rank = 1; rank < nranks; ++rank)
      {
        send_displs[rank] = send_displs[rank - 1] + send_counts[rank - 1];
        recv_displs[rank] = recv_displs[rank - 1] + recv_counts[rank - 1];
      }

      std::vector<double> send_flat(send_displs.back() + send_counts.back());
      for (int rank = 0; rank < nranks; ++rank)
      {
        if (rank < static_cast<int>(send_buffers.size()) && !send_buffers[rank].empty())
          std::copy(send_buffers[rank].begin(), send_buffers[rank].end(), send_flat.begin() + send_displs[rank]);
      }

      received.resize(recv_displs.back() + recv_counts.back());
      MPI_Alltoallv(send_flat.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                    received.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                    MPI_COMM_WORLD);
    }
#else
    received = send_buffers.empty() ? std::vector<double>() : send_buffers.front();
#endif
    IMSRGProfiler::timer["MPI_AlltoallvDoubles"] += omp_get_wtime() - t_start;
    return received;
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

  void RestrictOperatorToOwnedChannels(Operator& op)
  {
    if (!OwnerOnlyStorageEnabled() || !op.TwoBody.IsAllocated())
      return;

    EnsureChannelOwnership(*op.GetModelSpace());
    for (auto it = op.TwoBody.MatEl.begin(); it != op.TwoBody.MatEl.end();)
    {
      if (OwnsTwoBodyMatrix(*op.GetModelSpace(), it->first))
        ++it;
      else
        it = op.TwoBody.MatEl.erase(it);
    }
  }

  void PrefetchTwoBodyMatrices(Operator& op)
  {
    PrefetchTwoBodyMatrices(op, MatrixKeys(op.TwoBody));
  }

  void PrefetchTwoBodyMatrices(Operator& op, const std::vector<std::array<std::size_t, 2>>& requested_keys)
  {
    double t_start = omp_get_wtime();
    if (!OwnerOnlyStorageEnabled() || !op.TwoBody.IsAllocated())
    {
      IMSRGProfiler::timer["MPI_PrefetchTwoBodyMatrices"] += omp_get_wtime() - t_start;
      return;
    }

    ModelSpace& modelspace = *op.GetModelSpace();
    EnsureChannelOwnership(modelspace);

    std::vector<std::array<std::size_t, 2>> local_keys;
    local_keys.reserve(requested_keys.size());
    for (const auto& key : requested_keys)
      local_keys.push_back(CanonicalKey(key));
    std::sort(local_keys.begin(), local_keys.end());
    local_keys.erase(std::unique(local_keys.begin(), local_keys.end()), local_keys.end());

    std::vector<std::array<std::size_t, 2>> legal_keys = MatrixKeys(op.TwoBody);
    std::vector<int> needed(legal_keys.size(), 0);
    for (std::size_t i = 0; i < legal_keys.size(); ++i)
    {
      if (std::binary_search(local_keys.begin(), local_keys.end(), legal_keys[i]))
        needed[i] = 1;
    }

    int nranks = Size();
    std::vector<int> needed_by_rank(needed.size() * static_cast<std::size_t>(nranks), 0);
#ifdef IMSRG_USE_MPI
    if (Enabled() && !needed.empty())
    {
      double t_allgather = omp_get_wtime();
      MPI_Allgather(needed.data(), static_cast<int>(needed.size()), MPI_INT,
                    needed_by_rank.data(), static_cast<int>(needed.size()), MPI_INT,
                    MPI_COMM_WORLD);
      IMSRGProfiler::timer["MPI_PrefetchNeededAllgather"] += omp_get_wtime() - t_allgather;
    }
    else
#endif
    {
      std::copy(needed.begin(), needed.end(), needed_by_rank.begin());
    }

    for (std::size_t ikey = 0; ikey < legal_keys.size(); ++ikey)
    {
      std::vector<int> requesters;
      requesters.reserve(nranks);
      for (int rank = 0; rank < nranks; ++rank)
      {
        if (needed_by_rank[static_cast<std::size_t>(rank) * legal_keys.size() + ikey])
          requesters.push_back(rank);
      }
      if (requesters.empty())
        continue;

      const auto& key = legal_keys[ikey];
      std::size_t ch_bra = key[0];
      std::size_t ch_ket = key[1];
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(ch_bra);
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(ch_ket);
      int owner = TwoBodyChannelOwner(modelspace, ch_bra);
      bool rank_needs_matrix = needed[ikey] != 0;
      bool rank_sends_matrix = Rank() == owner;

      if (!rank_needs_matrix && !rank_sends_matrix)
        continue;

      auto it = op.TwoBody.MatEl.find(key);
      if (it == op.TwoBody.MatEl.end())
      {
        it = op.TwoBody.MatEl.emplace(
          key, arma::mat(tbc_bra.GetNumberKets(), tbc_ket.GetNumberKets(), arma::fill::zeros)).first;
      }
      TransferMatrixFromOwnerToRequesters(it->second, owner, requesters);
    }
    IMSRGProfiler::timer["MPI_PrefetchTwoBodyMatrices"] += omp_get_wtime() - t_start;
  }

  void PrefetchTwoBodyMatrixElements(Operator& op, const std::vector<TwoBodyElementRequest>& requested_elements)
  {
    double t_start = omp_get_wtime();
    if (!OwnerOnlyStorageEnabled() || !op.TwoBody.IsAllocated())
    {
      IMSRGProfiler::timer["MPI_PrefetchTwoBodyElements"] += omp_get_wtime() - t_start;
      return;
    }

    ModelSpace& modelspace = *op.GetModelSpace();
    EnsureChannelOwnership(modelspace);

    std::vector<TwoBodyElementRequest> local_requests;
    local_requests.reserve(requested_elements.size());
    for (auto request : requested_elements)
    {
      if (request.ch_bra > request.ch_ket)
      {
        std::swap(request.ch_bra, request.ch_ket);
        std::swap(request.bra_ind, request.ket_ind);
      }
      local_requests.push_back(request);
    }
    auto request_less = [](const TwoBodyElementRequest& lhs, const TwoBodyElementRequest& rhs) {
      return std::tie(lhs.ch_bra, lhs.ch_ket, lhs.bra_ind, lhs.ket_ind) <
             std::tie(rhs.ch_bra, rhs.ch_ket, rhs.bra_ind, rhs.ket_ind);
    };
    auto request_equal = [](const TwoBodyElementRequest& lhs, const TwoBodyElementRequest& rhs) {
      return lhs.ch_bra == rhs.ch_bra && lhs.ch_ket == rhs.ch_ket &&
             lhs.bra_ind == rhs.bra_ind && lhs.ket_ind == rhs.ket_ind;
    };
    std::sort(local_requests.begin(), local_requests.end(), request_less);
    local_requests.erase(std::unique(local_requests.begin(), local_requests.end(), request_equal), local_requests.end());

    int nranks = Size();
    std::vector<std::vector<double>> request_buffers(nranks);
    for (const auto& request : local_requests)
    {
      int owner = TwoBodyChannelOwner(modelspace, request.ch_bra);
      if (owner == Rank())
        continue;
      auto& buffer = request_buffers[owner];
      buffer.push_back(static_cast<double>(Rank()));
      buffer.push_back(static_cast<double>(request.ch_bra));
      buffer.push_back(static_cast<double>(request.ch_ket));
      buffer.push_back(static_cast<double>(request.bra_ind));
      buffer.push_back(static_cast<double>(request.ket_ind));
    }

    std::vector<double> incoming_requests = AlltoallvDoubles(request_buffers);
    std::vector<std::vector<double>> value_buffers(nranks);
    for (std::size_t ipacket = 0; ipacket + 4 < incoming_requests.size(); ipacket += 5)
    {
      int requester = static_cast<int>(incoming_requests[ipacket]);
      std::size_t ch_bra = static_cast<std::size_t>(incoming_requests[ipacket + 1]);
      std::size_t ch_ket = static_cast<std::size_t>(incoming_requests[ipacket + 2]);
      std::size_t bra_ind = static_cast<std::size_t>(incoming_requests[ipacket + 3]);
      std::size_t ket_ind = static_cast<std::size_t>(incoming_requests[ipacket + 4]);

      double value = 0;
      if (!op.TwoBody.TryGetStoredElement(ch_bra, ch_ket, bra_ind, ket_ind, value))
      {
        Abort("MPI element prefetch owner is missing a requested TwoBody element (" +
              std::to_string(ch_bra) + "," + std::to_string(ch_ket) + ")[" +
              std::to_string(bra_ind) + "," + std::to_string(ket_ind) + "].");
      }
      auto& buffer = value_buffers[requester];
      buffer.push_back(static_cast<double>(ch_bra));
      buffer.push_back(static_cast<double>(ch_ket));
      buffer.push_back(static_cast<double>(bra_ind));
      buffer.push_back(static_cast<double>(ket_ind));
      buffer.push_back(value);
    }

    std::vector<double> incoming_values = AlltoallvDoubles(value_buffers);
    for (std::size_t ipacket = 0; ipacket + 4 < incoming_values.size(); ipacket += 5)
    {
      std::size_t ch_bra = static_cast<std::size_t>(incoming_values[ipacket]);
      std::size_t ch_ket = static_cast<std::size_t>(incoming_values[ipacket + 1]);
      std::size_t bra_ind = static_cast<std::size_t>(incoming_values[ipacket + 2]);
      std::size_t ket_ind = static_cast<std::size_t>(incoming_values[ipacket + 3]);
      double value = incoming_values[ipacket + 4];
      op.TwoBody.SetSparseElement(ch_bra, ch_ket, bra_ind, ket_ind, value);
    }

    IMSRGProfiler::timer["MPI_PrefetchTwoBodyElements"] += omp_get_wtime() - t_start;
  }

  void ClearTwoBodyCache(Operator& op)
  {
    double t_start = omp_get_wtime();
    // Remove temporary prefetched non-owner matrices; owned resident storage remains.
    RestrictOperatorToOwnedChannels(op);
    op.TwoBody.ClearSparseElements();
    IMSRGProfiler::timer["MPI_ClearTwoBodyCache"] += omp_get_wtime() - t_start;
  }

  void GatherOperatorToRoot(Operator& op)
  {
    double t_start = omp_get_wtime();
    if (!OwnerOnlyStorageEnabled() || !op.TwoBody.IsAllocated())
    {
      IMSRGProfiler::timer["MPI_GatherOperatorToRoot"] += omp_get_wtime() - t_start;
      return;
    }
    PrefetchTwoBodyMatrices(op);
    if (!IsRoot())
      ClearTwoBodyCache(op);
    IMSRGProfiler::timer["MPI_GatherOperatorToRoot"] += omp_get_wtime() - t_start;
  }

  double TwoBodyNorm(const TwoBodyME& two_body)
  {
    if (!OwnerOnlyStorageEnabled())
      return two_body.Norm();

    double norm_squared = 0.0;
    if (two_body.IsAllocated())
    {
      EnsureChannelOwnership(*two_body.modelspace);
      for (const auto& itmat : two_body.MatEl)
      {
        const auto ch_bra = itmat.first[0];
        if (TwoBodyChannelOwner(*two_body.modelspace, ch_bra) != Rank())
          continue;

        const arma::mat& matrix = itmat.second;
        int Jbra = two_body.modelspace->GetTwoBodyChannel(ch_bra).J;
        int Jket = two_body.modelspace->GetTwoBodyChannel(itmat.first[1]).J;
        int degeneracy = (2 * Jket + 1) * (std::min(Jbra, Jket + two_body.rank_J) - std::max(-Jbra, Jket - two_body.rank_J) + 1);
        double weighted_norm = arma::norm(matrix, "fro") * degeneracy;
        norm_squared += (ch_bra == itmat.first[1]) ? weighted_norm * weighted_norm : 2 * weighted_norm * weighted_norm;
      }
    }

    AllreduceInPlace(norm_squared);
    return std::sqrt(norm_squared);
  }

  double OneBodyNorm(const Operator& op)
  {
    return op.OneBodyNorm();
  }

  double TwoBodyNorm(const Operator& op)
  {
    return TwoBodyNorm(op.TwoBody);
  }

  double ThreeBodyNorm(const Operator& op)
  {
    return op.ThreeBodyNorm();
  }

  double Norm(const Operator& op)
  {
    if (!Enabled())
      return op.Norm();

    if (op.IsNumberConserving())
    {
      double n1 = OneBodyNorm(op);
      double n2 = TwoBodyNorm(op);
      double n3 = ThreeBodyNorm(op);
      return std::sqrt(n1 * n1 + n2 * n2 + n3 * n3);
    }

    return op.Norm();
  }

  double MP2Energy(const Operator& op)
  {
    if (!OwnerOnlyStorageEnabled())
      return const_cast<Operator&>(op).GetMP2_Energy();

    ModelSpace& modelspace = *op.GetModelSpace();
    EnsureChannelOwnership(modelspace);

    double emp2 = 0.0;
    std::vector<index_t> particles_vec(modelspace.particles.begin(), modelspace.particles.end());
    int nparticles = static_cast<int>(particles_vec.size());
    for (int ii = 0; ii < nparticles; ++ii)
    {
      index_t i = particles_vec[ii];
      double ei = op.OneBody(i, i);
      Orbit& oi = modelspace.GetOrbit(i);
      for (auto& a : modelspace.holes)
      {
        Orbit& oa = modelspace.GetOrbit(a);
        double ea = op.OneBody(a, a);
        if (static_cast<int>(i % Size()) == Rank() && std::abs(op.OneBody(i, a)) > 1e-9)
          emp2 += (oa.j2 + 1) * oa.occ * op.OneBody(i, a) * op.OneBody(i, a) / (op.OneBody(a, a) - op.OneBody(i, i));

        for (index_t j : modelspace.particles)
        {
          if (j < i)
            continue;
          double ej = op.OneBody(j, j);
          Orbit& oj = modelspace.GetOrbit(j);
          for (auto& b : modelspace.holes)
          {
            if (b < a)
              continue;
            Orbit& ob = modelspace.GetOrbit(b);
            if ((oi.l + oj.l + oa.l + ob.l) % 2 > 0)
              continue;
            if ((oi.tz2 + oj.tz2) != (oa.tz2 + ob.tz2))
              continue;

            double eb = op.OneBody(b, b);
            double denom = ea + eb - ei - ej;
            int Jmin = std::max(std::abs(oi.j2 - oj.j2), std::abs(oa.j2 - ob.j2)) / 2;
            int Jmax = std::min(oi.j2 + oj.j2, oa.j2 + ob.j2) / 2;
            int dJ = 1;
            if (a == b || i == j)
            {
              Jmin += Jmin % 2;
              dJ = 2;
            }

            for (int J = Jmin; J <= Jmax; J += dJ)
            {
              int parity_bra = (oa.l + ob.l) % 2;
              int parity_ket = (oi.l + oj.l) % 2;
              int tz_bra = (oa.tz2 + ob.tz2) / 2;
              int tz_ket = (oi.tz2 + oj.tz2) / 2;
              int ch_bra = modelspace.GetTwoBodyChannelIndex(J, parity_bra, tz_bra);
              int ch_ket = modelspace.GetTwoBodyChannelIndex(J, parity_ket, tz_ket);
              std::array<std::size_t, 2> key{
                static_cast<std::size_t>(std::min(ch_bra, ch_ket)),
                static_cast<std::size_t>(std::max(ch_bra, ch_ket))};
              if (!OwnsTwoBodyMatrix(modelspace, key))
                continue;

              double tbme = op.TwoBody.GetTBME_J_norm(J, a, b, i, j);
              if (std::abs(tbme) > 1e-9)
                emp2 += (2 * J + 1) * oa.occ * ob.occ * tbme * tbme / denom;
            }
          }
        }
      }
    }

    AllreduceInPlace(emp2);
    return emp2;
  }
}
