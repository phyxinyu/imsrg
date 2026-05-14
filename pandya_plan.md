# Pandya MPI Key-List Prefetch Plan

## Summary

- Goal: reduce the temporary X/Y two-body storage used by the MPI Pandya path in `comm222_phss`.
- Keep the current owner-only design for resident scalar two-body channels, cc-channel `Z_bar`, and inverse-Pandya contribution packets.
- Replace the current Pandya-stage full legal scalar matrix prefetch with a rank-local key-list prefetch: each rank only requests scalar two-body matrices needed to compute its owned cc channels.
- Preserve the existing `PrefetchTwoBodyMatrices(...)` broadcast/global-union behavior for generator denominator and other stages.

## Current State

- `Z_bar` is already distributed by cross-coupled channel owner.
- Inverse Pandya already produces contribution packets and sends them to the target scalar channel owner via `Alltoallv`.
- The remaining coarse part is input access: before `comm222_phss`, both `X_work` and `Y_work` currently call full `PrefetchTwoBodyMatrices(...)`, so each rank temporarily receives all legal scalar two-body matrices.
- After `comm222_phss`, `ClearTwoBodyCache(X_work/Y_work)` removes non-owner temporary matrices and restores owner-only resident storage.

## Key Changes

- Add a rank-local prefetch API in `MpiSupport`:
  - `PrefetchTwoBodyMatricesForLocalReads(Operator&, const std::vector<std::array<std::size_t, 2>>& local_keys)`.
  - `local_keys` means only the matrices this rank will read in the next phase.
  - Owned matrices are not requested; missing non-owner matrices are requested from their scalar channel owner.

- Implement rank-local prefetch with two `Alltoallv` rounds:
  - Request round: requester sends canonical matrix key ids to each key owner.
  - Response round: owner sends matrix payloads only to ranks that requested them.
  - Receiver inserts received matrices into `op.TwoBody.MatEl` as temporary cache entries.
  - Keep existing chunking/size safeguards for large matrix payloads where practical.

- Add a response helper rather than overloading the current flat helper too heavily:
  - Either add `AlltoallvDoublesByRank(...)` returning `std::vector<std::vector<double>>`, or keep source-rank boundaries through explicit counts/displacements.
  - Response packets should include enough metadata to reconstruct and validate each matrix, for example `[key_id, n_rows, n_cols, data...]`.

- Add a Pandya input key collector in `Commutator.cc`:
  - Traverse only cc channels owned by the current rank.
  - Mirror the index/J selection in `DoPandyaTransformation_SingleChannel_XandY`.
  - For each nonzero candidate access to `GetTBME_J_norm_twoOps(..., J_std, J_std, c, b, a, d, ...)`, derive the scalar matrix key from `(c,b)` and `(a,d)`.
  - Canonicalize, sort, and unique the resulting key list.

- Change the `comm222_phss` call site:
  - Replace full prefetch for `X_work/Y_work` with rank-local Pandya key-list prefetch.
  - Keep `comm222_phss(...)` itself and `AddInversePandyaTransformation(...)` numerically unchanged.
  - Keep `ClearTwoBodyCache(X_work/Y_work)` after the Pandya phase.

## Communication Cost

- Current full prefetch cost is roughly: all needed scalar matrices broadcast to all ranks.
- Rank-local prefetch cost is roughly: each scalar matrix sent only to ranks that actually need it for owned cc channels.
- For small spaces such as `emax=4`, rank-local prefetch may be similar or slightly slower because the two `Alltoallv` rounds and packing overhead are not free.
- For larger spaces or higher `np`, rank-local prefetch should reduce temporary memory and avoid many unnecessary matrix transfers.
- Keep the old full-prefetch path available as a fallback/debug option until benchmarks confirm the new path is better on target cases.

## Test Plan

- Build and static checks:
  - `git diff --check`
  - `cmake --build build -j 4`
  - `cmake --build build-mpi -j 4`

- MPI correctness:
  - `python3 work/scripts/mpi_test.py --np 1 --omp 1 --smax 0.2 --dsmax 0.01`
  - `python3 work/scripts/mpi_test.py --np 2 --omp 1 --smax 0.2 --dsmax 0.01`
  - `python3 work/scripts/mpi_test.py --np 4 --omp 1 --smax 0.2 --dsmax 0.01`
  - Compare flow file physical columns and final `.snt`; walltime and memory may differ.

- MPI storage behavior:
  - During Pandya, non-owner X/Y matrices should be limited to the collected Pandya key list.
  - After `ClearTwoBodyCache`, X/Y `MatEl` should again contain only owner matrices.
  - `Z` should still be written only by scalar channel owners through inverse-Pandya contribution packets.

## Assumptions

- This plan only targets scalar IMSRG(2) `comm222_phss`.
- It does not change Pandya formulas, `Z_bar` multiplication, or inverse-Pandya contribution packet semantics.
- Existing broadcast/global-union `PrefetchTwoBodyMatrices(...)` remains available and unchanged.
- Rank-local prefetch is an optimization for memory/communication footprint; numerical output must remain unchanged.
