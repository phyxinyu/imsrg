# Distributed 3N NO2B MPI Normal Ordering

## Goal

Reduce memory pressure when reading NO2B 3N inputs under MPI. The 3N input is only needed before normal ordering; after normal ordering the MPI IMSRG(2) path evolves a 0B/1B/2B Hamiltonian.

## Implementation Strategy

- Limit the temporary `ReadFile()` buffer to a small fixed chunk instead of allocating the entire 3N file-sized vector.
- Under MPI, store only NO2B 3N storage channels that can contribute to two-body channels owned by the local rank.
- In normal ordering, only compute two-body channels owned by the local rank.
- After normal ordering, allreduce scalar and one-body pieces where they are partial; keep two-body storage owner-only.
- Deallocate input 3N storage immediately after the NO2B Hamiltonian has been formed.

## Current Conservative Ownership

The NO2B 3N storage is organized by `(J2, P2, J1, P1, T3)`, while the output Hamiltonian is organized by two-body channels. The first implementation stores a 3N storage channel on ranks that own at least one two-body channel with matching `J2` and `P2`.

This is conservative: it may duplicate some 3N channels across ranks, but it avoids missing matrix elements. A later refinement can include `Tz` and contraction-orbit constraints to reduce duplication further.

## Validation

- Build with `IMSRG_USE_MPI=ON`.
- Run `3bme_type=no2b` with `mpi_imsrg2=true`.
- Confirm logs report the intended MPI rank count, not `1 ranks`.
- Compare small-model outputs against serial/no-MPI or single-rank results.
- Check memory reduction against previous runs that allocated a full-file-sized read buffer on every rank.
