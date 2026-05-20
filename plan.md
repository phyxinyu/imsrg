# MPI-IMSRG(2) 代码改进计划

## Context

当前 MPI-IMSRG(2) owner-only v2 已经实现：

- 常驻 two-body storage 按 owner channel 分布。
- 普通 scalar commutator 不再全量 prefetch 输入。
- Pandya inverse 已使用 contribution packet 合并到目标 owner。
- `WriteFlowStatus` 的 `E(MP2)` 已改为分布式计算。
- generator denominator 已从阶段级全量 prefetch 改为 matrix-key-list prefetch。

本计划只记录后续可改进点；不再把已经完成的 key-list generator prefetch 当作待办。

---

## 本轮已实施

- `MPI_Init` 返回值检查已添加。
- `AllreduceInPlace(arma::mat&)` 与 `BroadcastMatrixFromRank` 已共用大矩阵 chunk helper。
- `comm122ss_slower` / `comm222_phss_slower` 在 MPI 模式下已显式 `Abort`。
- `WriteFlowStatus` 已添加 collective 语义注释。
- `ClearTwoBodyCache` 已添加注释说明它只清除临时 prefetched non-owner matrices。
- 非 MPI-aware solver 子函数已添加 direct-call MPI guard。

---

## 剩余改进点

- 给 MPI 通信路径补独立 profiler timer，覆盖 prefetch、clear cache、Bcast、Alltoallv、Allreduce 和 Gather，避免通信开销继续隐藏在上层 timer 中。
- 优先优化 `comm222_phss` 前的 `X_work` / `Y_work` TwoBody prefetch，避免默认全量 matrix-key 广播；当前日志显示 `comm222_phss` 自身没有变快，但 `CommutatorScalarScalar` 和 `system time` 明显增加。
- 评估将 prefetch 的 owner-to-all `MPI_Bcast(MPI_COMM_WORLD)` 改为 owner-to-requesters 通信；当前全体广播实现简单、collective 顺序稳定，但只要任意 rank 需要某个矩阵，所有 rank 都会接收和分配该矩阵，稀疏需求下通信和内存浪费明显。
- 优化 inverse Pandya 的 `Alltoallv` contribution packet，减少包大小和包数量；可评估整数索引与 double 数据分离传输，以及发送前合并相同 `(ch, ibra, iket)` 贡献。
- 合并小规模 `ZeroBody` / `OneBody` Allreduce，减少 scalar commutator 和 generator update 中的 collective 次数。
- 优化 generator denominator prefetch 的 key 生成和复用，避免重复构造相同 matrix-key list。
- 低优先级评估最终 `GatherOperatorToRoot`；它只在输出阶段发生，除非输出阶段成为瓶颈，否则不作为第一批性能优化。

---

## 验证方式

本轮代码已完成以下验证：

- `git diff --check`
- `cmake --build build -j 4`
- `cmake --build build-mpi -j 4`
- `python3 work/scripts/mpi_test.py --np 1 --omp 1 --smax 0.02 --dsmax 0.01`
- `python3 work/scripts/mpi_test.py --np 2 --omp 1 --smax 0.02 --dsmax 0.01`
- 已对比 `np=1/2` flow file 的物理数值列；结果一致，wall time / memory 不同。

## Assumptions

- `plan.md` 只作为改进计划文档，不创建 `doc/mpi_code_improvements.md`。
- 当前 MPI owner-only v2 和 generator matrix-key-list prefetch 已视为现状。
- B/C 类维护性改动已实施；后续只剩 D 类性能优化。
