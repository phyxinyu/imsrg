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

## 剩余改进点（按优先级排序）

### 🔵 优先级 D：性能优化

**1. Pandya prefetch 从阶段级全量改为 key-list**

- 位置：[src/Commutator.cc](src/Commutator.cc)
- 当前状态：`comm222_phss` 前仍对 `X_work/Y_work` 做阶段内全量合法 matrix prefetch。
- 目标：只预取 Pandya transformation 实际访问的 `(ch_bra, ch_ket)` key 列表，减少通信量和峰值内存。
- 注意：逆 Pandya contribution packet 已经实现；这里优化的是 Pandya 正变换/输入读取阶段。

**2. generator denominator 从 matrix-key 级继续压缩**

- 位置：[src/Generator.cc](src/Generator.cc)
- 当前状态：`GetDenominatorMatrixKeys` 已实现 matrix-key-list prefetch，不再是阶段级全量 prefetch。
- 目标：进一步压缩到 element/value 级 denominator cache，或重写为 owner-local contribution 公式，继续降低 generator 阶段峰值内存。
- 注意：必须保持 `np=1` 与 `np=2/4` 的 `Eta`、flow status 和最终 TBME 数值一致。

---

## 建议实施顺序

1. **D1**：Pandya key-list prefetch。它仍是当前最大的阶段性内存/通信优化点。
2. **D2**：generator denominator element/value 级压缩。需要额外数值对比，建议单独立项。

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
