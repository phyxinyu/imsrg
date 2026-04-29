# MPI-IMSRG(2) 标量版计划

## Summary
- 第一版只支持标量、number-conserving、particle rank <= 2 的 Magnus IMSRG(2) flow；暂不考虑张量、dagger、PV、IMSRG(3) 和外部算符变换。
- 单体项和零体项每个 rank 都保存完整副本；二体项按 scalar two-body channel 分布存储和计算。
- `Solve_magnus_euler()` 的外层流程保持同步串行；MPI 只进入 `Generator`、`CommutatorScalarScalar`、BCH 内部的 channel 级计算。

## Key Changes
- 增加可选 MPI 构建与运行开关：`IMSRG_USE_MPI`、运行参数 `mpi_imsrg2=true`。未启用时现有串行/OpenMP 行为不变。
- 新增 MPI 辅助层：rank/size、channel owner 映射、`Allreduce`、`Alltoallv`、root gather/write 等工具。
- 二体 channel owner 使用静态负载均衡：按 `nKets(ch)^2` 估算代价，贪心分配到 rank；每个 rank 只保留自己拥有的 `TwoBodyME` channel block。
- MPI 模式下新增分布式 IMSRG(2) 路径：
  - `Generator::Update`：单体生成器每 rank 计算完整副本；二体 `Eta` 只在 owner rank 计算。
  - `CommutatorScalarScalar`：零体/单体贡献先局部累加，再 `MPI_Allreduce`；二体输出只在 owner rank 写入。
  - BCH：复用 MPI commutator，所有 norm 和收敛判据使用全局归约。

## Pandya Handling
- `comm222_phss` 不全量复制 `Z_bar`。
- 对每个 cross-coupled channel，相关 rank 先计算自己能提供的 Pandya-transformed 矩阵元片段，并发送给该 CC channel 的 owner。
- CC owner 组装本 channel 的临时 `Xbar/Ybar/Zbar`，做矩阵乘法。
- 逆 Pandya 阶段不直接写全局 `Z`；而是生成 `(target_ch, ibra, iket, value)` contribution packet，通过 `MPI_Alltoallv` 发给 `target_ch` 的 owner。
- target owner 合并重复贡献后写入本地 `Z.TwoBody.GetMatrix(target_ch)`。

## Test Plan
- 非 MPI 构建：现有测试和典型运行结果保持不变。
- MPI 单 rank：`mpi_imsrg2=true` 与串行结果逐项比较。
- MPI 多 rank：2、4、8 rank 跑小模型空间，比较 flow file 中 `E0`、`eta norm`、`omega norm`、最终二体矩阵元，容差按浮点归约误差设为约 `1e-10` 到 `1e-8`。
- 分项测试 `comm220ss`、`comm121ss`、`comm122ss`、`comm222_pp_hh_221ss`、`comm222_phss`，尤其单独验证 Pandya 通信结果。
- 最终写文件时由 rank 0 gather 二体 channel，确认输出格式仍兼容现有串行读写。

## Assumptions
- 第一版 MPI 只覆盖 `IMSRG3=false`、标量 Hamiltonian flow；遇到张量算符、dagger、PV 或 IMSRG(3) 时直接报错或回退串行。
- 初始 HF/normal-ordering 阶段暂时保持现有实现；进入 `IMSRGSolver` 后再分发二体 channel。
- `write_omega`/scratch 在 MPI 第一版中先禁用，后续再设计 per-rank 或 gathered Omega 存储。
- 继续保留 OpenMP，推荐运行方式是 MPI rank 间分 channel、rank 内 OpenMP 加速矩阵运算。
