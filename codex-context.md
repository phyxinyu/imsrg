# MPI-IMSRG(2) 标量版计划

## Summary
- 第一版只支持标量、number-conserving、particle rank <= 2 的 Magnus IMSRG(2) flow；暂不考虑张量、dagger、PV、IMSRG(3) 和外部算符变换。
- 单体项和零体项每个 rank 都保存完整副本；二体项按 scalar two-body channel 分配 owner。当前 owner-only v2 已经避免常驻保存非 owner two-body channel；普通 scalar generator/commutator 不再全量 prefetch 输入，只有 Pandya/cross-coupled 路径保留显式阶段通信。
- `Solve_magnus_euler()` 的外层流程保持同步串行；MPI 只进入 `Generator`、`CommutatorScalarScalar`、BCH 内部的 channel 级计算。

## Implemented So Far
- 增加可选 MPI 构建与运行开关：`IMSRG_USE_MPI`、运行参数 `mpi_imsrg2=true`。未启用时现有串行/OpenMP 行为不变。
- 新增 MPI 辅助层：rank/size、channel owner 映射、`Barrier`、`Abort`、`Allreduce`、operator/two-body 归约等工具。
- 二体 channel owner 使用静态负载均衡：按 `nKets(ch)^2` 估算 two-body channel 代价，按 `2*nKets(ch)^2` 估算 cross-coupled channel 代价，贪心分配到 rank。
- `imsrg_mpi::Enabled()` 在 MPI 编译且运行参数启用时生效；`mpirun -np 1` 也会进入 MPI/owner-only 路径，方便单 rank 对比串行。非 MPI 构建下请求 `mpi_imsrg2=true` 会提示并回退串行。
- `imsrg++.cc` 已在程序入口初始化/收尾 MPI，并限制 MPI 模式只跑 `IMSRG3=false`、`method=magnus/magnus_euler`、`write_omega=false`、`scratch=""`、无外部算符变换。
- MPI 模式下已有的 IMSRG(2) 路径：
  - `Generator::Update`：单体生成器每 rank 计算完整副本；二体 `Eta` 只在 owner rank 计算。
  - `CommutatorScalarScalar`：零体/单体贡献先局部累加，再 `MPI_Allreduce`；二体贡献按 owner rank 分工计算并常驻保存 owner channel，不再在 owner-only 模式下把二体 `Z` allreduce 成完整副本。
  - `comm110ss`、`comm111ss`、`comm121ss`、`comm122ss`、`comm220ss`、`comm222_pp_hh_221ss`、`comm222_phss` 等路径已有 rank/channel 过滤。

## Latest Iteration
- 实现 owner-only v2 的普通项无预取路径：
  - 移除 `Generator::Update` 入口对 `H_s` 的全量 `PrefetchTwoBodyMatrices`；二体 `Eta` 只遍历/读取本 rank owner channel。
  - 移除 `CommutatorScalarScalar` 入口对 `X/Y` 的全量 prefetch；普通 scalar terms 直接在 owner-only 常驻二体矩阵上计算。
  - `comm222_phss` 调用前后单独 prefetch/clear `X/Y`，把远端输入缓存限制到 Pandya 阶段。
  - `comm121ss` 在 owner-only 模式下改为所有 rank 都计算全体 one-body 行的本地 two-body contribution，再由外层 one-body `Allreduce` 合并；非 owner-only MPI 仍保留按行分工。
  - `TwoBodyME::GetTBME*` 标量读取在 owner-only 且本 rank 无该矩阵时返回 0，表示“本 rank 对该分布式求和项没有贡献”；矩阵级 `GetMatrix()` 仍然对未 owned/未 prefetched 矩阵抛异常。
- 实现 owner-only 常驻二体存储 v1：
  - `MpiSupport` 新增 `SetOwnerOnlyStorage` / `OwnerOnlyStorageEnabled`、`RestrictOperatorToOwnedChannels`、`PrefetchTwoBodyMatrices`、`ClearTwoBodyCache`、`GatherOperatorToRoot`、`BroadcastMatrixFromRank` 等工具。
  - `TwoBodyME::Allocate()` 在 owner-only 模式下只分配本 rank 拥有的 scalar two-body channel；`GetMatrix()` 访问未 owned 且未 prefetch 的矩阵会直接抛异常，避免隐式 MPI collective 或静默读错。
  - `SetTBME` / `AddToTBME` / non-hermitian 写路径在 owner-only 模式下跳过非 owner channel，普通 `+=`/`-=` 也只更新本地已有矩阵。
  - `Generator::Update` 阶段显式 prefetch `H_s`，生成后只 allreduce 零体/单体，二体 `Eta` 保持 owner-only。
  - `CommutatorScalarScalar` 阶段显式 prefetch `X/Y` 临时工作副本，计算输出 `Z` 时只保留 owner channel，结束后清理输入临时缓存。
  - 程序进入 `IMSRGSolver` 前会开启 owner-only 并限制 `HNO`；最终写文件前调用 `GatherOperatorToRoot`，只有 rank 0 保留完整 gathered 副本并写输出。
  - `WriteFlowStatus` 中非 root 也参与 MPI-aware 范数和 prefetch collective，但只有 root 写 flow file/打印；`H_s.GetMP2_Energy()` 在 root gather 临时副本上计算。
- 在 `MpiSupport` 中新增 MPI-aware 范数 API：`Norm(const Operator&)`、`OneBodyNorm`、`TwoBodyNorm`、`ThreeBodyNorm`、`TwoBodyNorm(const TwoBodyME&)`。
- MPI 关闭时这些 API 直接调用原有串行范数；MPI 开启时二体范数只统计当前 rank 拥有的 channel，再对范数平方做全局 `Allreduce`，避免重复计算完整副本。
- `BCH::Standard_BCH_Transform` 和 `BCH::BCH_Product` 已改用 `imsrg_mpi::Norm()`，BCH 初始阈值、nested commutator 收敛判断和 BCH product 截断判断都使用全局范数。
- `IMSRGSolver::Solve_magnus_euler` 中的 `Eta`/`Omega` 范数、步长选择和 `omega_norm_max` 判断已改用 `imsrg_mpi::Norm()`。
- `IMSRGSolver::GatherOmega` 中 hunter 是否需要合并的判断也改用全局范数。

## Planned Work
- 把 Pandya 的 `PrefetchTwoBodyMatrices` 从当前“Pandya 阶段按合法 channel 全量预取”的 v2，细化为 Pandya 所需 key 列表预取，进一步降低 Pandya 阶段峰值内存。
- 增加真正的 `Alltoallv` contribution packet 工具，用于 cross-coupled/Pandya 路径把逆 Pandya 贡献发送到目标 two-body channel owner。
- 将 Pandya `Z_bar` 从当前“owner 计算后阶段内 broadcast 临时缓存”的 v1，推进到只在 cross-coupled owner 上保存并通过 packet 合并回 two-body owner。
- 补齐 MPI 小模型验证脚本：单 rank 对比串行，多 rank 对比 flow file 和最终矩阵元。

## Pandya Handling
- 当前 v1 中 `comm222_phss` 的 `Z_bar` 只由 cross-coupled channel owner 计算。
- 为了先保证核心 flow 可运行，逆 Pandya 前会把 `Z_bar` 按 cross-coupled owner broadcast 成阶段内临时缓存；`AddInversePandyaTransformation` 只写本 rank 拥有的目标 two-body channel，非 owner 写入跳过。
- 后续计划把这一步替换为真正的 contribution packet：逆 Pandya 生成 `(target_ch, ibra, iket, value)`，经 `MPI_Alltoallv` 发给 `target_ch` owner 合并。

## Test Plan
- 非 MPI 构建：现有测试和典型运行结果保持不变。
- MPI 单 rank：`mpi_imsrg2=true` 与串行结果逐项比较。
- MPI 多 rank：2、4、8 rank 跑小模型空间，比较 flow file 中 `E0`、`eta norm`、`omega norm`、最终二体矩阵元，容差按浮点归约误差设为约 `1e-10` 到 `1e-8`。
- 分项测试 `comm220ss`、`comm121ss`、`comm122ss`、`comm222_pp_hh_221ss`、`comm222_phss`，尤其单独验证 Pandya 通信结果。
- 最终写文件时由 rank 0 gather 二体 channel，确认输出格式仍兼容现有串行读写。

## Current Verification
- 非 MPI 当前构建通过：`cmake --build build -j 4`。
- 当前 `ctest --test-dir build --output-on-failure` 会失败，因为这个 build 没有生成/暴露 Python 模块 `pyIMSRG`，6 个 Python 测试都停在 `ModuleNotFoundError: No module named 'pyIMSRG'`。
- 当前本机环境未找到 `mpicxx`/`mpirun`，`cmake -DIMSRG_USE_MPI=ON` 配置失败于 `Could NOT find MPI_CXX`；多 rank 验证需要先安装或加载 MPI CXX 环境。

## Assumptions
- 第一版 MPI 只覆盖 `IMSRG3=false`、标量 Hamiltonian flow；遇到张量算符、dagger、PV 或 IMSRG(3) 时直接报错或回退串行。
- 初始 HF/normal-ordering 阶段暂时保持现有实现；进入 `IMSRGSolver` 后再分发二体 channel。
- `write_omega`/scratch 在 MPI 第一版中先禁用，后续再设计 per-rank 或 gathered Omega 存储。
- 继续保留 OpenMP，推荐运行方式是 MPI rank 间分 channel、rank 内 OpenMP 加速矩阵运算。
