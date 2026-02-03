# Changelog

## [3.0] - Wave-based Scheduler & Vectorized Kernel
- **性能变化**: `tests/submission_tests.py` 约 **1425 cycles**（`rounds=16, batch=256`），相对 2.5 的 **1634 cycles** 大幅下降，保持正确性。
- **架构升级**: 引入 "Wave-Based" 调度器，跨向量调度同类指令，最大化 VALU (6 slots) 利用率；重写 `schedule_ops` 以支持波次调度。
- **实现要点**: 全向量化 Hash 流水线支持融合指令；使用 ALU (12 slots) 并行计算指针更新；新增 `architecture/v3.0.md` 文档。

## [2.2] - Engine-pressure scheduler & hash ALU offload
- **性能变化**: `tests/submission_tests.py` 约 **1676 cycles**（`rounds=16, batch=256`），较 1678 进一步下降，正确性保持。
- **调度器优化**: 增加按引擎压力的优先放置，再进行 backfill，提高每周期 slot 填充率。
- **哈希压缩策略**: 对 hash 的 op1/op3 采用少量 ALU 旁路，缓解 VALU 峰值压力。

## [2.3] - Multi-slot engine fill & hash ALU balance
- **性能变化**: `tests/submission_tests.py` 约 **1669 cycles**（`rounds=16, batch=256`），继续下降并保持正确性。
- **调度器优化**: 在引擎压力优先放置中，单引擎可多槽填充，减少同周期空位。
- **哈希链路优化**: 将 op2 的 ALU 旁路与 op1/op3 对齐（中等比例），进一步缓和 VALU 峰值。

## [2.5] - Flow vselect idx update
- **性能变化**: `tests/submission_tests.py` 约 **1634 cycles**（`rounds=16, batch=256`），相对 2.4 的 **1660 cycles** 再降约 **26 cycles**，保持正确性。
- **Slot 结构变化**: `trace_any.py` 显示 **VALU ≈ 8024**, **LOAD ≈ 2187**, **FLOW ≈ 450**；将 idx 更新中的 `+1` 计算改为 `flow vselect`，减少 VALU 峰值并让 FLOW 有效参与填槽。
- **实现要点**: `offset = vselect(parity, two_v, one_v)` 后续仍走 `idx = idx * 2 + offset`，不改变数值语义。

## [2.4] - Prelude VLIW packing
- **性能变化**: `tests/submission_tests.py` 约 **1660 cycles**（`rounds=16, batch=256`），进一步下降并保持正确性。
- **早期利用率提升**: 将 pause 之前的初始化/预载指令打包为 VLIW 段，减少起始空槽。

## [2.1] - Scheduler 2.0 backfill & depth3 safe mux
- **性能变化**: `tests/submission_tests.py` 约 **1678 cycles**（`rounds=16, batch=256`），从 1685 回落并保持正确性。
- **调度器 2.0**: 每周期多轮放置与 backfill，先按关键路径优先，再按引擎空余比例填洞，减少空槽与 WAR 因序阻塞。
- **深度优化状态**: depth3 安全 mux 复活并稳定；depth4 mux 代码保留但默认关闭以避免性能回退。

## [2.0] - Depth3 gather rollback & wrap idx prune
- **性能变化**: cycles 约从 ~1680 降至 **~1678**（`rounds=16, batch=256`），维持正确性。
- **正确性修复**: 临时禁用 depth3 mux，回退到 gather，规避 WAR 重排导致的错误选择。
- **指令削减**: 在 wrap 与最后一轮跳过 idx 更新（不影响最终值），减少部分 VALU slot。

## [1.0] - VLIW 调度器升级 & 架构深度文档化
- **性能突破**: 实现了从贪心打包到**依赖感知列表调度 (List Scheduling)** 的质变，Cycle 数从 ~2402 降至 **1771** (提升 ~26%)。
- **调度器优化**:
    - 精确建模 RAW, WAW, WAR 依赖，支持写后读 (WAR) 同周期发射。
    - 引入基于关键路径 (Rank) 的优先级排序，最大化流水线填充率。
    - 优化 Slot 分配策略，平衡 Load 与 VALU 压力。
- **架构文档化 (中文)**:
    - 编写了详尽的 [v1.0-vliw-scheduler.md](file:///Users/findshan/Documents/Projects/ai_PROJECT/codex_promote/source_repo/architecture/v1.0-vliw-scheduler.md)，包含依赖图、调度流程图及流水线填充对比图。
    - 增加了理论性能分析，推导了 1525 cycle 的下界极限，并对比了实际性能表现。
- **工程改进**:
    - 删除了冗余的初始 idx_buf 清零和常量加载，减少了基础指令开销。
    - 优化了 batch=256 场景下的尾部处理逻辑。

## [0.8] - 深度感知回绕消除 (Depth-aware wrap elimination)
- 跳过了非终端深度的索引回绕检查；简化了深度0的索引更新。
- 减少了约 1000 个 VALU 操作，Cycle 降至 2423。

## [0.7] - 分组哈希流水线与重叠 Gather
- 将向量块分组为 6 路流水线，以每个阶段打包 VALU slot 并减少 bundle 数量。
- 增加了深度 0/1 的专门化处理，避免根节点/第一层节点的 gather 加载。
- 在当前组的哈希/索引计算期间交错下一组的 gather 加载，提高 Load/VALU 利用率。
- 测试 Cycles: 2525。

## [0.6] - 轮次局部暂存驻留 (Round-local scratch residency)
- 重排序循环以在轮次间保持 idx/val 在暂存区，仅需单次加载/存储。
- 保留了向量/标量路径中的哈希融合和奇偶位优化。
- 测试 Cycles: 10380。

## [0.5] - 哈希融合与负载平滑
- 将符合条件的哈希阶段与向量 multiply_add 融合，减少 VALU 操作。
- 在哈希阶段间交错下一批次的 load_offset 以平滑 Load 压力。
- 在向量和标量路径中用位运算奇偶性替换了取模奇偶性。
- 测试 Cycles: 9324。

## [0.4] - 深度流水线优化
- 将下一批次的 gather 交错到哈希阶段以重叠 Load/VALU。
- 在分阶段 gather 之前预取下一批次的 vload/addr 设置。
- 测试 Cycles: 10862。

## [0.3] - 流水线优化
- 增加了双缓冲向量暂存区以重叠 gather 与 hash/commit。
- 引入了批处理块的流水线向量加载/计算调度。
- 测试 Cycles: 12846。

## [0.2] - 向量化 (Vectorization)
- 增加了内层循环的 SIMD 向量路径，带有标量尾部回退。
- 将向量回绕/分支逻辑切换为 ALU/VALU 以减少 Flow 压力。
- 测试 Cycles: 13389。

## [0.1] - 内核优化
- 增加了保守的 VLIW slot 打包以减少指令 bundle 数量。
- 重用计算出的输入地址以避免冗余的 ALU 地址计算。
- 使调试比较可选，以保持打包调度的确定性。
- 测试 Cycles: 98582。
- 启用时 Trace 输出现在默认为 `trace/trace.json`。

## [0.0] - 初始导入
- 导入上游 Anthropic 原始性能 take-home。
- 添加了 `architecture/` 概览和图表。
- 添加了 `optimizations/v1.0-optimizations.md` 规划文档。
- 内核或模拟器无代码更改；保留基线。
