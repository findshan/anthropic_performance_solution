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

## [0.8] - Depth-aware wrap elimination pass
- Skipped idx wrap checks on non-terminal depths; simplified depth-0 idx update.
- Reduced valu ops from ~9780 to ~8756 and lowered cycles to 2423.
- Test cycles: `tests/submission_tests.py` -> 2423 cycles.

## [0.7] - Grouped hash pipeline with overlapped gather
- Grouped vector blocks into 6-way pipelines to pack VALU slots per stage and reduce bundle count.
- Added round-depth specialization for depth 0/1 to avoid gather loads on root/first-level nodes.
- Interleaved next-group gather loads during current-group hash/idx to raise load/valu utilization.
- Test cycles: `tests/submission_tests.py` -> 2525 cycles.

## [0.6] - Round-local scratch residency pass
- Reordered loops to keep idx/val in scratch across rounds with single load/store.
- Kept hash fusion and parity bit optimizations in vector/scalar paths.
- Test cycles: `tests/submission_tests.py` -> 10380 cycles.

## [0.5] - Hash fusion and load smoothing pass
- Fused eligible hash stages with vector multiply_add to reduce valu ops.
- Interleaved next-batch load_offset across hash stages to smooth load pressure.
- Replaced parity modulo with bitwise parity in vector and scalar paths.
- Test cycles: `tests/submission_tests.py` -> 9324 cycles.

## [0.4] - Deep pipeline optimization pass
- Interleaved next-batch gather into hash stages to overlap load/valu.
- Prefetched next-batch vload/addr setup before staged gather.
- Test cycles: `tests/submission_tests.py` -> 10862 cycles.

## [0.3] - Pipeline optimization pass
- Added double-buffered vector scratch to overlap gather with hash/commit.
- Introduced pipelined vector load/compute schedule for batch chunks.
- Test cycles: `tests/submission_tests.py` -> 12846 cycles.

## [0.2] - Vectorization pass
- Added SIMD vector path for inner loop with scalar tail fallback.
- Switched vector wrap/branch logic to ALU/VALU to reduce flow pressure.
- Test cycles: `tests/submission_tests.py` -> 13389 cycles.

## [0.1] - Kernel optimization pass
- Added conservative VLIW slot packing to reduce instruction bundle count.
- Reused computed input addresses to avoid redundant ALU address calculations.
- Made debug compares optional to keep the packed schedule deterministic.
- Test cycles: `tests/submission_tests.py` -> 98582 cycles.
- Trace output now defaults to `trace/trace.json` when enabled.

## [0.0] - Initial import
- Imported upstream Anthropic original performance take-home.
- Added `architecture/` with overview and diagrams.
- Added `optimizations/v1.0-optimizations.md` planning document.
- No code changes to kernel or simulator; baseline preserved.
