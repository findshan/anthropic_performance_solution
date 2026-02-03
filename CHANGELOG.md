# Changelog

## [3.0] - Wave-based Scheduler & Vectorized Kernel
- **性能变化**: `tests/submission_tests.py` 约 **1425 cycles**（`rounds=16, batch=256`），相对 2.5 的 **1634 cycles** 大幅下降，保持正确性。
- **架构升级**: 引入 "Wave-Based" 调度器，跨向量调度同类指令，最大化 VALU (6 slots) 利用率；重写 `schedule_ops` 以支持波次调度。
- **实现要点**: 全向量化 Hash 流水线支持融合指令；使用 ALU (12 slots) 并行计算指针更新；新增 `architecture/v3.0.md` 文档。

## [3.0] - Wave-based Scheduler & Vectorized Kernel (2026-01)
- **Performance**: ~**1425 cycles** (`rounds=16, batch=256`)
  - **15% improvement** from v2.0 (1678 → 1425 cycles)
  - **90.4% improvement** from baseline (147734 → 1425 cycles, ~104x faster)
- **Architecture**: Introduced "Wave-Based" scheduler, scheduling similar instructions across vectors to maximize VALU (6 slots) utilization; rewrote `schedule_ops` to support wave scheduling
- **Implementation**: Fully vectorized Hash pipeline with instruction fusion; used ALU (12 slots) for parallel pointer updates

## [2.0] - Advanced Scheduler & Depth-3 Optimization (2026-01)
- **Performance**: ~**1678 cycles**
  - **5.3% improvement** from v1.0 (1771 → 1678 cycles)
  - **98.9% improvement** from baseline (147734 → 1678 cycles, ~88x faster)
- **Scheduler**: Introduced multi-pass scheduling with backfill strategies; packed initialization instructions before pause loop into VLIW bundles
- **Optimization**: Depth-3 gather rollback and wrap index pruning; flow vselect index update moved to reduce VALU peak usage
- **Correctness**: Rolled back depth-3 mux to gather to fix WAR (Write-After-Read) hazards

## [1.0] - Baseline to Production (2026-01)
- **Performance**: ~**1771 cycles**
  - **88% improvement** from baseline (147734 → 1771 cycles, ~83x faster)
- **Major Optimizations**:
  - **VLIW Scheduler**: Implemented dependency-aware list scheduling (dropped from ~2402 to 1771 cycles, ~26% improvement at this stage)
  - **Depth-aware optimization**: Skipped index wrap checks on non-terminal depths; simplified depth-0 index update
  - **Grouped hash pipeline**: 6-way pipeline packing for VALU slots per stage; round-depth specialization for depth 0/1 to avoid gather loads on root/first-level nodes
  - **Load overlap**: Interleaved next-group gather loads during current-group hash/index computation
  - **Round-local scratch residency**: Reordered loops to keep idx/val in scratch across rounds with single load/store
  - **Hash fusion**: Fused eligible hash stages with vector multiply_add to reduce VALU ops
  - **Load smoothing**: Interleaved next-batch load_offset across hash stages
  - **Pipeline optimization**: Double-buffered vector scratch to overlap gather with hash/commit; pipelined vector load/compute schedule
  - **Vectorization**: Added SIMD vector path for inner loop with scalar tail fallback
  - **Kernel optimization**: Conservative VLIW slot packing; reused computed input addresses
