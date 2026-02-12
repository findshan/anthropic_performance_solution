# Changelog

All notable changes to this project will be documented here.

## [3.2] - Pre-loop & Scheduling Optimizations
- **Performance**: ~**1388 cycles** (`rounds=16, batch=256`)
  - **1.9% improvement** from v3.1 (1415 -> 1388 cycles)
  - **99.06% improvement** from baseline (147734 -> 1388 cycles, ~106x faster)
- **Pre-loop Optimization**: 
  - Moved initial `vload` ops to main loop to overlap with VALU execution (-15 cycles).
  - Replaced 32 `offset_base` constant loads with parallel ALU tree-doubling pointer computation (-12 cycles).
  - Deduplicated hash constants (shared broadcast for identical values) to save load/flow slots.
- **Scheduler**: Added round-robin tiebreaker for VALU candidates and removed conservative load-store conflict check to improve drain phase packing.

## [3.1] - First-Principles Scheduler Refactor
- **Performance**: ~**1415 cycles** (`rounds=16, batch=256`)
  - **0.7% improvement** from v3.0 (1425 -> 1415 cycles)
  - **99.0% improvement** from baseline (147734 -> 1415 cycles, ~104x faster)
- **Scheduler architecture**: Refactored `schedule_ops` from strict per-group gating to guarded cross-group issuing with explicit per-slot read/write-set analysis.
- **Correctness guardrails**:
  - Keep `store` on strict group boundaries.
  - Allow cross-group `load` only when no store has already been issued in the same cycle.
  - Preserve same-cycle hazard safety using scratch dependency checks per vector.
- **Kernel cleanup**:
  - Removed unused `v_four` constant preloading/broadcast from prelude.
  - Kept `prefetch_depth=2` with `depth>=3` dynamic gather path.

## [3.0] - Wave-based Scheduler & Vectorized Kernel
- **Performance**: ~**1425 cycles** (`rounds=16, batch=256`)
  - **15% improvement** from v2.0 (1678 → 1425 cycles)
  - **90.4% improvement** from baseline (147734 → 1425 cycles, ~104x faster)
- **Architecture**: Introduced "Wave-Based" scheduler, scheduling similar instructions across vectors to maximize VALU (6 slots) utilization; rewrote `schedule_ops` to support wave scheduling
- **Implementation**: Fully vectorized Hash pipeline with instruction fusion; used ALU (12 slots) for parallel pointer updates

## [2.0] - Advanced Scheduler & Depth-3 Optimization
- **Performance**: ~**1678 cycles**
  - **5.3% improvement** from v1.0 (1771 → 1678 cycles)
  - **98.9% improvement** from baseline (147734 → 1678 cycles, ~88x faster)
- **Scheduler**: Introduced multi-pass scheduling with backfill strategies; packed initialization instructions before pause loop into VLIW bundles
- **Optimization**: Depth-3 gather rollback and wrap index pruning; flow vselect index update moved to reduce VALU peak usage
- **Correctness**: Rolled back depth-3 mux to gather to fix WAR (Write-After-Read) hazards

## [1.0] - Baseline to Production
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
