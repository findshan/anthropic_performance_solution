# Changelog

All notable changes to this project will be documented here.

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
