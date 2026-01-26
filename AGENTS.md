# Repository Guidelines

## Project Structure & Module Organization
- `perf_takehome.py` holds the kernel builder and the main test harness; most performance work happens here.
- `problem.py` defines the simulator, instruction set, and data model used by the kernel.
- `tests/` contains submission checks (notably `tests/submission_tests.py`). Do not modify this folder when benchmarking.
- `trace/` stores Perfetto/Chrome trace outputs; `trace_any.py` can analyze them.
- `architecture/` and `optimizations/` document design notes and optimization plans; `CHANGELOG.md` tracks outcomes.

## Build, Test, and Development Commands
- `python perf_takehome.py` runs the in-file unittest suite for local iteration.
- `python perf_takehome.py Tests.test_kernel_cycles` runs the cycle test only.
- `python perf_takehome.py Tests.test_kernel_trace` emits a trace for Perfetto-based debugging.
- `python tests/submission_tests.py` runs official correctness/speed thresholds.
- `python trace_any.py trace/trace.json` summarizes a saved trace (use `--top 20` for hotspots).

## Coding Style & Naming Conventions
- Python, 4-space indentation, ASCII-only unless the file already contains Unicode.
- Use `snake_case` for functions/variables, `UpperCamelCase` for classes, and `test_` prefixes for tests.
- Prefer small, well-named helpers over large inline blocks; add brief comments only when the logic is non-obvious.

## Testing Guidelines
- Tests use `unittest`. Add new tests under `tests/` and keep them deterministic.
- Validate submissions with `python tests/submission_tests.py` and keep `tests/` unchanged (`git diff origin/main tests/` should be empty).

## Commit & Pull Request Guidelines
- Commit messages are short and descriptive; recent history favors a `v0.x` prefix (e.g., `v0.6 round-local scratch residency`).
- PRs should include: a concise description, cycle counts before/after, tests run, and any updated diagrams or changelog entries.
- If you update optimization docs, include before/after diagrams and concrete impact in `CHANGELOG.md`.
