# From Human–AI Collaboration to Autoresearch

This project family has two stages, both using Claude Opus 4.5.

## Stage 1 — Interactive collaboration

In February 2026, we worked with Claude Opus 4.5 interactively on Anthropic's
Performance Take-home. The human supplied architectural reasoning, constraints,
and experiment direction; the model implemented and tested many of the low-level
changes. This produced the public `anthropic_performance_solution` line, reaching
the 1368-cycle range, but did not cross the then-public 1363-cycle reference.

## Stage 2 — Autoresearch

We then changed the workflow rather than changing the model. We built a minimal
autoresearch harness: the agent edits only the kernel, runs one experiment,
records the result, keeps improvements, and resumes from a file-backed history.
With the same Claude Opus 4.5 model operating through this loop, the `opt_work`
track reached 1361 cycles and passed all nine official tests.

## What this repository is

`perf-evolve` is the reusable experiment harness. The benchmark solution is a
separate project. The central claim is about workflow: a constrained benchmark,
an auditable measurement loop, and persistent experiment memory can turn an
interactive optimization process into an increasingly autonomous one.

## Reproducibility and provenance

The exact model/interface details for each historical run were not captured in a
machine-readable manifest at the time. Until those records are reconstructed,
claims should say `Claude Opus 4.5-assisted` and distinguish interactive from
autoresearch execution, without claiming that every edit was fully autonomous.

## Publication boundary

The upstream challenge asks participants not to publish complete solutions to
avoid spoilers. Before publishing the 1361-cycle kernel, decide whether to share
the full implementation, a redacted benchmark, or only the harness and analysis.
