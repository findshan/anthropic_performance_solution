# Rules

rules 2
- Mermaid: quote labels like `["..."]` when they contain spaces or punctuation; avoid parentheses, slashes, plus/minus, and HTML; prefer `\n` for line breaks; keep Gantt task names alphanumeric/underscore only.
- Changelog: always include concrete benefits/impact (e.g., performance gains, reduced cycles, risk reduction), not just a list of optimizations.
- Optimization docs: every expected benefit must include an estimated cycles2 delta, and each version must include before/after comparison diagrams.
- Optimization role: act as a high-quality performance optimization expert; prioritize correctness, follow stated constraints, and target cycles2 under 1200 where feasible.
- Optimization visuals: diagrams must make the specific optimization obvious at a glance and reduce reviewer decision effort.
