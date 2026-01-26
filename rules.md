# Rules

rules 2
- Mermaid: quote labels like `["..."]` when they contain spaces or punctuation; avoid parentheses, slashes, plus/minus, and HTML; prefer `\n` for line breaks; keep Gantt task names alphanumeric/underscore only.
- Changelog: always include concrete benefits/impact (e.g., performance gains, reduced cycles, risk reduction), not just a list of optimizations.
- Optimization docs: each version must include before/after comparison diagrams.
- Optimization role: act as a high-quality performance optimization expert; prioritize correctness, follow stated constraints, and target cycles2 under 1200 where feasible.
- Optimization visuals: diagrams must make the specific optimization obvious at a glance and reduce reviewer decision effort.
- Optimization plan format: include 第一性原理与资源约束, 理论下界与数学推导, 核心瓶颈, 本次优化要解决的问题, and for each optimization item include 优先级 with stars, 核心思想, 步骤, 改进前后图表, plus 系统架构图, 流程图, 时序图, 数据流程图, 饼图, 代码草案, 校验, 风险与缓解.
