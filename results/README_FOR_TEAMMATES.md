Final project commit: 5fdc189aac39267cb96f04f4f3e409535be656a5 (short: 5fdc189).

For the report writers — DO NOT DEVIATE FROM THESE THREE DOCUMENTS:

report/math_brief_source.md (formulas)
report/results_brief.md (numbers)
report/prohibitions.md (hard rules)

All numbers in this report are traceable to results/calculus_experiment.json and results/baseline.json at commit 5fdc189aac39267cb96f04f4f3e409535be656a5. Test evidence: results/final_test_run.txt (42/42 pass, Python 3.14.2, pytest 9.2.0).

Previous chain: 7506878 -> 5fdc189 -> 960dd43 -> 28ba5bf -> 8d12322 — final 5fdc189aac39267cb96f04f4f3e409535be656a5

The two things most likely to lose marks:

1. Do not call the reward policy RL. No MDP, no Q-learning, no Bellman. It is a feedback-driven heuristic policy. See prohibitions §1. File name rl_policy_state.json is legacy artefact.

2. Do NOT claim gradient descent reduced the hard cost. Hard cost at grid winner [0.30,0.45] is 0.2829. Hard cost at returned point [0.60,0.75] is 0.3872. GD increased hard cost from 0.2829 to 0.3872 — relative increase 36.87% — and smoothed cost from 0.2907 to 0.4399 — relative increase 51.36%. Neither metric improved during GD. Improvement over baseline (0.3999→0.3872 -3.16%) attributable to grid search, NOT gradient descent. See results_brief §1 and math_brief §5.

Before you write each section, run:
python -c 'import json; c=json.load(open("results/calculus_experiment.json")); print(len(c["optimized"]["history"]), c["optimized"]["derivative_check"]["gradient_descent_iterations"], c["optimized"]["history"][-1]==c["optimized"]["history"][-2])'
Expected: 30 29 False. If differently, JSON regenerated and every number must be re-verified.

Three-number claim — use this exact table:

| Point | C_hard | C_smooth | Note |
|-------|--------|----------|------|
| Baseline [0.35,0.70] | 0.39987259019990573 | — | reference |
| Grid winner [0.30,0.45] | 0.2829383159770988 | 0.29071968134044274 | lowest hard cost found |
| Returned [0.60,0.75] | 0.38722259534246795 | 0.43994209280287067 | GD boundary-clipped point |

Do NOT say "calculus optimization reduced hard cost" without acknowledging grid-search dominates and GD step increased hard cost.

Chapter 4 honest claim — must appear exactly:

"The combined grid + gradient descent procedure returns thresholds (0.60, 0.75) with hard cost 0.3872, compared to the baseline (0.35, 0.70) with hard cost 0.3999 — an improvement of 0.0127 absolute / 3.16% relative. However, this improvement is attributable to the grid search, not to the gradient descent step. The grid-winner (0.30, 0.45) has hard cost 0.2829 — the lowest hard cost found anywhere by the procedure. During the 29 gradient descent iterations, both the smoothed cost (0.2907 → 0.4399, +51.36%) and the hard cost (0.2829 → 0.3872, +36.87%) increased, because projection onto the gap and box constraints is not a descent direction for the smoothed objective. The gradient descent step therefore returns a boundary-clipped feasible point satisfying second-order conditions, not a hard-cost minimiser."

Second-order claim — must appear exactly:

"The Hessian of the smoothed surrogate at (0.60, 0.75) has determinant 6.923728 > 0 and trace 17.49 > 0, so it is positive-definite there. This is a necessary condition for a local minimum of the smoothed surrogate on the feasible set; it is not a proof of constrained optimality, because we have not verified the KKT multipliers."

KKT claim must be absent — use:

"The returned point is a boundary-clipped feasible point. We do not claim KKT optimality because the multipliers were not verified."

Reward policy — must appear exactly:

"ARES uses a feedback-driven heuristic policy to adapt thresholds. This is not formal reinforcement learning; there is no MDP, no value function, and no Q-learning. The file name rl_policy_state.json is a legacy artefact from an earlier design."

Final commit: 5fdc189aac39267cb96f04f4f3e409535be656a5
