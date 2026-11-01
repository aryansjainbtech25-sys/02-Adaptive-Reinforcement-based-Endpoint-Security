Final project commit: ab92b84aba3161866284a746a85746ac720ef81c (short: ab92b84). — Original honest finding 5fdc189aac39267cb96f04f4f3e409535be656a5 — len 30 gd 29 duplicate False





For the report writers — DO NOT DEVIATE FROM THESE THREE DOCUMENTS:

report/math_brief_source.md (formulas)
report/results_brief.md (numbers)
report/prohibitions.md (hard rules)

All numbers traceable to results/calculus_experiment.json and results/baseline.json at commit 5fdc189 — test evidence results/final_test_run.txt (42/42 pass, Python 3.14.2, pytest 9.2.0)

The two things most likely to lose marks:

1. Do not call the reward policy RL. No MDP, no Q-learning, no Bellman. It is a feedback-driven heuristic policy. See prohibitions §1. File name rl_policy_state.json is legacy artefact.

2. Do NOT claim gradient descent reduced the hard cost. Hard cost at grid winner [0.30,0.45] is 0.2829. Hard cost at returned point [0.60,0.75] is 0.3872. GD increased hard cost +36.87% and smoothed +51.36%. Improvement over baseline (0.3999→0.3872 -3.16%) attributable to grid search, NOT gradient descent. See results_brief §1 and math_brief §5.

Before you write each section, run:
python -c 'import json; c=json.load(open("results/calculus_experiment.json")); print(len(c["optimized"]["history"]), c["optimized"]["derivative_check"]["gradient_descent_iterations"], c["optimized"]["history"][-1]==c["optimized"]["history"][-2])'
Expected: 30 29 False. If differently, JSON regenerated and every number must be re-verified.

Three-number claim — use this exact table:

Point | C_hard | C_smooth | Note
Baseline [0.35,0.70] | 0.39987259019990573 | — | reference
Grid winner [0.30,0.45] | 0.2829383159770988 | 0.29071968134044274 | lowest hard cost found
Returned [0.60,0.75] | 0.38722259534246795 | 0.43994209280287067 | GD boundary-clipped point

Do NOT say "calculus optimization reduced hard cost" without acknowledging grid-search dominates and GD step increased hard cost.

Final commit: 960dd43275e1a50507238aa4f6e3727b7b228c46
