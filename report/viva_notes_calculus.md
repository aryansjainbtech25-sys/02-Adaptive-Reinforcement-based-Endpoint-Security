# Task 10 — Calculus Threshold Viva Notes (corrected per checklist)

(a) Hard cost: expected_cost is exact operational cost C = C_FP*FPR + C_FN*FNR + C_review*Review + 0.05*[(r-0.35)^2+(b-0.70)^2] with no eps smoothing; used for final reporting baseline 0.380772 -> optimal 0.361136.

(b) Smoothed surrogate: expected_cost_smoothed replaces indicators with sigmoid((p-theta)/tau) tau=0.05 giving differentiable FPR~,FNR~,Review~; needed because hard rates are stepwise with zero gradient almost everywhere, so gradient descent would not move without smoothing.

(c) Gradient: Numerical via central finite differences in compute_cost_gradient_fd: dC/dr≈[C(r+eps,b)-C(r-eps,b)]/(2*eps) eps=1e-4 on smoothed cost, alias compute_cost_derivative kept for backward compatibility.

(d) At the returned (0.6, 0.75) the full Hessian has det = 9.36 > 0 and both diagonal entries > 0, so the point is a local positive-definite quadratic point. However, the block-threshold gradient is ≈1.55, so this is a boundary-clipped solution — the block coordinate stopped moving because a box/gap constraint was active (on our real run boundary_hit = ['gap_0.15']; no box bound was active. If a box were active, label would be review_box_* / block_box_*), not because ∇C = 0 (is_critical_point=False, is_stationary_point=False). We report (0.6, 0.75) as the best feasible point reached by the grid + GD procedure, not as an interior stationary point. is_minimum=True is in Hessian sense only.
