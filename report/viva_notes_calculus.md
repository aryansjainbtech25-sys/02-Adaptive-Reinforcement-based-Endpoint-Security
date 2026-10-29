# Task 10 — Calculus Threshold Viva Notes (4 sentences)

(a) Hard cost: expected_cost is exact operational cost C = C_FP*FPR + C_FN*FNR + C_review*Review + 0.05*[(r-0.35)^2+(b-0.70)^2] computed from hard Allow/Review/Block counts with no eps smoothing; it is used for final reporting (baseline cost 0.3807 -> optimal 0.3611) and presentation metrics, not for gradient.

(b) Smoothed surrogate: expected_cost_smoothed replaces step indicators with sigmoid((p-theta)/tau) with tau=0.05, giving differentiable FPR~, FNR~, Review~; it is needed because hard rates are piecewise constant with zero gradient almost everywhere, so gradient descent would not move without smoothing.

(c) Gradient: Not analytic — it is numerical gradient via central finite differences in compute_cost_gradient_fd, i.e., dC/dr ≈ [C(r+eps,b)-C(r-eps,b)]/(2*eps) with eps=1e-4 on the smoothed cost, and backward-compatible alias compute_cost_derivative is kept for old imports.

(d) is_minimum: Checks full Hessian positive definite, not just diagonal: computes d2C/dr2, d2C/db2 via second central differences and mixed partial d2C/drdb = [C(r+eps,b+eps)-C(r+eps,b-eps)-C(r-eps,b+eps)+C(r-eps,b-eps)]/(4*eps^2), then det = d2r*d2b - cross^2; is_minimum=True only if d2r>0, d2b>0 and det>0 (true minimum), while is_minimum_along_each_axis checks diagonal only; on real held-out data det=9.36614>0, so true minimum at 0.6/0.75.
