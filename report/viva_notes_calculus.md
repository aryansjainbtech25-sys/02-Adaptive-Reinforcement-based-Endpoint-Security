# Task 10 — Calculus Threshold Viva Notes (corrected)

(a) Hard cost: expected_cost is exact operational cost C = C_FP*FPR + C_FN*FNR + C_review*Review + 0.05*[(r-0.35)^2+(b-0.70)^2] with no eps smoothing; used for final reporting baseline 0.380772 -> optimal 0.361136.

(b) Smoothed surrogate: expected_cost_smoothed replaces indicators with sigmoid((p-theta)/tau) tau=0.05 giving differentiable FPR~,FNR~,Review~; needed because hard rates are stepwise with zero gradient almost everywhere.

(c) Gradient: Numerical via central finite differences in compute_cost_gradient_fd: dC/dr≈[C(r+eps,b)-C(r-eps,b)]/(2*eps) eps=1e-4 on smoothed cost, alias compute_cost_derivative kept.

(d) is_minimum: Checks full Hessian positive definite: d2r>0,d2b>0,det=d2r*d2b-cross^2>0 with cross=[C(r+e,b+e)-C(r+e,b-e)-C(r-e,b+e)+C(r-e,b-e)]/(4e^2). On real held-out at returned (0.6,0.75): d2r=0.484839,d2b=19.328268,cross=-0.070477,det=9.36614>0 so Hessian PD, but first_derivative_block=1.550729 so is_critical_point=False (|grad|>grad_tol). This is boundary-clipped solution: block threshold stopped by clip/gap constraint (|Δθ|<1e-4) not ∇C≈0. We report it as best feasible point found by grid+GD, not KKT optimal. Positive-definite Hessian at boundary still indicates local bowl but not interior minimum.
