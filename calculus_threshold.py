
"""
calculus_threshold.py — Calculus-Based Threshold Optimization for ARES
Corrections applied: B1-B7 + checklist (history[-1]==returned, boundary_hit, final_cost, etc.)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def _compute_two_thr_rates(y_true, y_proba, t_review, t_block):
    allow = y_proba < t_review
    review = (y_proba >= t_review) & (y_proba < t_block)
    block = y_proba >= t_block
    fp_review = np.sum((y_true == 0) & review)
    fp_block = np.sum((y_true == 0) & block)
    tn = np.sum((y_true == 0) & allow)
    fn_allow = np.sum((y_true == 1) & allow)
    fn_review = np.sum((y_true == 1) & review)
    tp = np.sum((y_true == 1) & block)
    total_benign = np.sum(y_true == 0)
    total_phish = np.sum(y_true == 1)
    fpr = (fp_review + fp_block) / max(total_benign, 1)
    fnr = (fn_allow + fn_review) / max(total_phish, 1)
    review_rate = np.sum(review) / max(len(y_true), 1)
    return {"fpr": fpr, "fnr": fnr, "review_rate": review_rate, "fp_review": fp_review, "fp_block": fp_block, "fn_allow": fn_allow, "fn_review": fn_review, "tn": tn, "tp": tp}

def _smoothed_rates(y_true, y_proba, t_review, t_block, tau=0.05):
    y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
    s_review = _sigmoid((y_proba - t_review) / tau)
    s_block = _sigmoid((y_proba - t_block) / tau)
    fpr_tilde = np.mean(s_review[y_true==0]) if np.any(y_true==0) else 0.0
    fnr_tilde = 1.0 - (np.mean(s_block[y_true==1]) if np.any(y_true==1) else 0.0)
    review_tilde = np.mean(s_review * (1.0 - s_block))
    return fpr_tilde, fnr_tilde, review_tilde

def expected_cost(y_true, y_proba, t_review, t_block, cost_fp=1.0, cost_fn=5.0, cost_review=0.5):
    rates = _compute_two_thr_rates(y_true, y_proba, t_review, t_block)
    reg = 0.05 * ((t_review - 0.35)**2 + (t_block - 0.70)**2)
    return float(rates["fpr"]*cost_fp + rates["fnr"]*cost_fn + rates["review_rate"]*cost_review + reg)

def expected_cost_smoothed(y_true, y_proba, t_review, t_block, cost_fp=1.0, cost_fn=5.0, cost_review=0.5, tau=0.05):
    fpr_t, fnr_t, rev_t = _smoothed_rates(y_true, y_proba, t_review, t_block, tau)
    reg = 0.05 * ((t_review - 0.35)**2 + (t_block - 0.70)**2)
    return float(fpr_t*cost_fp + fnr_t*cost_fn + rev_t*cost_review + reg)

def compute_cost_gradient_fd(y_true, y_proba, t_review, t_block, cost_fp=1.0, cost_fn=5.0, cost_review=0.5, eps=1e-4, tau=0.05):
    def c(r,b): return expected_cost_smoothed(y_true, y_proba, r,b, cost_fp, cost_fn, cost_review, tau)
    dC_d_review = (c(t_review+eps, t_block) - c(t_review-eps, t_block))/(2*eps)
    dC_d_block = (c(t_review, t_block+eps) - c(t_review, t_block-eps))/(2*eps)
    return {"dC_d_review": dC_d_review, "dC_d_block": dC_d_block}

compute_cost_derivative = compute_cost_gradient_fd

def compute_second_derivative(y_true, y_proba, t_review, t_block, cost_fp=1.0, cost_fn=5.0, cost_review=0.5, eps=1e-4, tau=0.05):
    def c(r,b): return expected_cost_smoothed(y_true, y_proba, r,b, cost_fp, cost_fn, cost_review, tau)
    c0=c(t_review,t_block)
    d2_r=(c(t_review+eps,t_block)-2*c0+c(t_review-eps,t_block))/(eps**2)
    d2_b=(c(t_review,t_block+eps)-2*c0+c(t_review,t_block-eps))/(eps**2)
    c_pp=c(t_review+eps,t_block+eps); c_pm=c(t_review+eps,t_block-eps)
    c_mp=c(t_review-eps,t_block+eps); c_mm=c(t_review-eps,t_block-eps)
    d2_rb=(c_pp-c_pm-c_mp+c_mm)/(4*eps**2)
    return {"d2C_d_review2": d2_r, "d2C_d_block2": d2_b, "d2C_d_review_dblock": d2_rb}

def compute_optimal_thresholds(y_true, y_proba, cost_fp=1.0, cost_fn=5.0, cost_review=0.5, grid_steps=100, tau=0.05, lr=0.05, max_iter=200, grad_tol=1e-2):
    y_true=np.asarray(y_true).astype(int); y_proba=np.asarray(y_proba,dtype=float)
    review_grid=np.linspace(0.15,0.60,grid_steps); block_grid=np.linspace(0.45,0.95,grid_steps)
    best_cost=float('inf'); best_r,best_b=0.35,0.70
    used_fallback=True
    for r in review_grid:
        for b in block_grid[block_grid>=r+0.15]:
            cost=expected_cost_smoothed(y_true,y_proba,r,b,cost_fp,cost_fn,cost_review,tau)
            if cost<best_cost:
                best_cost,best_r,best_b=cost,r,b
                used_fallback=False
    tr,tb=best_r,best_b; history=[]
    stopped_by="max_iter"
    boundary_hit_final=[]
    for k in range(max_iter):
        cost_k=expected_cost_smoothed(y_true,y_proba,tr,tb,cost_fp,cost_fn,cost_review,tau)
        grad=compute_cost_gradient_fd(y_true,y_proba,tr,tb,cost_fp,cost_fn,cost_review,tau=tau)
        tr_raw = tr - lr*grad["dC_d_review"]
        tb_raw = tb - lr*grad["dC_d_block"]
        tr_new=float(np.clip(tr_raw,0.15,0.60))
        tb_new=float(np.clip(tb_raw,0.45,0.95))
        # Enforce gap >=0.15
        if tb_new - tr_new < 0.15:
            tb_new=float(np.clip(tr_new+0.15,0.45,0.95))
            # Defensive guard: if tb_new hits 0.95 upper bound, lower tr_new to keep feasible.
            # Never fires with current bounds tr∈[0.15,0.60], tb∈[0.45,0.95], gap 0.15 (max gap 0.80), but kept as boilerplate.
            if tb_new > 0.95 - 1e-9 and tb_new - tr_new < 0.15:
                tr_new=float(np.clip(tb_new-0.15,0.15,0.60))
        history.append((tr,tb,cost_k))
        # Check step stop
        if abs(tr_new-tr)<1e-4 and abs(tb_new-tb)<1e-4:
            tr,tb=tr_new,tb_new
            stopped_by="step"
            break
        tr,tb=tr_new,tb_new

    # Final entry so history[-1] == returned thresholds
    # Note: final_cost and optimal_cost_smoothed are identical by construction; both provided for backward compatibility
    final_smoothed_cost = expected_cost_smoothed(y_true,y_proba,tr,tb,cost_fp,cost_fn,cost_review,tau)
    # Avoid duplicate when break fires with zero delta: last iteration already logged (tr,tb) as final point
    # In that case history[-1][:2] == (tr,tb), so we update its cost rather than append duplicate
    if history and abs(history[-1][0]-tr) < 1e-9 and abs(history[-1][1]-tb) < 1e-9:
        # Duplicate case: history[-1] is same θ as final, just update cost to smoothed final
        # This happens when stopped_by=="step" with zero delta: last loop already appended (tr,tb)
        history[-1] = (tr, tb, final_smoothed_cost)
    else:
        history.append((tr,tb,final_smoothed_cost))
    # Traceability note: If you see two identical [0.6,0.75,0.4399] entries at end, that's the duplicate case
    # described above. With this fix, duplicate is avoided; history[-1] is always the returned θ.

    deriv=compute_cost_gradient_fd(y_true,y_proba,tr,tb,cost_fp,cost_fn,cost_review,tau=tau)
    second=compute_second_derivative(y_true,y_proba,tr,tb,cost_fp,cost_fn,cost_review,tau=tau)
    det_H=second["d2C_d_review2"]*second["d2C_d_block2"]-second["d2C_d_review_dblock"]**2
    is_axis=second["d2C_d_review2"]>0 and second["d2C_d_block2"]>0
    is_min=is_axis and det_H>0
    is_critical = abs(deriv["dC_d_review"])<grad_tol and abs(deriv["dC_d_block"])<grad_tol

    # boundary_hit: final-point diagnostic (not iteration history) — which constraints are active at returned θ
    boundary_hit=[]
    if abs(tr-0.15)<1e-9 or abs(tr-0.60)<1e-9:
        boundary_hit.append(f"review_box_{tr:.2f}")
    if abs(tb-0.45)<1e-9 or abs(tb-0.95)<1e-9:
        boundary_hit.append(f"block_box_{tb:.2f}")
    if abs((tb - tr) - 0.15) < 1e-9:
        boundary_hit.append(f"gap_0.15")

    derivative_check={
        "first_derivative_review": round(deriv["dC_d_review"],6),
        "first_derivative_block": round(deriv["dC_d_block"],6),
        "is_critical_point": is_critical,
        "is_stationary_point": is_critical,
        "grad_tol": grad_tol,
        "second_derivative_review": round(second["d2C_d_review2"],6),
        "second_derivative_block": round(second["d2C_d_block2"],6),
        "cross_derivative_review_block": round(second["d2C_d_review_dblock"],6),
        "hessian_det": round(det_H,6),
        "is_minimum_along_each_axis": is_axis,
        "is_minimum": is_min,
        "boundary_hit": boundary_hit,
        "bayes_optimal_single_threshold": round(cost_fp/(cost_fp+cost_fn),4) if (cost_fp+cost_fn)>0 else 0.5,
        "calculus_principle": "Minimum where |∇C|<grad_tol and Hessian positive definite, else boundary-clipped best feasible",
        "gradient_descent_iterations": len(history)-1,
        "learning_rate": lr, "temperature_tau": tau,
        "stopped_by": stopped_by,
    }
    rates=_compute_two_thr_rates(y_true,y_proba,tr,tb)
    return {
        "review_threshold": round(float(tr),4), "block_threshold": round(float(tb),4),
        "optimal_cost": round(float(expected_cost(y_true,y_proba,tr,tb,cost_fp,cost_fn,cost_review)),6),
        "optimal_cost_smoothed": round(float(final_smoothed_cost),6),  # identical to final_cost by construction
        "final_cost": round(float(final_smoothed_cost),6),  # same as optimal_cost_smoothed; kept for backward compatibility
        "final_cost_hard": round(float(expected_cost(y_true,y_proba,tr,tb,cost_fp,cost_fn,cost_review)),6),
        "derivative_check": derivative_check,
        "rates_at_optimum": {k: round(float(v),4) if isinstance(v,(float,np.floating)) else int(v) for k,v in rates.items()},
        "cost_weights": {"fp":cost_fp,"fn":cost_fn,"review":cost_review}, "tau":tau, "grad_tol":grad_tol, "grid_fallback_used": used_fallback, "history":history,
    }

def validate_thresholds_calculus(y_true,y_proba,t_review,t_block,cost_fp=1.0,cost_fn=5.0,cost_review=0.5,tau=0.05):
    cost=expected_cost(y_true,y_proba,t_review,t_block,cost_fp,cost_fn,cost_review)
    cost_s=expected_cost_smoothed(y_true,y_proba,t_review,t_block,cost_fp,cost_fn,cost_review,tau)
    deriv=compute_cost_gradient_fd(y_true,y_proba,t_review,t_block,cost_fp,cost_fn,cost_review,tau=tau)
    second=compute_second_derivative(y_true,y_proba,t_review,t_block,cost_fp,cost_fn,cost_review,tau=tau)
    det=second["d2C_d_review2"]*second["d2C_d_block2"]-second["d2C_d_review_dblock"]**2
    return {"cost":cost,"cost_smoothed":cost_s,"first_derivative":deriv,"second_derivative":second,"hessian_det":det,"is_minimum":second["d2C_d_review2"]>0 and second["d2C_d_block2"]>0 and det>0,"tau":tau}
