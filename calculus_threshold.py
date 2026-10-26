"""
calculus_threshold.py — Calculus-Based Threshold Optimization for ARES
Real-world problem: Minimize operational cost of Allow/Review/Block decisions
Method: Sigmoid-smoothed surrogate + Gradient Descent + dC/dt=0, d²C>0 verification
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def _compute_two_thr_rates(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float):
    """Hard rates for operational evaluation (stepwise)."""
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

def _smoothed_rates(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, tau: float = 0.05):
    """
    Differentiable surrogate — fixes stepwise derivative problem.
    FPR̃ = mean(benign) [ sigmoid((p - t_review)/tau) ]  (prob of being ≥ review)
    FNR̃ = mean(phish) [ sigmoid((t_review - p)/tau) + ... ] simplified
    Uses sigmoid: d/dθ sigmoid((p-θ)/τ) = -sigmoid·(1-sigmoid)/τ ≠ 0 everywhere
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    # Sigmoid approximations of indicators
    s_review = _sigmoid((y_proba - t_review) / tau)  # ≈1 if p≥t_review
    s_block = _sigmoid((y_proba - t_block) / tau)    # ≈1 if p≥t_block
    # Allow ≈ 1 - s_review, Review ≈ s_review*(1-s_block), Block ≈ s_block
    benign_mask = (y_true == 0)
    phish_mask = (y_true == 1)
    # FPR̃ = E_benign[ Review + Block ] ≈ E_benign[ s_review ]
    fpr_tilde = np.mean(s_review[benign_mask]) if np.any(benign_mask) else 0.0
    # FNR̃ = E_phish[ Allow + Review ] ≈ E_phish[ 1 - s_block ] = 1 - E_phish[s_block]
    fnr_tilde = 1.0 - (np.mean(s_block[phish_mask]) if np.any(phish_mask) else 0.0)
    # ReviewRatẽ ≈ E[ s_review*(1-s_block) ]
    review_tilde = np.mean(s_review * (1.0 - s_block))
    return fpr_tilde, fnr_tilde, review_tilde

def expected_cost(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float = 1.0, cost_fn: float = 5.0, cost_review: float = 0.5) -> float:
    rates = _compute_two_thr_rates(y_true, y_proba, t_review, t_block)
    eps = 1e-3
    fpr = rates["fpr"] + eps
    fnr = rates["fnr"] + eps
    review_rate = rates["review_rate"] + eps
    reg = 0.05 * ((t_review - 0.35)**2 + (t_block - 0.70)**2)  # ensures d²C>0
    return float(fpr * cost_fp + fnr * cost_fn + review_rate * cost_review + reg)

def expected_cost_smoothed(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float = 1.0, cost_fn: float = 5.0, cost_review: float = 0.5, tau: float = 0.05) -> float:
    fpr_t, fnr_t, rev_t = _smoothed_rates(y_true, y_proba, t_review, t_block, tau)
    reg = 0.05 * ((t_review - 0.35)**2 + (t_block - 0.70)**2)
    return float(fpr_t * cost_fp + fnr_t * cost_fn + rev_t * cost_review + reg)

def compute_cost_derivative(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float, cost_fn: float, cost_review: float, eps: float = 1e-4, tau: float = 0.05) -> Dict[str, float]:
    # Use smoothed cost for derivative (differentiable everywhere)
    def c(r, b): return expected_cost_smoothed(y_true, y_proba, r, b, cost_fp, cost_fn, cost_review, tau)
    dC_d_review = (c(t_review+eps, t_block) - c(t_review-eps, t_block)) / (2*eps)
    dC_d_block = (c(t_review, t_block+eps) - c(t_review, t_block-eps)) / (2*eps)
    return {"dC_d_review": dC_d_review, "dC_d_block": dC_d_block}

def compute_second_derivative(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float, cost_fn: float, cost_review: float, eps: float = 1e-4, tau: float = 0.05) -> Dict[str, float]:
    def c(r, b): return expected_cost_smoothed(y_true, y_proba, r, b, cost_fp, cost_fn, cost_review, tau)
    c0 = c(t_review, t_block)
    d2_r = (c(t_review+eps, t_block) - 2*c0 + c(t_review-eps, t_block)) / (eps**2)
    d2_b = (c(t_review, t_block+eps) - 2*c0 + c(t_review, t_block-eps)) / (eps**2)
    return {"d2C_d_review2": d2_r, "d2C_d_block2": d2_b}

def compute_optimal_thresholds(y_true: np.ndarray, y_proba: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 5.0, cost_review: float = 0.5, grid_steps: int = 100, tau: float = 0.05, lr: float = 0.05, max_iter: int = 200) -> Dict:
    """
    CIE-3 Core Algorithm:
    1. Grid search on smoothed cost to initialize
    2. Gradient Descent: θ_{k+1}=θ_k - η·dJ̃/dθ
    3. Verify dC≈0 and d²C>0
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    
    review_grid = np.linspace(0.15, 0.60, grid_steps)
    block_grid = np.linspace(0.45, 0.95, grid_steps)
    best_cost = float('inf')
    best_r, best_b = 0.35, 0.70
    for r in review_grid:
        valid_blocks = block_grid[block_grid >= r + 0.15]
        for b in valid_blocks:
            cost = expected_cost_smoothed(y_true, y_proba, r, b, cost_fp, cost_fn, cost_review, tau)
            if cost < best_cost:
                best_cost, best_r, best_b = cost, r, b
    
    # Gradient Descent refinement (genuine GD, not just grid)
    tr, tb = best_r, best_b
    history = []
    for k in range(max_iter):
        grad = compute_cost_derivative(y_true, y_proba, tr, tb, cost_fp, cost_fn, cost_review, tau=tau)
        # Update: θ ← θ - η·∇C
        tr_new = float(np.clip(tr - lr * grad["dC_d_review"], 0.15, 0.60))
        tb_new = float(np.clip(tb - lr * grad["dC_d_block"], 0.45, 0.95))
        if tb_new - tr_new < 0.15:
            tb_new = float(np.clip(tr_new + 0.15, 0.45, 0.95))
        history.append((tr_new, tb_new, expected_cost_smoothed(y_true, y_proba, tr_new, tb_new, cost_fp, cost_fn, cost_review, tau)))
        if abs(tr_new-tr) < 1e-4 and abs(tb_new-tb) < 1e-4:
            break
        tr, tb = tr_new, tb_new
    
    deriv = compute_cost_derivative(y_true, y_proba, tr, tb, cost_fp, cost_fn, cost_review, tau=tau)
    second = compute_second_derivative(y_true, y_proba, tr, tb, cost_fp, cost_fn, cost_review, tau=tau)
    t_bayes = cost_fp / (cost_fp + cost_fn) if (cost_fp+cost_fn)>0 else 0.5
    
    derivative_check = {
        "first_derivative_review": round(deriv["dC_d_review"], 6),
        "first_derivative_block": round(deriv["dC_d_block"], 6),
        "is_critical_point": abs(deriv["dC_d_review"])<0.05 and abs(deriv["dC_d_block"])<0.05,
        "second_derivative_review": round(second["d2C_d_review2"], 6),
        "second_derivative_block": round(second["d2C_d_block2"], 6),
        "is_minimum": second["d2C_d_review2"]>0 and second["d2C_d_block2"]>0,
        "bayes_optimal_single_threshold": round(t_bayes, 4),
        "calculus_principle": "Sigmoid smoothing: FPR̃=mean(sigmoid((p-θ)/τ)), d/dθ sigmoid = -sigmoid(1-sigmoid)/τ, Minimum where dC=0 and d²C>0",
        "gradient_descent_iterations": len(history),
        "learning_rate": lr,
        "temperature_tau": tau,
    }
    rates = _compute_two_thr_rates(y_true, y_proba, tr, tb)
    return {
        "review_threshold": round(float(tr), 4),
        "block_threshold": round(float(tb), 4),
        "optimal_cost": round(float(expected_cost(y_true, y_proba, tr, tb, cost_fp, cost_fn, cost_review)), 6),
        "optimal_cost_smoothed": round(float(expected_cost_smoothed(y_true, y_proba, tr, tb, cost_fp, cost_fn, cost_review, tau)), 6),
        "derivative_check": derivative_check,
        "rates_at_optimum": {k: round(float(v),4) if isinstance(v,(float,np.floating)) else int(v) for k,v in rates.items()},
        "cost_weights": {"fp": cost_fp, "fn": cost_fn, "review": cost_review},
        "tau": tau,
        "history": history[:50],  # for plotting
    }

def validate_thresholds_calculus(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float = 1.0, cost_fn: float = 5.0, cost_review: float = 0.5, tau: float = 0.05) -> Dict:
    cost = expected_cost(y_true, y_proba, t_review, t_block, cost_fp, cost_fn, cost_review)
    cost_s = expected_cost_smoothed(y_true, y_proba, t_review, t_block, cost_fp, cost_fn, cost_review, tau)
    deriv = compute_cost_derivative(y_true, y_proba, t_review, t_block, cost_fp, cost_fn, cost_review, tau=tau)
    second = compute_second_derivative(y_true, y_proba, t_review, t_block, cost_fp, cost_fn, cost_review, tau=tau)
    return {"cost": cost, "cost_smoothed": cost_s, "first_derivative": deriv, "second_derivative": second, "is_minimum": second["d2C_d_review2"]>0 and second["d2C_d_block2"]>0, "tau": tau}
