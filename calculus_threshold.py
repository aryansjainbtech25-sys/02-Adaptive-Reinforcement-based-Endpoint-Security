"""
calculus_threshold.py — Calculus-Based Threshold Optimization for ARES

This module provides the mathematical justification for Allow/Review/Block thresholds.
Uses differential calculus to minimize expected security cost.

Core Idea:
Define cost function C(t) = FP(t)*C_fp + FN(t)*C_fn + Review(t)*C_review
where FP(t), FN(t), Review(t) are functions of threshold t.

Then:
1. Compute dC/dt (first derivative) — rate of change of cost
2. Find t* where dC/dt = 0 (critical point)
3. Verify d^2C/dt^2 > 0 (second derivative test for minimum)

For two thresholds (review_threshold < block_threshold), we optimize jointly.

This replaces heuristic 0.35/0.70 with mathematically optimal values.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

def _compute_rates(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> Tuple[float, float, float]:
    """Compute FPR, FNR, ReviewRate at threshold thr (single threshold)."""
    y_pred = (y_proba >= thr).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return fpr, fnr, 0.0

def _compute_two_thr_rates(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float):
    """For two thresholds: Allow if p < t_review, Review if t_review <= p < t_block, Block if p >= t_block"""
    allow = y_proba < t_review
    review = (y_proba >= t_review) & (y_proba < t_block)
    block = y_proba >= t_block
    
    # False positives: benign classified as Review or Block
    fp_review = np.sum((y_true == 0) & review)
    fp_block = np.sum((y_true == 0) & block)
    tn = np.sum((y_true == 0) & allow)
    
    # False negatives: phishing classified as Allow or Review (if Review is not blocking)
    fn_allow = np.sum((y_true == 1) & allow)
    fn_review = np.sum((y_true == 1) & review)
    tp = np.sum((y_true == 1) & block)
    
    total_benign = np.sum(y_true == 0)
    total_phish = np.sum(y_true == 1)
    
    fpr = (fp_review + fp_block) / max(total_benign, 1)
    fnr = (fn_allow + fn_review) / max(total_phish, 1)
    review_rate = np.sum(review) / max(len(y_true), 1)
    
    return {
        "fpr": fpr,
        "fnr": fnr,
        "review_rate": review_rate,
        "fp_review": fp_review,
        "fp_block": fp_block,
        "fn_allow": fn_allow,
        "fn_review": fn_review,
        "tn": tn,
        "tp": tp,
    }

def expected_cost(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float = 1.0, cost_fn: float = 5.0, cost_review: float = 0.2) -> float:
    """
    Expected cost function C(t_review, t_block)
    C = FPR * C_fp + FNR * C_fn + ReviewRate * C_review
    
    Where:
    - C_fp: cost of false positive (blocking benign)
    - C_fn: cost of false negative (allowing phishing) — typically 5x higher
    - C_review: cost of manual review
    """
    rates = _compute_two_thr_rates(y_true, y_proba, t_review, t_block)
    cost = rates["fpr"] * cost_fp + rates["fnr"] * cost_fn + rates["review_rate"] * cost_review
    return float(cost)

def compute_cost_derivative(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float, cost_fn: float, cost_review: float, eps: float = 1e-4) -> Dict[str, float]:
    """
    Numerical first derivative dC/dt using central difference:
    dC/dt ≈ [C(t+eps) - C(t-eps)] / (2*eps)
    
    Calculus concept: derivative measures instantaneous rate of cost change.
    """
    # dC/d_t_review
    c_plus = expected_cost(y_true, y_proba, t_review + eps, t_block, cost_fp, cost_fn, cost_review)
    c_minus = expected_cost(y_true, y_proba, t_review - eps, t_block, cost_fp, cost_fn, cost_review)
    dC_d_review = (c_plus - c_minus) / (2 * eps)
    
    # dC/d_t_block
    c_plus_b = expected_cost(y_true, y_proba, t_review, t_block + eps, cost_fp, cost_fn, cost_review)
    c_minus_b = expected_cost(y_true, y_proba, t_review, t_block - eps, cost_fp, cost_fn, cost_review)
    dC_d_block = (c_plus_b - c_minus_b) / (2 * eps)
    
    return {"dC_d_review": dC_d_review, "dC_d_block": dC_d_block}

def compute_second_derivative(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float, cost_fn: float, cost_review: float, eps: float = 1e-4) -> Dict[str, float]:
    """
    Second derivative d^2C/dt^2 for verification of minimum:
    d^2C/dt^2 > 0 => local minimum (convex)
    d^2C/dt^2 < 0 => local maximum
    """
    c = expected_cost(y_true, y_proba, t_review, t_block, cost_fp, cost_fn, cost_review)
    c_plus_r = expected_cost(y_true, y_proba, t_review + eps, t_block, cost_fp, cost_fn, cost_review)
    c_minus_r = expected_cost(y_true, y_proba, t_review - eps, t_block, cost_fp, cost_fn, cost_review)
    d2_review = (c_plus_r - 2*c + c_minus_r) / (eps**2)
    
    c_plus_b = expected_cost(y_true, y_proba, t_review, t_block + eps, cost_fp, cost_fn, cost_review)
    c_minus_b = expected_cost(y_true, y_proba, t_review, t_block - eps, cost_fp, cost_fn, cost_review)
    d2_block = (c_plus_b - 2*c + c_minus_b) / (eps**2)
    
    return {"d2C_d_review2": d2_review, "d2C_d_block2": d2_block}

def compute_optimal_thresholds(y_true: np.ndarray, y_proba: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 5.0, cost_review: float = 0.2, grid_steps: int = 100) -> Dict:
    """
    Find optimal thresholds via calculus-informed grid search + derivative verification.
    
    Step 1: Grid search to find approximate minimum of C(t_review, t_block)
    Step 2: At optimum, verify first derivative ≈ 0 and second derivative > 0
    
    Mathematical justification:
    - Critical point: dC/dt = 0
    - Minimum: d^2C/dt^2 > 0
    - Bayes optimal single threshold: t* = C_fp / (C_fp + C_fn)
    
    Returns thresholds that minimize expected security cost.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    
    # Define search space respecting 0.15 <= review < block <= 0.95 and gap >=0.15
    review_grid = np.linspace(0.15, 0.60, grid_steps)
    block_grid = np.linspace(0.45, 0.95, grid_steps)
    
    best_cost = float('inf')
    best_review = 0.35
    best_block = 0.70
    
    # Calculus: Evaluate cost surface C(t_review, t_block)
    for r in review_grid:
        # block must be at least r + 0.15
        valid_blocks = block_grid[block_grid >= r + 0.15]
        for b in valid_blocks:
            cost = expected_cost(y_true, y_proba, r, b, cost_fp, cost_fn, cost_review)
            if cost < best_cost:
                best_cost = cost
                best_review = r
                best_block = b
    
    # At optimum, compute calculus verification
    deriv = compute_cost_derivative(y_true, y_proba, best_review, best_block, cost_fp, cost_fn, cost_review)
    second = compute_second_derivative(y_true, y_proba, best_review, best_block, cost_fp, cost_fn, cost_review)
    
    # Analytical Bayes threshold for reference: t_bayes = C_fp / (C_fp + C_fn)
    t_bayes = cost_fp / (cost_fp + cost_fn) if (cost_fp + cost_fn) > 0 else 0.5
    
    derivative_check = {
        "first_derivative_review": round(deriv["dC_d_review"], 6),
        "first_derivative_block": round(deriv["dC_d_block"], 6),
        "is_critical_point": abs(deriv["dC_d_review"]) < 0.1 and abs(deriv["dC_d_block"]) < 0.1,
        "second_derivative_review": round(second["d2C_d_review2"], 6),
        "second_derivative_block": round(second["d2C_d_block2"], 6),
        "is_minimum": second["d2C_d_review2"] > 0 and second["d2C_d_block2"] > 0,
        "bayes_optimal_single_threshold": round(t_bayes, 4),
        "calculus_principle": "Minimum where dC/dt=0 and d2C/dt2>0",
    }
    
    rates = _compute_two_thr_rates(y_true, y_proba, best_review, best_block)
    
    return {
        "review_threshold": round(float(best_review), 4),
        "block_threshold": round(float(best_block), 4),
        "optimal_cost": round(float(best_cost), 6),
        "derivative_check": derivative_check,
        "rates_at_optimum": {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v) for k, v in rates.items()},
        "cost_weights": {"fp": cost_fp, "fn": cost_fn, "review": cost_review},
    }

def validate_thresholds_calculus(y_true: np.ndarray, y_proba: np.ndarray, t_review: float, t_block: float, cost_fp: float = 1.0, cost_fn: float = 5.0, cost_review: float = 0.2) -> Dict:
    """Validate that given thresholds satisfy calculus optimality conditions."""
    cost = expected_cost(y_true, y_proba, t_review, t_block, cost_fp, cost_fn, cost_review)
    deriv = compute_cost_derivative(y_true, y_proba, t_review, t_block, cost_fp, cost_fn, cost_review)
    second = compute_second_derivative(y_true, y_proba, t_review, t_block, cost_fp, cost_fn, cost_review)
    
    return {
        "cost": cost,
        "first_derivative": deriv,
        "second_derivative": second,
        "is_minimum": second["d2C_d_review2"] > 0 and second["d2C_d_block2"] > 0,
    }
