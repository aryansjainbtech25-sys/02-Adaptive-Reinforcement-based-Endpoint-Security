
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calculus_threshold import expected_cost, expected_cost_smoothed, _compute_two_thr_rates, _smoothed_rates, compute_cost_gradient_fd, compute_optimal_thresholds

def test_T1_hard_cost_hand_check():
    y = np.array([0,0,1,1]); p = np.array([0.1,0.5,0.6,0.9])
    cost = expected_cost(y,p,0.35,0.70, cost_fp=1.0, cost_fn=5.0, cost_review=0.5)
    assert abs(cost - 3.25) < 1e-6

def test_T2_smoothed_vs_hard_consistency():
    y = np.array([0,0,1,1]); p = np.array([0.1,0.2,0.8,0.9])
    tau = 0.001
    hard = _compute_two_thr_rates(y,p,0.35,0.70)
    fpr_s, fnr_s, rev_s = _smoothed_rates(y,p,0.35,0.70,tau=tau)
    assert abs(fpr_s - hard["fpr"]) < 0.02
    assert abs(fnr_s - hard["fnr"]) < 0.02
    assert abs(rev_s - hard["review_rate"]) < 0.02
    c_hard = expected_cost(y,p,0.35,0.70,1.0,5.0,0.5)
    c_smooth = expected_cost_smoothed(y,p,0.35,0.70,1.0,5.0,0.5,tau=tau)
    assert abs(c_hard - c_smooth) < 0.05

def test_T3_gradient_direction_sanity():
    y = np.array([0,0,1,1]); p = np.array([0.1,0.5,0.6,0.9])
    r,b = 0.35,0.70
    grad = compute_cost_gradient_fd(y,p,r,b, cost_fp=1.0, cost_fn=5.0, cost_review=0.5, tau=0.05)
    h = 1e-4
    cost = expected_cost_smoothed(y,p,r,b,1.0,5.0,0.5,0.05)
    cost_r_plus = expected_cost_smoothed(y,p,r+h,b,1.0,5.0,0.5,0.05)
    cost_b_plus = expected_cost_smoothed(y,p,r,b+h,1.0,5.0,0.5,0.05)
    if abs(grad["dC_d_review"]) > 1e-6:
        assert (cost_r_plus-cost)/h * grad["dC_d_review"] > -1e-10
    if abs(grad["dC_d_block"]) > 1e-6:
        assert (cost_b_plus-cost)/h * grad["dC_d_block"] > -1e-10

def test_T4_constraint_box_and_gap():
    y = np.array([0,0,0,1,1,1]); p = np.array([0.1,0.2,0.4,0.6,0.8,0.9])
    res = compute_optimal_thresholds(y,p)
    r = res["review_threshold"]; b = res["block_threshold"]
    assert 0.15-1e-9 <= r <= 0.60+1e-9
    assert 0.45-1e-9 <= b <= 0.95+1e-9
    assert b - r >= 0.15-1e-9

def test_T5_determinism():
    y = np.array([0,0,1,1,0,1]); p = np.array([0.1,0.5,0.6,0.9,0.2,0.85])
    res1 = compute_optimal_thresholds(y,p, grad_tol=1e-2)
    res2 = compute_optimal_thresholds(y,p, grad_tol=1e-2)
    assert res1["review_threshold"] == res2["review_threshold"]
    assert res1["block_threshold"] == res2["block_threshold"]
    assert res1["optimal_cost"] == res2["optimal_cost"]
    assert res1["derivative_check"]["is_minimum"] == res2["derivative_check"]["is_minimum"]
    assert res1["history"] == res2["history"]

def test_T6_boundary_sanity():
    # Probabilities spaced so thresholds fall inside box (no boundary hit)
    y = np.array([0,0,0,1,1,1])
    p = np.array([0.1,0.2,0.3,0.7,0.8,0.9])
    res = compute_optimal_thresholds(y,p, grad_tol=0.1, max_iter=200)
    # Should be feasible
    assert res["block_threshold"] - res["review_threshold"] >= 0.15-1e-9
    dc = res["derivative_check"]
    assert "boundary_hit" in dc
    assert "is_stationary_point" in dc
    # Stronger: must be either critical or gap-active, else test meaningless
    assert dc["is_critical_point"] == True or "gap_0.15" in dc["boundary_hit"]
    if dc["is_critical_point"]:
        assert not any("box" in b for b in dc["boundary_hit"])

def test_T7_finite_difference_gradient_check():
    y = np.array([0,0,1,1,0,1]); p = np.array([0.1,0.5,0.6,0.9,0.2,0.85])
    r0,b0 = 0.35,0.70
    h = 1e-4
    grad = compute_cost_gradient_fd(y,p,r0,b0, cost_fp=1.0, cost_fn=5.0, cost_review=0.5, tau=0.05)
    c_r_plus = expected_cost_smoothed(y,p,r0+h,b0,1.0,5.0,0.5,0.05)
    c_r_minus = expected_cost_smoothed(y,p,r0-h,b0,1.0,5.0,0.5,0.05)
    c_b_plus = expected_cost_smoothed(y,p,r0,b0+h,1.0,5.0,0.5,0.05)
    c_b_minus = expected_cost_smoothed(y,p,r0,b0-h,1.0,5.0,0.5,0.05)
    fd_r = (c_r_plus - c_r_minus)/(2*h)
    fd_b = (c_b_plus - c_b_minus)/(2*h)
    # Should match gradient with rel ~1e-2
    assert abs(fd_r - grad["dC_d_review"]) < 1e-2 or abs(fd_r - grad["dC_d_review"])/(abs(grad["dC_d_review"])+1e-9) < 1e-2
    assert abs(fd_b - grad["dC_d_block"]) < 1e-2 or abs(fd_b - grad["dC_d_block"])/(abs(grad["dC_d_block"])+1e-9) < 1e-2

def test_history_final_entry_matches_returned():
    y=np.array([0,0,1,1]); p=np.array([0.1,0.5,0.6,0.9])
    res=compute_optimal_thresholds(y,p,max_iter=3)
    # history[-1] should correspond to returned thresholds within rounding
    assert abs(res["history"][-1][0] - res["review_threshold"]) < 1e-4
    assert abs(res["history"][-1][1] - res["block_threshold"]) < 1e-4
    # Also check cost matches final_cost
    assert abs(res["history"][-1][2] - res["final_cost"]) < 1e-6
