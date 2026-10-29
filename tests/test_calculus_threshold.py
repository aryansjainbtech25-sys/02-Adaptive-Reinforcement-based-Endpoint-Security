"""
tests/test_calculus_threshold.py — Task 9
T1 Hard cost hand-check
T2 Smoothed vs hard consistency
T3 Gradient direction sanity
T4 Constraint box + gap
T5 Determinism
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calculus_threshold import (
    expected_cost,
    expected_cost_smoothed,
    _compute_two_thr_rates,
    _smoothed_rates,
    compute_cost_gradient_fd,
    compute_optimal_thresholds
)

def test_T1_hard_cost_hand_check():
    """T1: y=[0,0,1,1] p=[0.1,0.5,0.6,0.9] r=0.35 b=0.70 -> fpr=0.5 fnr=0.5 review=0.5 cost=3.25"""
    y = np.array([0,0,1,1])
    p = np.array([0.1,0.5,0.6,0.9])
    cost = expected_cost(y,p,0.35,0.70, cost_fp=1.0, cost_fn=5.0, cost_review=0.5)
    assert abs(cost - 3.25) < 1e-6, f"Expected 3.25 got {cost}"

def test_T2_smoothed_vs_hard_consistency():
    """T2: with tau=0.001 and probas far from thresholds, smoothed ~ hard"""
    y = np.array([0,0,1,1])
    p = np.array([0.1,0.2,0.8,0.9])  # far from 0.35/0.70
    tau = 0.001
    hard = _compute_two_thr_rates(y,p,0.35,0.70)
    fpr_s, fnr_s, rev_s = _smoothed_rates(y,p,0.35,0.70,tau=tau)
    assert abs(fpr_s - hard["fpr"]) < 0.02, f"fpr smoothed {fpr_s} vs hard {hard['fpr']}"
    assert abs(fnr_s - hard["fnr"]) < 0.02
    assert abs(rev_s - hard["review_rate"]) < 0.02

def test_T3_gradient_direction_sanity():
    """T3: if dC/dr >0, increasing r should increase smoothed cost"""
    y = np.array([0,0,1,1])
    p = np.array([0.1,0.5,0.6,0.9])
    r,b = 0.35,0.70
    grad = compute_cost_gradient_fd(y,p,r,b, cost_fp=1.0, cost_fn=5.0, cost_review=0.5, tau=0.05)
    h = 1e-4
    cost = expected_cost_smoothed(y,p,r,b)
    cost_r_plus = expected_cost_smoothed(y,p,r+h,b)
    if grad["dC_d_review"] > 0:
        assert cost_r_plus > cost
    else:
        assert cost_r_plus < cost

    cost_b_plus = expected_cost_smoothed(y,p,r,b+h)
    if grad["dC_d_block"] > 0:
        assert cost_b_plus > cost
    else:
        assert cost_b_plus < cost

def test_T4_constraint_box_and_gap():
    """T4: returned thresholds inside box and gap >=0.15 -1e-9"""
    y = np.array([0,0,0,1,1,1])
    p = np.array([0.1,0.2,0.4,0.6,0.8,0.9])
    res = compute_optimal_thresholds(y,p)
    r = res["review_threshold"]
    b = res["block_threshold"]
    assert 0.15 -1e-9 <= r <= 0.60 +1e-9
    assert 0.45 -1e-9 <= b <= 0.95 +1e-9
    assert b - r >= 0.15 -1e-9, f"gap {b-r} <0.15"

def test_T5_determinism():
    """T5: twice same input gives identical thresholds"""
    y = np.array([0,0,1,1,0,1])
    p = np.array([0.1,0.5,0.6,0.9,0.2,0.85])
    res1 = compute_optimal_thresholds(y,p, grad_tol=1e-2)
    res2 = compute_optimal_thresholds(y,p, grad_tol=1e-2)
    assert res1["review_threshold"] == res2["review_threshold"]
    assert res1["block_threshold"] == res2["block_threshold"]
    assert res1["optimal_cost"] == res2["optimal_cost"]
