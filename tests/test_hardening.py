import ast
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pandas as pd

from adaptive_agent import (
    AdaptiveSecurityAgent,
    _validate_loaded_bundle,
    analyze_url_risk_factors,
    load_model_bundle,
)
from feature_extractor import extract_url_features, feature_names, is_valid_url
from qr_utils import QRDecodeError, decode_qr_code, decode_qr_codes
from train_model import (
    DATA_PATH,
    build_dataset,
    validate_raw_dataset,
    verify_no_group_leakage,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"


def test_every_python_file_compiles_and_parses_cleanly():
    """Priority 1 & 9: Compile every .py file in the project via py_compile and ast.parse."""
    py_files = sorted(PROJECT_DIR.rglob("*.py"))
    assert len(py_files) >= 8, f"Expected at least 8 Python files, found {len(py_files)}"
    for py_file in py_files:
        py_compile.compile(str(py_file), doraise=True)
        source_text = py_file.read_text(encoding="utf-8")
        ast.parse(source_text, filename=str(py_file))


def test_streamlit_app_startup_smoke():
    """
    Priority 1 & 9: Execute app.py top-to-bottom under a headless Streamlit mock (or AppTest)
    so any runtime NameError, KeyError, or bad function call inside app.py is caught immediately.
    """
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.sidebar.__enter__ = lambda self: mock_st.sidebar
    mock_st.sidebar.__exit__ = lambda self, *args: None

    def _make_cols(spec):
        n = spec if isinstance(spec, int) else len(spec)
        cols = []
        for _ in range(n):
            c = MagicMock()
            c.__enter__ = lambda s: s
            c.__exit__ = lambda s, *a: None
            c.button.return_value = False
            cols.append(c)
        return cols

    mock_st.columns.side_effect = _make_cols
    mock_st.tabs.side_effect = lambda names: _make_cols(len(names))
    mock_st.expander.return_value.__enter__ = lambda s: s
    mock_st.expander.return_value.__exit__ = lambda s, *a: None
    mock_st.checkbox.return_value = True
    mock_st.button.return_value = False
    mock_st.sidebar.button.return_value = False
    mock_st.file_uploader.return_value = None
    mock_st.camera_input.return_value = None
    mock_st.selectbox.return_value = "Low-Risk HTTPS — Google (test_qr.png)"
    mock_st.text_input.return_value = "https://example.com"
    mock_st.cache_resource = lambda **kwargs: (lambda fn: fn)

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        app_code = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        namespace = {"__file__": str(PROJECT_DIR / "app.py"), "__name__": "__main__"}
        exec(compile(app_code, str(PROJECT_DIR / "app.py"), "exec"), namespace)
        # Also test run_url_analysis directly from the executed app module
        bundle = namespace["get_cached_bundle"](MODEL_PATH)
        agent = namespace["agent"]
        res = namespace["run_url_analysis"](bundle, "https://example.com", agent)
        assert res["predicted_label"] == "Benign"
        assert "calibrated_probabilities" in res
        assert "uncalibrated_rf_probabilities" in res


def test_valid_forms_accepted():
    valid_urls = [
        "http://[::1]:8080/a",
        "https://example.com:8443/p",
        "https://example.com./a",
        "http://xn--pple-43d.com/",
        "http://\u0430pple.com/",
        "example.com?r=https://evil.com",
    ]
    for u in valid_urls:
        assert is_valid_url(u), f"Expected valid URL: {u}"


def test_invalid_forms_rejected():
    invalid_urls = [
        "mailto:a@b.com",
        "WIFI:S:x;T:WPA;P:y;;",
        "javascript:alert(1)",
        "ftp://x.com",
        "https://example.com:99999",
        "a b.com",
        "http://example..com/a",
    ]
    for u in invalid_urls:
        assert not is_valid_url(u), f"Expected invalid URL: {u}"


def test_backslash_parser_differential_is_closed():
    assert extract_url_features("http://192.168.1.1\\@google.com")["has_ip"] == 1


def test_query_slashes_not_counted_as_subdirs():
    f = extract_url_features("https://example.com/a?next=https://evil.com/x/y#/f/z")
    assert f["num_subdirs"] == 1 and f["embedded_url_in_query"] == 1


def test_benign_superwords_masked_and_compounds_kept():
    assert extract_url_features("https://accountant.com")["suspicious_words"] == 0
    assert extract_url_features("https://securebankinggroup.com")["susp_compound_hits"] == 2


def test_host_dots_vs_path_dots_explanation_accuracy():
    """Priority 6: Path dots must not trigger 'Deep hostname subdomain hierarchy'."""
    path_dots_url = "https://example.com/archive.v1.v2.v3.tar.gz"
    feats = extract_url_features(path_dots_url)
    assert feats["subdomain_count"] == 0
    assert feats["num_dots"] == 6
    info = analyze_url_risk_factors(path_dots_url, feats, 0.10, 0.10)
    assert not any("subdomain hierarchy" in rf.lower() for rf in info["risk_factors"])
    assert any("across the full url" in rf.lower() for rf in info["risk_factors"])


def _qr(t: str) -> np.ndarray:
    q = cv2.QRCodeEncoder.create().encode(t)
    q = cv2.copyMakeBorder(q, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(q, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)


def test_inverted_and_multi_qr():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        a, b = _qr("https://a.example.com/x"), _qr("https://b.example.net/y")
        cv2.imwrite(str(tmp_path / "inv.png"), 255 - a)
        assert decode_qr_code(tmp_path / "inv.png") == "https://a.example.com/x"

        cv2.imwrite(str(tmp_path / "two.png"), np.hstack([a, b]))
        assert len(decode_qr_codes(tmp_path / "two.png")) == 2
        try:
            decode_qr_code(tmp_path / "two.png")
            assert False, "Expected QRDecodeError on multi-QR image"
        except QRDecodeError:
            pass


def test_agent_feedback_policy_contradictions_and_recovery():
    """Priority 5: Test probability-aware updates, contradictions, duplicate redaction, and malformed state recovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fb_path = tmp_path / "f.csv"
        st_path = tmp_path / "s.json"
        ag = AdaptiveSecurityAgent(fb_path, st_path)

        # 1. False negative lowers thresholds proportionally to probability margin
        ag.record_feedback(
            "https://x.com/login?token=abc111", 0.50, "Medium", "Review", "false_negative"
        )
        t_after_fn = ag.review_threshold
        assert t_after_fn < ag.DEFAULT_REVIEW_THRESHOLD

        # 2. Duplicate vote with a different secret token on the same URL is deduplicated
        dup = ag.record_feedback(
            "https://x.com/login?token=xyz999", 0.50, "Medium", "Review", "false_negative"
        )
        assert dup["duplicate"] is True
        assert ag.review_threshold == t_after_fn

        # 3. Contradictory verdict for the same URL is rejected
        try:
            ag.record_feedback(
                "https://x.com/login?token=xyz999", 0.50, "Medium", "Review", "correct"
            )
            assert False, "Expected ValueError on contradictory verdict for same URL"
        except ValueError as err:
            assert "Contradictory feedback" in str(err)

        # 4. Correct verdict on a new URL gently stabilizes threshold back toward default
        ag.record_feedback("https://safe.org", 0.05, "Low", "Allow", "correct")
        assert ag.review_threshold > t_after_fn

        # 5. Invalid inputs and action/verdict mismatches are rejected
        for v, act in (
            ("typo", "Review"),
            ("false_positive", "Allow"),
            ("false_negative", "Block"),
        ):
            try:
                ag.record_feedback("https://other.org", 0.5, "Medium", act, v)
                assert False, f"Expected ValueError for ({v}, {act})"
            except ValueError:
                pass

        # 6. Malformed state JSON falls back to replaying valid rows from feedback.csv
        st_path.write_text("{corrupted json", encoding="utf-8")
        ag_recovered = AdaptiveSecurityAgent(fb_path, st_path)
        assert ag_recovered.total_episodes == 2
        assert len(ag_recovered.diagnostics) >= 1

        # 7. Reset restores exact default thresholds
        ag_recovered.reset_policy()
        assert ag_recovered.review_threshold == ag.DEFAULT_REVIEW_THRESHOLD
        assert ag_recovered.block_threshold == ag.DEFAULT_BLOCK_THRESHOLD


def test_untrusted_model_path_and_schema_validation():
    """Priority 2 & 6: Refuse untrusted model paths and reject altered feature orders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        untrusted = Path(tmpdir) / "untrusted.joblib"
        untrusted.write_bytes(b"dummy")
        try:
            load_model_bundle(untrusted, allow_external_path=False)
            assert False, "Expected ValueError for model outside trusted models/ dir"
        except ValueError as err:
            assert "trusted models directory" in str(err)

    bundle = load_model_bundle(MODEL_PATH, force_pure_numpy=True)
    bad_bundle = dict(bundle)
    bad_bundle["features"] = list(reversed(feature_names))
    try:
        _validate_loaded_bundle(bad_bundle)
        assert False, "Expected ValueError on reordered feature_names"
    except ValueError:
        pass


def test_training_determinism_and_zero_domain_leakage():
    """Priority 4: Verify seeded build_dataset is 100% deterministic and splits have zero domain overlap."""
    raw_df = pd.read_csv(DATA_PATH, nrows=500)
    summary = validate_raw_dataset(raw_df)
    assert summary["total_rows"] == 500

    df1 = build_dataset(DATA_PATH, seed=123, phish_per_benign=1.0, max_benign_domains=25)
    df2 = build_dataset(DATA_PATH, seed=123, phish_per_benign=1.0, max_benign_domains=25)
    df3 = build_dataset(DATA_PATH, seed=999, phish_per_benign=1.0, max_benign_domains=25)

    pd.testing.assert_frame_equal(df1, df2)
    assert not df1["url"].equals(df3["url"])

    # Partition groups deterministically by domain hash and verify zero leakage
    unique_domains = sorted(df1["group"].unique())
    split_point = int(len(unique_domains) * 0.8)
    train_doms = set(unique_domains[:split_point])
    train_idx = np.where(df1["group"].isin(train_doms))[0]
    test_idx = np.where(~df1["group"].isin(train_doms))[0]
    verify_no_group_leakage(df1["group"], train_idx, test_idx)


class TestHardeningSuite(unittest.TestCase):
    def test_all_hardening_checks(self):
        test_every_python_file_compiles_and_parses_cleanly()
        test_streamlit_app_startup_smoke()
        test_valid_forms_accepted()
        test_invalid_forms_rejected()
        test_backslash_parser_differential_is_closed()
        test_query_slashes_not_counted_as_subdirs()
        test_benign_superwords_masked_and_compounds_kept()
        test_host_dots_vs_path_dots_explanation_accuracy()
        test_inverted_and_multi_qr()
        test_agent_feedback_policy_contradictions_and_recovery()
        test_untrusted_model_path_and_schema_validation()
        test_training_determinism_and_zero_domain_leakage()


if __name__ == "__main__":
    unittest.main()
