import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from adaptive_agent import AdaptiveSecurityAgent, load_model_bundle
from feature_extractor import feature_names
from qr_utils import (
    MAX_SIDE,
    QRDecodeError,
    decode_qr_code,
    decode_qr_codes,
    generate_qr_image,
)
from test_pipeline import predict_qr_code, predict_url_direct

TESTS_DIR = Path(__file__).resolve().parent
IMAGES_DIR = TESTS_DIR / "images"
PROJECT_DIR = TESTS_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"


def test_sample_benign_qr_dataset_contract_and_decode():
    p = IMAGES_DIR / "sample_benign_qr_code02.png"
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to read {p}"
    assert img.dtype == np.uint8 and img.max() > 1, (
        f"{p.name} is not a valid {{0,255}} uint8 image."
    )
    decoded = decode_qr_code(p)
    assert decoded == "www.ecslimited.com/environ.cfm"


def test_decode_all_synthetic_qr_images():
    expected = {
        "http_url.png": "http://example.com",
        "https_url.png": "https://example.com",
        "ip_url.png": "http://192.0.2.1/login",
        "plain_text.png": "Hello ARES",
        "query_url.png": "https://example.com/login?id=123",
        "test_qr.png": "https://google.com",
    }
    for filename, expected_text in expected.items():
        actual = decode_qr_code(IMAGES_DIR / filename)
        assert (
            actual == expected_text
        ), f"Mismatch for {filename}: {actual!r} != {expected_text!r}"


def test_max_side_3000px_boundary_and_rgba_qr_decode():
    """
    Regression test for Priority 7: an image whose largest side is exactly MAX_SIDE (3000px)
    becomes 3040px after 20px border padding and must still decode at scale=1.
    Also tests RGBA transparent background compositing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        qr_small = generate_qr_image("https://boundary-test.example.com/ok", scale=10)
        # Resize to exactly 3000x3000 (MAX_SIDE)
        qr_3000 = cv2.resize(
            qr_small, (MAX_SIDE, MAX_SIDE), interpolation=cv2.INTER_NEAREST
        )
        p_3000 = tmp / "qr_3000.png"
        cv2.imwrite(str(p_3000), qr_3000)
        assert decode_qr_code(p_3000) == "https://boundary-test.example.com/ok"

        # Create an RGBA image where white modules are transparent (alpha=0) and black modules are opaque
        h, w = qr_small.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        black_mask = qr_small < 128
        rgba[black_mask] = [0, 0, 0, 255]
        p_rgba = tmp / "qr_rgba.png"
        cv2.imwrite(str(p_rgba), rgba)
        assert decode_qr_code(p_rgba) == "https://boundary-test.example.com/ok"


def test_missing_and_invalid_qr_images_raise_errors():
    try:
        decode_qr_code(IMAGES_DIR / "non_existent_qr.png")
        assert False, "Expected FileNotFoundError for missing image"
    except FileNotFoundError:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        blank_path = tmp / "blank.png"
        blank = np.full((100, 100), 255, dtype=np.uint8)
        cv2.imwrite(str(blank_path), blank)
        try:
            decode_qr_code(blank_path)
            assert False, "Expected QRDecodeError when no QR code is present"
        except QRDecodeError:
            pass

        corrupt_path = tmp / "corrupt.png"
        corrupt_path.write_bytes(b"not a valid png file")
        try:
            decode_qr_code(corrupt_path)
            assert False, "Expected QRDecodeError on corrupt image file"
        except QRDecodeError:
            pass


def test_model_bundle_schema_and_pipeline_predictions():
    bundle = load_model_bundle(MODEL_PATH)
    assert list(bundle["features"]) == list(feature_names)
    assert "model_metadata" in bundle

    https_res = predict_qr_code(IMAGES_DIR / "https_url.png")
    assert https_res["decoded_url"] == "https://example.com"
    assert https_res["prediction"] == "Benign"
    assert https_res["recommended_action"] == "Allow"
    # Verify calibrated and uncalibrated base-RF probabilities are distinct and consistent
    assert 0.0 <= https_res["calibrated_phishing_probability"] <= 1.0
    assert 0.0 <= https_res["uncalibrated_rf_phishing_probability"] <= 1.0
    assert (
        https_res["calibrated_phishing_probability"]
        != https_res["uncalibrated_rf_phishing_probability"]
    )
    assert abs(sum(https_res["calibrated_probabilities"].values()) - 1.0) < 1e-5
    assert abs(sum(https_res["uncalibrated_rf_probabilities"].values()) - 1.0) < 1e-5

    ip_res = predict_qr_code(IMAGES_DIR / "ip_url.png")
    assert ip_res["decoded_url"] == "http://192.0.2.1/login"
    assert ip_res["prediction"] == "Phishing"
    assert ip_res["recommended_action"] in ("Review", "Block")

    direct_res = predict_url_direct("https://google.com")
    assert direct_res["prediction"] == "Benign"

    try:
        predict_qr_code(IMAGES_DIR / "plain_text.png")
        assert False, "Expected ValueError for plain-text QR code"
    except ValueError:
        pass


def test_adaptive_security_agent_feedback_loop():
    with tempfile.TemporaryDirectory() as tmpdir:
        fb_csv = Path(tmpdir) / "feedback.csv"
        state_json = Path(tmpdir) / "rl_state.json"
        agent = AdaptiveSecurityAgent(
            feedback_csv_path=fb_csv, state_json_path=state_json
        )
        initial_review = agent.review_threshold

        agent.record_feedback(
            url="http://192.0.2.1/login?token=secret999",
            probability=0.45,
            level="Medium",
            action="Review",
            user_verdict="false_negative",
        )
        assert agent.review_threshold < initial_review
        assert fb_csv.exists()
        assert "secret999" not in fb_csv.read_text(encoding="utf-8")
        assert state_json.exists()


class TestQRAndPipelineSuite(unittest.TestCase):
    def test_all_qr_and_pipeline_checks(self):
        test_sample_benign_qr_dataset_contract_and_decode()
        test_decode_all_synthetic_qr_images()
        test_max_side_3000px_boundary_and_rgba_qr_decode()
        test_missing_and_invalid_qr_images_raise_errors()
        test_model_bundle_schema_and_pipeline_predictions()
        test_adaptive_security_agent_feedback_loop()


if __name__ == "__main__":
    unittest.main()
