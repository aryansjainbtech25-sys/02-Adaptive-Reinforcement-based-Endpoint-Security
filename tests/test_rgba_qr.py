from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from qr_utils import load_image_safely, decode_qr_code, generate_qr_image

def main():
    payload = "https://example.com/test-rgba"
    qr = generate_qr_image(payload, scale=4)

    h, w = qr.shape
    # Black-on-transparent: black modules opaque
    rgba_black_on_trans = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_black_on_trans[..., 3] = np.where(qr < 128, 255, 0)
    rgba_black_on_trans[..., 0:3] = 0

    # White-on-transparent: white modules opaque (inverted QR)
    rgba_white_on_trans = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_white_on_trans[..., 3] = np.where(qr >= 128, 255, 0)
    rgba_white_on_trans[..., 0:3] = 255

    Path("results").mkdir(exist_ok=True)
    cv2.imwrite("results/test_black_on_trans.png", rgba_black_on_trans)
    cv2.imwrite("results/test_white_on_trans.png", rgba_white_on_trans)

    print("Testing black-on-transparent...")
    try:
        img1 = load_image_safely(Path("results/test_black_on_trans.png"))
        cv2.imwrite("results/test_black_composited.png", img1)
        payload1 = decode_qr_code(Path("results/test_black_on_trans.png"))
        print(f"  black-on-trans PASS: decoded '{payload1}'")
    except Exception as e:
        print(f"  black-on-trans FAIL: {e}")

    print("Testing white-on-transparent (inverted)...")
    try:
        img2 = load_image_safely(Path("results/test_white_on_trans.png"))
        cv2.imwrite("results/test_white_composited.png", img2)
        payload2 = decode_qr_code(Path("results/test_white_on_trans.png"))
        print(f"  white-on-trans PASS: decoded '{payload2}' (via inversion variant)")
    except Exception as e:
        print(f"  white-on-trans FAIL: {e}")
        print("  Note: White-on-transparent is inverted QR (white modules).")
        print("  Current implementation composites over white (255) → all-white → blank.")
        print("  Optional fix: composite over black (0) as additional variant in _variants,")
        print("  or document as limitation: inverted QRs are non-standard and rare.")

    print("Done")

if __name__ == "__main__":
    main()
