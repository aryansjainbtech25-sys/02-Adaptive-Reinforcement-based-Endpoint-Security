import cv2
import numpy as np
from pathlib import Path

from qr_utils import decode_qr_code

p = Path("images/sample_benign_qr_code02.png")

img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

assert img is not None, f"Failed to read {p}"

uniq = np.unique(img)

# Validate dataset contract: QR images must be {0,255} uint8, not {0,1}
if not (img.dtype == np.uint8 and img.max() > 1):
    raise SystemExit(
        f"⚠  {p.name} is not a valid {{0,255}} uint8 image. "
        f"dtype={img.dtype}, min/max={img.min()}/{img.max()}, "
        f"unique={uniq[:5]}...\n"
        f"   Regenerate it with: "
        f"cv2.imwrite(path, (X[idx].astype(np.uint8)) * 255)"
    )

try:
    result = decode_qr_code(p)
    print(f"Decoded QR-Code data: {result}")

except FileNotFoundError as error:
    print(f"Decoding QR-Code failed: {error}")

except ValueError as error:
    print(f"Decoding QR-Code failed: {error}")
