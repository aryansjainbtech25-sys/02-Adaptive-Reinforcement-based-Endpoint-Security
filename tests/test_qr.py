# Import OpenCV for reading and checking the image
import cv2

# Import NumPy for array operations
import numpy as np

# Import Path for handling the image file path
from pathlib import Path

# Import the QR-code decoding function from qr_utils.py
from qr_utils import decode_qr_code

# Define the path to the saved benign QR-code image
p = Path("trials/results/sample_benign_qr_code02.png")

# Read the image in grayscale
img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

# Stop with a clear message if the image could not be read
assert img is not None, f"Failed to read {p}"

# Find the distinct pixel values in the image
uniq = np.unique(img)

# Check that the image is uint8 and uses the expected 0/255 range
if not (img.dtype == np.uint8 and img.max() > 1):
    raise SystemExit(
        f"⚠️  {p.name} is not a valid {{0,255}} uint8 image. "
        f"dtype={img.dtype}, min/max={img.min()}/{img.max()}, "
        f"unique={uniq[:5]}...\n"
        f"   Regenerate it with: "
        f"cv2.imwrite(path, (X[idx].astype(np.uint8)) * 255)"
    )

# Try to decode the QR code and handle possible errors
try:
    result = decode_qr_code(p)
    print(f"Decoded QR-Code data: {result}")

except FileNotFoundError as error:
    print(f"Decoding QR-Code failed: {error}")

except ValueError as error:
    print(f"Decoding QR-Code failed: {error}")