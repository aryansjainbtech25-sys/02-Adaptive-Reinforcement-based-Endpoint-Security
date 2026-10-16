import cv2
from qr_utils import decode_qr_code

# Create a simple test QR-code image using OpenCV
qr = cv2.QRCodeEncoder.create()
test_qr = qr.encode("https://google.com")

cv2.imwrite("trials/results/test_qr.png", test_qr)

print(decode_qr_code("trials/results/test_qr.png"))