from pathlib import Path
import cv2
import numpy as np

def image_to_uint8(image):
    """Convert a dataset image to a uint8 image OpenCV can decode."""
    image = np.asarray(image)

    if image.dtype == bool:
        return (image.astype(np.uint8)) * 255

    if image.dtype == np.uint8:
        if image.max() <= 1:
            return image * 255
        return image

    image = image.astype(np.float32)

    minimum = float(image.min())
    maximum = float(image.max())
    difference = maximum - minimum

    # Avoid division by zero for flat images
    if difference < 1e-9:
        return np.zeros(image.shape, dtype=np.uint8)

    return (((image - minimum) / difference) * 255).astype(np.uint8)

def decode_qr_code(image_path: str) -> str:
    """
    Decode a QR code from an image file.

    Args:
        image_path: Path to the image containing the QR code.

    Returns:
        The decoded QR-code text.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be read or no QR code is found.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"The specified image path does not exist: {image_path}"
        )

    # Grayscale is required - dataset QR codes are single-channel and fail detection in color mode
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Failed to read the image from the specified path: {image_path}"
        )

    image = image_to_uint8(image)

    # QR spec requires quiet zone; dataset images lack it, so add white border
    padded_image = cv2.copyMakeBorder(
        image,
        20, 20, 20, 20,
        cv2.BORDER_CONSTANT,
        value=255
    )

    detector = cv2.QRCodeDetector()

    # Original codes are very small ( ~21px ), need upscaling for reliable detection
    for scale in [4, 6, 8, 10]:
        enlarged_image = cv2.resize(
            padded_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        data, points, _ = detector.detectAndDecode(enlarged_image)

        data = (data or "").strip()
        if data:
            return data

    raise ValueError(f"No QR code found in {image_path.name}")
