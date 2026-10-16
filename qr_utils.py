# Import Path to handle file paths in a platform-independent way
from pathlib import Path

# Import OpenCV for image reading, resizing, padding, and QR-code detection
import cv2

# Import NumPy for array operations
import numpy as np


# Convert an image to the uint8 format required by OpenCV
def image_to_uint8(image):
    """Convert a dataset image to a uint8 image OpenCV can decode."""

    # Ensure the input is a NumPy array
    image = np.asarray(image)

    # Convert Boolean images to uint8: True -> 255, False -> 0
    if image.dtype == bool:
        return (image.astype(np.uint8)) * 255

    # If the image is already uint8, check whether it uses 0/1 values
    if image.dtype == np.uint8:
        # If the maximum value is 1, scale 0/1 values to 0/255
        if image.max() <= 1:
            return image * 255

        # Otherwise, the image is already in the expected 0-255 format
        return image

    # Convert other numeric image types to float for safe normalization
    image = image.astype(np.float32)

    # Find the minimum and maximum pixel values
    minimum = float(image.min())
    maximum = float(image.max())
    difference = maximum - minimum

    # If all pixels have nearly the same value, return an all-black uint8 image
    # This prevents division by zero during normalization
    if difference < 1e-9:
        return np.zeros(image.shape, dtype=np.uint8)

    # Normalize pixel values to 0-1, scale them to 0-255, and convert to uint8
    return (((image - minimum) / difference) * 255).astype(np.uint8)


# Decode a QR code from an image file
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

    # Convert the input path into a Path object
    image_path = Path(image_path)

    # Raise a clear error if the image file does not exist
    if not image_path.exists():
        raise FileNotFoundError(
            f"The specified image path does not exist: {image_path}"
        )

    # Read the image in grayscale.
    # This is important: the dataset QR codes are single-channel images,
    # and reading them as color can prevent OpenCV from detecting them.
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Raise an error if OpenCV could not read the file
    if image is None:
        raise ValueError(
            f"Failed to read the image from the specified path: {image_path}"
        )

    # Ensure the image uses uint8 values in the range 0-255
    image = image_to_uint8(image)

    # Add a 20-pixel white quiet zone around the QR code
    padded_image = cv2.copyMakeBorder(
        image,
        20, 20, 20, 20,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # Create an OpenCV QR-code detector
    detector = cv2.QRCodeDetector()

    # Try several upscale factors because the original QR codes are very small
    for scale in [4, 6, 8, 10]:
        # Enlarge the padded image while keeping QR-code pixels sharp
        enlarged_image = cv2.resize(
            padded_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        # Detect and decode the QR code
        data, points, _ = detector.detectAndDecode(enlarged_image)

        # Clean the decoded text and return it if decoding succeeded
        data = (data or "").strip()
        if data:
            return data

    # Raise an error if no readable QR code was found at any scale
    raise ValueError(f"No QR code found in {image_path.name}")