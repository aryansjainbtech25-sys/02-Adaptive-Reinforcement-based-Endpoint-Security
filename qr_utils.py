from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

MAX_PIXELS = 25_000_000  # Reject decompression bombs before full pixel allocation
MAX_SIDE = 3000          # Downscale images larger than 3000px per side
BORDER_PAD = 20          # Quiet-zone white border added around images


class QRDecodeError(ValueError):
    """Raised when an image cannot be read, exceeds safety limits, or contains 0 or >1 QR codes."""


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a dataset or uploaded image array to a uint8 grayscale image OpenCV can decode."""
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

    # Avoid division by zero for flat/blank images
    if difference < 1e-9:
        return np.zeros(image.shape, dtype=np.uint8)

    return (((image - minimum) / difference) * 255).astype(np.uint8)


def load_image_safely(image_path: Path) -> np.ndarray:
    """Header-check pixel count, composite RGBA alpha onto white, and return uint8 grayscale."""
    from PIL import Image

    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception as exc:
        raise QRDecodeError(f"Unreadable image: {exc}") from exc

    if w <= 0 or h <= 0:
        raise QRDecodeError(f"Invalid image dimensions ({w}x{h}).")
    if w * h > MAX_PIXELS:
        raise QRDecodeError(
            f"Image too large ({w}x{h} = {w * h:,} pixels); safety limit is {MAX_PIXELS:,} pixels."
        )

    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.size == 0:
        raise QRDecodeError(f"Failed to read the image: {image_path}")

    if raw.ndim == 3 and raw.shape[2] == 4:
        # Composite transparent background onto white (255) so black QR modules remain visible
        alpha = raw[..., 3:4].astype(np.float32) / 255.0
        rgb = raw[..., :3].astype(np.float32)
        raw = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)

    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    img = image_to_uint8(raw)
    if max(img.shape[:2]) > MAX_SIDE:
        scale_factor = MAX_SIDE / float(max(img.shape[:2]))
        img = cv2.resize(
            img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
        )
    return img


def _detect_all(detector: cv2.QRCodeDetector, img: np.ndarray) -> list[str]:
    """Decode every QR code in `img` (multi-detector first, single-detector fallback)."""
    found: list[str] = []
    try:
        ok, infos, _, _ = detector.detectAndDecodeMulti(img)
        if ok and infos is not None:
            found.extend([t.strip() for t in infos if t and t.strip()])
    except cv2.error:
        pass
    if not found:
        try:
            data, _, _ = detector.detectAndDecode(img)
            if data and data.strip():
                found.append(data.strip())
        except cv2.error:
            pass
    return found


def _variants(img: np.ndarray):
    """
    Yield progressively preprocessed candidate images (and their inverted counterparts).
    Always yields `scale = 1` even when adding the 20px quiet-zone border pushes a
    3000px image to 3040px, fixing the boundary bug where 3000px images skipped all scales.
    """
    padded = cv2.copyMakeBorder(
        img,
        BORDER_PAD,
        BORDER_PAD,
        BORDER_PAD,
        BORDER_PAD,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    max_padded_limit = MAX_SIDE + (2 * BORDER_PAD)
    scales = [1, 2, 4, 6, 8, 10] if min(padded.shape[:2]) < 250 else [1, 2]

    for scale in scales:
        if scale > 1 and max(padded.shape[:2]) * scale > max_padded_limit:
            break
        base = (
            padded
            if scale == 1
            else cv2.resize(
                padded,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
        )
        yield base
        yield 255 - base

    _, otsu = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        padded,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    for binarized in (otsu, adaptive):
        yield binarized
        yield 255 - binarized


def decode_qr_codes(image_path: str | Path) -> list[str]:
    """Return ALL distinct QR payloads found in the image (in discovery order). Raises if none."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(
            f"The specified image path does not exist: {image_path}"
        )

    img = load_image_safely(image_path)
    detector = cv2.QRCodeDetector()
    best: list[str] = []

    for candidate in _variants(img):
        found = _detect_all(detector, candidate)
        if len(set(found)) > len(set(best)):
            best = found
        if len(set(best)) > 1:
            break

    if not best:
        raise QRDecodeError(f"No QR code found in {image_path.name}")
    return list(dict.fromkeys(best))


def decode_qr_code(image_path: str | Path) -> str:
    """Single-payload QR decoder API. Fails closed when the image contains multiple distinct QR codes."""
    payloads = decode_qr_codes(image_path)
    if len(payloads) > 1:
        raise QRDecodeError(
            f"{len(payloads)} different QR codes found; refusing to guess which one a phone would open."
        )
    return payloads[0]


def generate_qr_image(payload: str, scale: int = 8) -> np.ndarray:
    """
    Generate a uint8 QR-code matrix with a quiet zone for any non-empty string.
    Note: Encoding a URL into a QR image does not validate or make the destination safe.
    """
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("Payload must be a non-empty string to generate a QR code.")
    if len(payload.strip()) > 2048:
        raise ValueError("Payload exceeds maximum supported QR character length (2048 chars).")
    if scale < 1 or scale > 20:
        raise ValueError("QR scale factor must be between 1 and 20.")

    encoder = cv2.QRCodeEncoder.create()
    qr = encoder.encode(payload.strip())
    if qr is None or qr.size == 0:
        raise QRDecodeError("OpenCV QRCodeEncoder failed to encode the supplied payload.")
    padded = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
