"""Frame preprocessing, geometric transforms and low-light enhancement.

Everything here is pure NumPy/OpenCV and model-agnostic, so it is exercised by
unit tests without any weights on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ibvap.core.types import BBox


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Records how an image was letterboxed so boxes can be mapped back.

    Getting this inverse wrong is the most common source of "detections are
    offset by a few pixels" bugs, and at a border fence a few pixels can be the
    difference between inside and outside a zone. It is therefore an explicit
    object rather than a tuple of loose floats.
    """

    scale: float
    pad_x: float
    pad_y: float
    src_width: int
    src_height: int
    dst_width: int
    dst_height: int

    def to_source(self, box: BBox) -> BBox:
        """Map a box from letterboxed model space back to source pixels."""
        inv = 1.0 / self.scale
        return BBox(
            (box.x1 - self.pad_x) * inv,
            (box.y1 - self.pad_y) * inv,
            (box.x2 - self.pad_x) * inv,
            (box.y2 - self.pad_y) * inv,
        ).clip(self.src_width, self.src_height)

    def to_source_array(self, boxes: np.ndarray) -> np.ndarray:
        """Vectorised inverse for an ``(N, 4)`` array of xyxy boxes."""
        if len(boxes) == 0:
            return boxes
        out = boxes.astype(np.float64, copy=True)
        # Strided slices (0::2 / 1::2) are views, so assignment writes through.
        # Fancy indexing (`out[:, [0, 2]]`) would silently clip a throwaway copy.
        out[:, 0::2] = np.clip((out[:, 0::2] - self.pad_x) / self.scale, 0, self.src_width)
        out[:, 1::2] = np.clip((out[:, 1::2] - self.pad_y) / self.scale, 0, self.src_height)
        return out


def letterbox(
    image: np.ndarray,
    dst_size: tuple[int, int],
    *,
    color: tuple[int, int, int] = (114, 114, 114),
    allow_upscale: bool = True,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio and pad to ``dst_size`` (width, height).

    Aspect-ratio-preserving resize matters more than usual here: border cameras
    are frequently mounted in portrait-ish crops on tower masts, and a naive
    square resize distorts human aspect ratio enough to hurt person recall.
    """
    dst_w, dst_h = dst_size
    src_h, src_w = image.shape[:2]

    scale = min(dst_w / src_w, dst_h / src_h)
    if not allow_upscale:
        scale = min(scale, 1.0)

    new_w, new_h = max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))
    # INTER_AREA is materially better than INTER_LINEAR when downscaling, which
    # is the usual direction (1080p source -> 640 px model input).
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_w, pad_h = dst_w - new_w, dst_h - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, LetterboxTransform(scale, left, top, src_w, src_h, dst_w, dst_h)


def to_nchw(
    image: np.ndarray,
    *,
    to_rgb: bool = True,
    normalize: bool = True,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Convert an HWC BGR uint8 image to a 1xCxHxW float32 tensor."""
    img = image[:, :, ::-1] if to_rgb else image
    tensor = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    if normalize:
        tensor /= 255.0
    if mean is not None:
        tensor -= np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    if std is not None:
        tensor /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return tensor[None, ...]


def crop(image: np.ndarray, box: BBox, *, padding: float = 0.0) -> np.ndarray:
    """Crop ``box`` from ``image``, optionally expanded by ``padding``.

    Returns an empty array when the box lies entirely outside the frame, so
    callers can skip with a cheap ``.size`` check instead of a try/except.
    """
    h, w = image.shape[:2]
    b = box.expand(padding).clip(w, h) if padding else box.clip(w, h)
    x1, y1, x2, y2 = b.as_int_tuple()
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2]


# --------------------------------------------------------------------------- #
# Low-light handling
# --------------------------------------------------------------------------- #


def mean_luma(image: np.ndarray) -> float:
    """Mean perceptual luminance in ``[0, 255]``.

    Sub-sampled: a full-frame mean at 25 fps across 32 cameras is pure waste
    when a 1-in-16 stride gives the same answer to well within a grey level.
    """
    if image.ndim == 2:
        return float(image[::4, ::4].mean())
    sub = image[::4, ::4]
    b, g, r = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    return float((0.114 * b + 0.587 * g + 0.299 * r).mean())


def is_night_frame(image: np.ndarray, threshold: float = 60.0) -> bool:
    """Classify a frame as night/IR by luminance."""
    return mean_luma(image) < threshold


def is_infrared(image: np.ndarray, tolerance: float = 6.0) -> bool:
    """Detect a monochrome IR frame delivered over a 3-channel stream.

    Night-mode CCTV switches to IR illumination and emits a grey image in a
    colour container. Knowing this lets downstream stages skip colour-dependent
    logic (vehicle colour attributes, for one) instead of reporting nonsense.
    """
    if image.ndim == 2:
        return True
    sub = image[::8, ::8].astype(np.int16)
    spread = np.abs(sub - sub.mean(axis=2, keepdims=True)).mean()
    return bool(spread < tolerance)


def enhance_low_light(
    image: np.ndarray,
    *,
    clip_limit: float = 2.5,
    tile_grid: int = 8,
    gamma: float = 0.85,
    denoise: bool = False,
) -> np.ndarray:
    """Improve detectability in dark frames via CLAHE on the luma channel.

    CLAHE is applied in LAB space so chroma is untouched - equalising RGB
    channels independently shifts colour and wrecks vehicle-colour attributes.
    Contrast-limited (rather than global) equalisation is essential for IR
    scenes, where a single bright illuminator would otherwise crush the rest of
    the frame to black.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lightness = clahe.apply(lightness)
    out = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)

    if gamma and abs(gamma - 1.0) > 1e-3:
        out = apply_gamma(out, gamma)
    if denoise:
        # Bilateral filtering preserves edges (limb boundaries, plate glyphs)
        # while removing the sensor noise that dominates high-gain night frames.
        out = cv2.bilateralFilter(out, d=5, sigmaColor=45, sigmaSpace=45)
    return out


_GAMMA_CACHE: dict[int, np.ndarray] = {}


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply a gamma curve using a cached 256-entry lookup table."""
    key = int(round(gamma * 1000))
    lut = _GAMMA_CACHE.get(key)
    if lut is None:
        inv = 1.0 / max(1e-3, gamma)
        lut = np.clip(((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255).astype(np.uint8)
        if len(_GAMMA_CACHE) < 64:  # bounded: gamma comes from config, not data
            _GAMMA_CACHE[key] = lut
    return cv2.LUT(image, lut)


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian - a standard focus/blur measure.

    Used by tamper detection to catch a defocused or sprayed lens, and by ANPR
    to reject plate crops too blurred to read.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def resize_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is ``max_side``. Returns image and scale."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized = cv2.resize(
        image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale
