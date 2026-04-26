"""
JPEG / raw image encoding shared between client and server.

The JPEG_ENCODE flag toggles between two paths:
  - JPEG: fast over the network, ~5 ms encode/decode each, lossy.
  - Raw: bit-perfect, ~20× larger, requires lower fps to fit Ngrok bandwidth.

Both paths produce/consume the same CameraFrame schema; the difference is
the `encoding` field and how `data_b64` is interpreted.

Bandwidth math (640x480x3 frames, 30 fps, 2 cameras):
  JPEG q85: ~45 KB/frame  →  ~22 Mbps      ✓ fits Ngrok free tier
  Raw RGB:  ~900 KB/frame →  ~440 Mbps     ✗ will not fit; throttle to 5 fps
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from shared.protocol import CameraFrame


# ---------------------------------------------------------------------------
# Encoding (client side)
# ---------------------------------------------------------------------------

def encode_frame(name: str, rgb: np.ndarray, *,
                 use_jpeg: bool, jpeg_quality: int = 85) -> CameraFrame:
    """
    Encode an HxWx3 uint8 RGB array as a CameraFrame.

    use_jpeg=True  → JPEG compression (default, ~45 KB)
    use_jpeg=False → Raw RGB bytes base64'd (~900 KB)
    """
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got shape {rgb.shape}")

    h, w, _ = rgb.shape

    if use_jpeg:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=jpeg_quality)
        data_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        encoding = "jpeg"
    else:
        # Raw row-major RGB uint8.
        data_b64 = base64.b64encode(rgb.tobytes()).decode("ascii")
        encoding = "raw"

    return CameraFrame(
        name=name,
        encoding=encoding,
        width=w,
        height=h,
        data_b64=data_b64,
    )


# ---------------------------------------------------------------------------
# Decoding (server side)
# ---------------------------------------------------------------------------

def decode_frame(frame: CameraFrame) -> np.ndarray:
    """Decode a CameraFrame back to HxWx3 uint8 RGB."""
    raw_bytes = base64.b64decode(frame.data_b64)

    if frame.encoding == "jpeg":
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    elif frame.encoding == "raw":
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
            frame.height, frame.width, 3
        ).copy()  # .copy() because frombuffer is read-only
    else:
        raise ValueError(f"Unknown encoding: {frame.encoding}")

    if arr.shape != (frame.height, frame.width, 3):
        raise ValueError(
            f"Decoded shape {arr.shape} doesn't match declared "
            f"({frame.height}, {frame.width}, 3)"
        )
    return arr


def estimate_bandwidth_mbps(width: int, height: int, fps: int,
                            n_cameras: int, *, use_jpeg: bool,
                            jpeg_quality: int = 85) -> float:
    """
    Rough bandwidth estimate for the configured stream.

    Used by the client at startup to print a warning if the user has chosen
    raw mode at high fps — that combination doesn't fit Ngrok free tier.
    """
    if use_jpeg:
        # JPEG size scales roughly with pixels and inversely with quality
        # squared. Empirical: q85 on a typical scene is ~0.15 bytes/pixel.
        bytes_per_frame = width * height * 0.15 * (jpeg_quality / 85.0) ** 2
    else:
        bytes_per_frame = width * height * 3
    return (bytes_per_frame * fps * n_cameras * 8) / 1_000_000
