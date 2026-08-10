"""Read a uint16 centimeter depth PNG in meters."""

from pathlib import Path

import cv2
import numpy as np


DEPTH_UNITS_PER_METER = 100.0


def read_depth_png(path):
    depth_png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise FileNotFoundError(f"Cannot read depth PNG: {path}")
    if depth_png.ndim != 2 or depth_png.dtype != np.uint16:
        raise ValueError(f"Expected a single-channel uint16 PNG: {path}")

    invalid = (depth_png == 0) | (depth_png == 65535)
    depth_meters = depth_png.astype(np.float32) / DEPTH_UNITS_PER_METER
    depth_meters[invalid] = np.nan
    return depth_meters


if __name__ == "__main__":
    DEPTH_PATH = Path(
        r"D:\UE_Render\Ruins-MultiCam-JPG-PNG\Depth\depth_000000.png"
    )
    SAVE_PREVIEW = False
    PREVIEW_PATH = DEPTH_PATH.with_name("depth_000000_preview.png")
    PREVIEW_MAX_DEPTH_METERS = 70.0

    depth = read_depth_png(DEPTH_PATH)
    valid = np.isfinite(depth)

    print(f"Path: {DEPTH_PATH}")
    print(f"Shape: {depth.shape}")
    print(f"Valid pixels: {np.count_nonzero(valid):,}")
    print(f"Depth range: {depth[valid].min():.3f} to {depth[valid].max():.3f} m")

    if SAVE_PREVIEW:
        normalized = np.nan_to_num(depth, nan=0.0)
        normalized = np.clip(normalized / PREVIEW_MAX_DEPTH_METERS, 0.0, 1.0)
        preview = np.rint(normalized * 255.0).astype(np.uint8)
        preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(PREVIEW_PATH), preview)
        print(f"Preview: {PREVIEW_PATH}")
