"""Convert RGB PNG and EXR depth files to JPEG and 16-bit PNG."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
from tqdm import tqdm


DATA_DIR = Path(r"D:\UE_Render\Ruins-MultiCam")

SOURCE_RGB_DIR = DATA_DIR / "image_png"
SOURCE_DEPTH_DIR = DATA_DIR / "depth_exr"
OUTPUT_RGB_DIR = DATA_DIR / "image_jpg"
OUTPUT_DEPTH_DIR = DATA_DIR / "depth_png"

FRAME_IDS = None
FRAME_START = None
FRAME_END_EXCLUSIVE = None
FRAME_STEP = 1
MAX_WORKERS = 16

JPEG_QUALITY = 95
JPEG_444_CHROMA = True
PNG_COMPRESSION = 6
OVERWRITE = False

EXR_SCALE_TO_METERS = 200.0
DEPTH_UNITS_PER_METER = 100.0
MIN_DEPTH_METERS = 0.01
MAX_DEPTH_METERS = 655.34
FAR_DEPTH_VALUE = np.uint16(65535)

cv2.setNumThreads(1)


def channel_score(part_name, channel_name):
    name = channel_name.lower()
    full_name = f"{part_name}.{channel_name}".lower()

    score = 0
    if name == "r":
        score += 100
    if name == "z":
        score += 95
    if "depth" in full_name:
        score += 80
    return score


def read_exr_depth(path):
    candidates = []

    with OpenEXR.File(str(path), separate_channels=True) as exr_file:
        for part in exr_file.parts:
            part_name = str(part.name())
            for channel_name, channel in part.channels.items():
                pixels = np.asarray(channel.pixels)
                if pixels.ndim == 2 and pixels.dtype != object:
                    candidates.append(
                        (
                            channel_score(part_name, channel_name),
                            pixels,
                        )
                    )

        if not candidates:
            raise ValueError(f"EXR has no usable depth channel: {path}")

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1].astype(np.float16, copy=True)


def encode_depth_png(depth_exr):
    depth_meters = depth_exr.astype(np.float32) * EXR_SCALE_TO_METERS
    far = depth_meters > MAX_DEPTH_METERS
    valid = (
        np.isfinite(depth_meters)
        & (depth_meters >= MIN_DEPTH_METERS)
        & (depth_meters <= MAX_DEPTH_METERS)
    )

    depth_png = np.zeros(depth_meters.shape, dtype=np.uint16)
    depth_png[far] = FAR_DEPTH_VALUE
    encoded = np.rint(depth_meters[valid] * DEPTH_UNITS_PER_METER)
    depth_png[valid] = np.clip(encoded, 1, 65534).astype(np.uint16)
    return depth_png


def frame_number(path):
    return int(path.stem.rsplit("_", 1)[-1])


def find_frame_ids():
    rgb_ids = {frame_number(path) for path in SOURCE_RGB_DIR.glob("image_*.png")}
    depth_ids = {
        frame_number(path) for path in SOURCE_DEPTH_DIR.glob("depth_*.exr")
    }
    available = sorted(rgb_ids & depth_ids)

    if FRAME_IDS is not None:
        requested = set(FRAME_IDS)
        missing = sorted(requested - set(available))
        if missing:
            raise FileNotFoundError(f"Frames are missing: {missing[:10]}")
        available = [frame for frame in available if frame in requested]

    if FRAME_START is not None:
        available = [frame for frame in available if frame >= FRAME_START]
    if FRAME_END_EXCLUSIVE is not None:
        available = [frame for frame in available if frame < FRAME_END_EXCLUSIVE]

    return available[::FRAME_STEP]


def convert_rgb(source_path, output_path):
    if output_path.exists() and not OVERWRITE:
        return

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {source_path}")

    options = [
        cv2.IMWRITE_JPEG_QUALITY,
        JPEG_QUALITY,
        cv2.IMWRITE_JPEG_OPTIMIZE,
        1,
    ]
    if JPEG_444_CHROMA:
        options.extend(
            [
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ]
        )

    if not cv2.imwrite(str(output_path), image, options):
        raise RuntimeError(f"Cannot write JPEG image: {output_path}")


def convert_depth(source_path, output_path):
    if output_path.exists() and not OVERWRITE:
        return

    depth_exr = read_exr_depth(source_path)
    depth_png = encode_depth_png(depth_exr)
    options = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]

    if not cv2.imwrite(str(output_path), depth_png, options):
        raise RuntimeError(f"Cannot write depth PNG: {output_path}")


def convert_frame(frame):
    convert_rgb(
        SOURCE_RGB_DIR / f"image_{frame:06d}.png",
        OUTPUT_RGB_DIR / f"image_{frame:06d}.jpg",
    )
    convert_depth(
        SOURCE_DEPTH_DIR / f"depth_{frame:06d}.exr",
        OUTPUT_DEPTH_DIR / f"depth_{frame:06d}.png",
    )


if __name__ == "__main__":
    if FRAME_STEP <= 0:
        raise ValueError("FRAME_STEP must be positive")
    if MAX_WORKERS <= 0:
        raise ValueError("MAX_WORKERS must be positive")
    if not 0 <= JPEG_QUALITY <= 100:
        raise ValueError("JPEG_QUALITY must be between 0 and 100")
    frames = find_frame_ids()
    if not frames:
        raise ValueError("No matching RGB and depth frames found")

    OUTPUT_RGB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(convert_frame, frames)
        for _ in tqdm(results, total=len(frames), desc="Converting RGB-D", unit="frame"):
            pass

    print(f"Saved RGB: {OUTPUT_RGB_DIR}")
    print(f"Saved depth: {OUTPUT_DEPTH_DIR}")
    print(f"Frames: {len(frames):,}")
    print("Depth encoding: uint16 centimeters")
    print("Invalid: 0, far/sky: 65535")
