"""Fuse JPEG RGB and uint16 centimeter depth frames into a binary PLY."""

from pathlib import Path
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from read_png_depth import read_depth_png


DATA_DIR = Path(r"D:\UE_Render\Ruins-MultiCam-JPG-PNG")
RGB_DIR = DATA_DIR / "RGB"
DEPTH_DIR = DATA_DIR / "Depth"
K_DIR = DATA_DIR / "Pose" / "K"
TWC_DIR = DATA_DIR / "Pose" / "T_wc"
OUTPUT_PLY = DATA_DIR / "global_pcd_from_png.ply"

FRAME_IDS = None
FRAME_START = None
FRAME_END_EXCLUSIVE = None
FRAME_STEP = 1

SAMPLE_RATIO = 0.0025
DEPTH_MODE = "z"
MIN_DEPTH_METERS = 0.01
MAX_DEPTH_METERS = 100.0
RANDOM_SEED = 42
OVERWRITE_OUTPUT = True

PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def frame_number(path):
    return int(path.stem.rsplit("_", 1)[-1])


def find_frame_ids():
    rgb_ids = {frame_number(path) for path in RGB_DIR.glob("image_*.jpg")}
    depth_ids = {frame_number(path) for path in DEPTH_DIR.glob("depth_*.png")}
    pose_ids = {int(path.stem) for path in TWC_DIR.glob("*.txt")}
    intrinsic_ids = {int(path.stem) for path in K_DIR.glob("*.txt")}
    available = sorted(rgb_ids & depth_ids & pose_ids & intrinsic_ids)

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


def load_matrix(path, shape):
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"Expected a {shape} matrix: {path}")
    return matrix


def make_vertices(frame, random_generator):
    rgb_path = RGB_DIR / f"image_{frame:06d}.jpg"
    depth_path = DEPTH_DIR / f"depth_{frame:06d}.png"

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {rgb_path}")
    depth = read_depth_png(depth_path)

    if image.shape[:2] != depth.shape:
        raise ValueError(f"RGB and depth size mismatch at frame {frame}")

    valid = (
        np.isfinite(depth)
        & (depth >= MIN_DEPTH_METERS)
        & (depth <= MAX_DEPTH_METERS)
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return np.empty(0, dtype=PLY_DTYPE)

    sample_count = max(1, int(round(len(valid_indices) * SAMPLE_RATIO)))
    if sample_count < len(valid_indices):
        valid_indices = random_generator.choice(
            valid_indices,
            size=sample_count,
            replace=False,
        )

    height, width = depth.shape
    rows, columns = np.divmod(valid_indices, width)
    sampled_depth = depth[rows, columns]

    intrinsic = load_matrix(K_DIR / f"{frame:06d}.txt", (3, 3))
    camera_to_world = load_matrix(TWC_DIR / f"{frame:06d}.txt", (4, 4))

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x_normalized = (columns - cx) / fx
    y_normalized = (rows - cy) / fy

    if DEPTH_MODE == "z":
        z = sampled_depth
    elif DEPTH_MODE == "ray":
        z = sampled_depth / np.sqrt(
            x_normalized**2 + y_normalized**2 + 1.0
        )
    else:
        raise ValueError('DEPTH_MODE must be "z" or "ray"')

    camera_points = np.column_stack(
        (x_normalized * z, y_normalized * z, z)
    )
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    world_points = camera_points @ rotation.T + translation

    vertices = np.empty(len(world_points), dtype=PLY_DTYPE)
    vertices["x"] = world_points[:, 0]
    vertices["y"] = world_points[:, 1]
    vertices["z"] = world_points[:, 2]
    vertices["red"] = image[rows, columns, 2]
    vertices["green"] = image[rows, columns, 1]
    vertices["blue"] = image[rows, columns, 0]
    return vertices


def ply_header(point_count):
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )


if __name__ == "__main__":
    if not 0 < SAMPLE_RATIO <= 1:
        raise ValueError("SAMPLE_RATIO must be in (0, 1]")
    if FRAME_STEP <= 0:
        raise ValueError("FRAME_STEP must be positive")
    if OUTPUT_PLY.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(f"Output already exists: {OUTPUT_PLY}")

    frames = find_frame_ids()
    if not frames:
        raise ValueError("No complete RGB-D-pose frames found")

    OUTPUT_PLY.parent.mkdir(parents=True, exist_ok=True)
    temporary_body = OUTPUT_PLY.with_suffix(".vertices.tmp")
    random_generator = np.random.default_rng(RANDOM_SEED)
    point_count = 0

    with temporary_body.open("wb") as body_file:
        for frame in tqdm(frames, desc="Fusing RGB-D", unit="frame"):
            vertices = make_vertices(frame, random_generator)
            body_file.write(vertices.tobytes())
            point_count += len(vertices)

    with OUTPUT_PLY.open("wb") as ply_file:
        ply_file.write(ply_header(point_count).encode("ascii"))
        with temporary_body.open("rb") as body_file:
            shutil.copyfileobj(body_file, ply_file, length=16 * 1024 * 1024)

    temporary_body.unlink()

    print(f"Saved: {OUTPUT_PLY}")
    print(f"Frames: {len(frames):,}")
    print(f"Points: {point_count:,}")
