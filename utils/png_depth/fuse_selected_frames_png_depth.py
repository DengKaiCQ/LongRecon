"""Fuse selected JPG and PNG depth frames at full resolution."""

from pathlib import Path

import cv2
import numpy as np

from read_png_depth import read_depth_png


DATA_DIR = Path(r"D:\UE_Render\Ruins-MultiCam-JPG-PNG")
RGB_DIR = DATA_DIR / "RGB"
DEPTH_DIR = DATA_DIR / "Depth"
K_DIR = DATA_DIR / "Pose" / "K"
TWC_DIR = DATA_DIR / "Pose" / "T_wc"
OUTPUT_PLY = DATA_DIR / "selected_frames_full_resolution.ply"

FRAME_IDS = range(100, 115)

DEPTH_MODE = "z"
MIN_DEPTH_METERS = 0.01
MAX_DEPTH_METERS = 100.0

ADD_CAMERA_FRUSTUMS = True
FRUSTUM_DEPTH_METERS = 0.5
FRUSTUM_LINE_RADIUS_METERS = 0.012
FRUSTUM_LINE_SAMPLES = 60
FRUSTUM_LINE_SIDES = 8
FRUSTUM_COLOR = (255, 0, 0)

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


def load_matrix(path, shape):
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"Expected a {shape} matrix: {path}")
    return matrix


def make_point_cloud(frame):
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
    rows, columns = np.nonzero(valid)
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


def make_line(start, end):
    direction = end - start
    direction = direction / np.linalg.norm(direction)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    axis_x = np.cross(direction, reference)
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = np.cross(direction, axis_x)

    angles = np.linspace(0.0, 2.0 * np.pi, FRUSTUM_LINE_SIDES, endpoint=False)
    ring = FRUSTUM_LINE_RADIUS_METERS * (
        np.cos(angles)[:, None] * axis_x
        + np.sin(angles)[:, None] * axis_y
    )
    centers = np.linspace(start, end, FRUSTUM_LINE_SAMPLES)
    return (centers[:, None, :] + ring[None, :, :]).reshape(-1, 3)


def make_camera_frustum(frame):
    image = cv2.imread(
        str(RGB_DIR / f"image_{frame:06d}.jpg"),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image for frame {frame}")

    height, width = image.shape[:2]
    intrinsic = load_matrix(K_DIR / f"{frame:06d}.txt", (3, 3))
    camera_to_world = load_matrix(TWC_DIR / f"{frame:06d}.txt", (4, 4))

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    image_corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]
    )
    x = (image_corners[:, 0] - cx) * FRUSTUM_DEPTH_METERS / fx
    y = (image_corners[:, 1] - cy) * FRUSTUM_DEPTH_METERS / fy
    corners = np.column_stack((x, y, np.full(4, FRUSTUM_DEPTH_METERS)))
    camera_points = np.vstack((np.zeros((1, 3)), corners))

    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    world_points = camera_points @ rotation.T + translation
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    ]
    points = np.vstack(
        [make_line(world_points[start], world_points[end]) for start, end in edges]
    )

    vertices = np.empty(len(points), dtype=PLY_DTYPE)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = FRUSTUM_COLOR[0]
    vertices["green"] = FRUSTUM_COLOR[1]
    vertices["blue"] = FRUSTUM_COLOR[2]
    return vertices


def write_ply(path, vertex_chunks):
    point_count = sum(len(vertices) for vertices in vertex_chunks)
    header = (
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

    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        for vertices in vertex_chunks:
            file.write(vertices.tobytes())

    return point_count


if __name__ == "__main__":
    if not FRAME_IDS:
        raise ValueError("FRAME_IDS is empty")

    vertex_chunks = []

    for index, frame in enumerate(FRAME_IDS, start=1):
        vertices = make_point_cloud(frame)
        vertex_chunks.append(vertices)

        if ADD_CAMERA_FRUSTUMS:
            vertex_chunks.append(make_camera_frustum(frame))

        print(
            f"Frames: {index}/{len(FRAME_IDS)}, "
            f"frame={frame}, points={len(vertices):,}"
        )

    OUTPUT_PLY.parent.mkdir(parents=True, exist_ok=True)
    total_points = write_ply(OUTPUT_PLY, vertex_chunks)

    print(f"Saved: {OUTPUT_PLY}")
    print(f"Total points: {total_points:,}")
