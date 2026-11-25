"""Export colorful C2W camera frustum wireframes to PLY and optionally GLB."""

from colorsys import hsv_to_rgb
import json
from pathlib import Path
import struct

import numpy as np


DATA_DIR = Path(r"D:\UE_Render\UE_Rendered\Seq-07-Tower")
POSE_DIR = DATA_DIR / "Pose"
TWC_DIR = POSE_DIR / "T_wc"
K_DIR = POSE_DIR / "K"

FRAME_STEP = 5
OUTPUT_PLY = DATA_DIR / f"camera_viz_{FRAME_STEP}_steps.ply"
is_glb = False

FRUSTUM_DEPTH_METERS = 1.2
LINE_RADIUS_METERS = 0.06
LINE_SIDES = 6

FRUSTUM_EDGES = (
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (2, 3), (3, 4), (4, 1),
)


def collect_matrices(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    files = {}
    for path in folder.glob("*.txt"):
        if path.stem.isdigit():
            files[int(path.stem)] = path

    if not files:
        raise FileNotFoundError(f"No matrix files found in {folder}")
    return files


def load_matrix(path, expected_shape):
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Invalid matrix shape in {path}: "
            f"expected {expected_shape}, got {matrix.shape}"
        )
    return matrix


def trajectory_length_meters(pose_files):
    frames = sorted(pose_files)
    # T_wc stores the camera center in world coordinates in its last column.
    centers = np.array([
        load_matrix(pose_files[frame], (4, 4))[:3, 3]
        for frame in frames
    ])
    if not np.isfinite(centers).all():
        raise ValueError("Camera centers must contain only finite values")
    return float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())


def make_frustum(intrinsic, depth=FRUSTUM_DEPTH_METERS):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    width = 2.0 * cx
    height = 2.0 * cy

    image_corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]
    )
    x = (image_corners[:, 0] - cx) * depth / fx
    y = (image_corners[:, 1] - cy) * depth / fy
    corners = np.column_stack((x, y, np.full(4, depth)))
    return np.vstack((np.zeros((1, 3)), corners))


def transform_points(points, camera_to_world):
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return points @ rotation.T + translation


def make_line(start, end, radius):
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Line endpoints must be different")
    direction /= length

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    axis_x = np.cross(direction, reference)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(direction, axis_x)

    angles = np.linspace(0.0, 2.0 * np.pi, LINE_SIDES, endpoint=False)
    ring = radius * (
        np.cos(angles)[:, None] * axis_x
        + np.sin(angles)[:, None] * axis_y
    )
    vertices = np.vstack((start + ring, end + ring))

    faces = []
    for side in range(LINE_SIDES):
        next_side = (side + 1) % LINE_SIDES
        faces.append((side, next_side, LINE_SIDES + next_side))
        faces.append((side, LINE_SIDES + next_side, LINE_SIDES + side))

    return vertices, faces


def add_frustum_edges(frustum, vertices, faces, radius):
    for start, end in FRUSTUM_EDGES:
        line_vertices, line_faces = make_line(frustum[start], frustum[end], radius)
        offset = len(vertices)
        vertices.extend(line_vertices)
        faces.extend(tuple(offset + index for index in face) for face in line_faces)


def write_ply(path, vertices, colors, faces):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"element face {len(faces)}\n")
        file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")

        for point, color in zip(vertices, colors):
            file.write(
                f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )

        for face in faces:
            file.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def build_wireframes(frames, pose_files, intrinsic_files):
    if not all(np.isfinite(value) and value > 0 for value in (
        FRUSTUM_DEPTH_METERS, LINE_RADIUS_METERS
    )):
        raise ValueError("Frustum depth and line radius must be finite and positive")

    frame_span = max(1, frames[-1] - frames[0])
    vertices, colors, faces = [], [], []
    for frame in frames:
        intrinsic = load_matrix(intrinsic_files[frame], (3, 3))
        camera_to_world = load_matrix(pose_files[frame], (4, 4))
        frustum = transform_points(make_frustum(intrinsic), camera_to_world)

        progress = (frame - frames[0]) / frame_span
        color = tuple(round(255 * channel) for channel in hsv_to_rgb(0.76 * (1.0 - progress), 0.9, 1.0))
        start = len(vertices)
        add_frustum_edges(frustum, vertices, faces, LINE_RADIUS_METERS)
        colors.extend([color] * (len(vertices) - start))

    return vertices, colors, faces


def write_glb(path, vertices, colors, faces):
    positions = np.asarray(vertices, dtype="<f4")
    indices = np.asarray(faces, dtype="<u4").ravel()

    # The PLY palette uses sRGB bytes and glTF expects linear vertex colors.
    srgb = np.asarray(colors, dtype="<f4") / 255.0
    linear_colors = np.where(
        srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
    ).astype("<f4", copy=False)

    color_offset = positions.nbytes
    index_offset = color_offset + linear_colors.nbytes
    binary_length = index_offset + indices.nbytes
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "COLOR_0": 1},
            "indices": 2,
            "material": 0,
        }]}],
        "materials": [{
            "name": "Colorful wireframes",
            "pbrMetallicRoughness": {"metallicFactor": 0.0},
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": True,
        }],
        "buffers": [{"byteLength": binary_length}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": color_offset, "byteLength": linear_colors.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": indices.nbytes, "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": 5126, "count": len(linear_colors), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": indices.size, "type": "SCALAR"},
        ],
    }

    metadata = json.dumps(document, separators=(",", ":")).encode("utf-8")
    metadata += b" " * (-len(metadata) % 4)
    with path.open("wb") as file:
        file.write(struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + binary_length))
        file.write(struct.pack("<I4s", len(metadata), b"JSON"))
        file.write(metadata)
        file.write(struct.pack("<I4s", binary_length, b"BIN\x00"))
        positions.tofile(file)
        linear_colors.tofile(file)
        indices.tofile(file)


if __name__ == "__main__":
    if FRAME_STEP < 1:
        raise ValueError("FRAME_STEP must be at least 1")
    if LINE_SIDES < 3:
        raise ValueError("LINE_SIDES must be at least 3")

    pose_files = collect_matrices(TWC_DIR)
    intrinsic_files = collect_matrices(K_DIR)
    frames = sorted(set(pose_files) & set(intrinsic_files))[::FRAME_STEP]
    if not frames:
        raise ValueError("T_wc and K have no matching frame numbers")

    # Use all poses for distance, regardless of the frustum display step.
    total_distance = trajectory_length_meters(pose_files)
    vertices, colors, faces = build_wireframes(frames, pose_files, intrinsic_files)
    write_ply(OUTPUT_PLY, vertices, colors, faces)
    print(f"Saved: {OUTPUT_PLY} ({len(frames):,} wireframe frustums)")

    if is_glb:
        output_glb = OUTPUT_PLY.with_suffix(".glb")
        write_glb(output_glb, vertices, colors, faces)
        print(f"Saved: {output_glb}")

    print(f"Frustums: {len(frames)}")
    print(f"Total camera distance ({len(pose_files):,} poses): {total_distance:,.3f} m")
