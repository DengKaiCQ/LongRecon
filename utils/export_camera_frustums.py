"""Export C2W camera poses as red frustums in a PLY mesh."""

from pathlib import Path

import numpy as np


DATA_DIR = Path(r"D:\UE_Render\Ruins-MultiCam")
POSE_ROOT = DATA_DIR / "Pose"
OUTPUT_PLY = DATA_DIR / "camera_frustums.ply"

# Keep only the camera views you want to export.
SELECTED_CAMERAS = [
    "Front",
    # "FrontRight",
    # "FrontLeft",
    # "RearRight",
    # "Rear",
    # "RearLeft",
]

FRAME_STEP = 10
FRUSTUM_DEPTH_METERS = 0.5
CAMERA_COLOR = (255, 0, 0)
LINE_RADIUS_METERS = 0.012
LINE_SIDES = 6


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


def make_frustum(intrinsic):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    width = 2.0 * cx
    height = 2.0 * cy
    depth = FRUSTUM_DEPTH_METERS

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


def make_line(start, end):
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
    ring = LINE_RADIUS_METERS * (
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


if __name__ == "__main__":
    if not SELECTED_CAMERAS:
        raise ValueError("SELECTED_CAMERAS cannot be empty")
    if len(set(SELECTED_CAMERAS)) != len(SELECTED_CAMERAS):
        raise ValueError("SELECTED_CAMERAS contains duplicate names")
    if FRAME_STEP < 1:
        raise ValueError("FRAME_STEP must be at least 1")
    if FRUSTUM_DEPTH_METERS <= 0:
        raise ValueError("FRUSTUM_DEPTH_METERS must be positive")
    if LINE_RADIUS_METERS <= 0:
        raise ValueError("LINE_RADIUS_METERS must be positive")
    if LINE_SIDES < 3:
        raise ValueError("LINE_SIDES must be at least 3")

    camera_data = []
    for camera_name in SELECTED_CAMERAS:
        camera_root = POSE_ROOT / camera_name
        pose_files = collect_matrices(camera_root / "T_wc")
        intrinsic_files = collect_matrices(camera_root / "K")
        frames = sorted(set(pose_files) & set(intrinsic_files))[::FRAME_STEP]

        if not frames:
            raise ValueError(
                f"{camera_name}: T_wc and K have no matching frame numbers"
            )

        camera_data.append(
            {
                "name": camera_name,
                "frames": frames,
                "T_wc": pose_files,
                "K": intrinsic_files,
            }
        )

    total_frustums = sum(len(data["frames"]) for data in camera_data)

    frustum_edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    ]

    vertices = []
    colors = []
    faces = []

    exported_frustums = 0
    for data in camera_data:
        for frame in data["frames"]:
            intrinsic = load_matrix(data["K"][frame], (3, 3))
            camera_to_world = load_matrix(data["T_wc"][frame], (4, 4))
            frustum = transform_points(make_frustum(intrinsic), camera_to_world)

            for start_index, end_index in frustum_edges:
                line_vertices, line_faces = make_line(
                    frustum[start_index], frustum[end_index]
                )
                vertex_offset = len(vertices)
                vertices.extend(line_vertices)
                colors.extend([CAMERA_COLOR] * len(line_vertices))
                faces.extend(
                    tuple(vertex_offset + vertex for vertex in face)
                    for face in line_faces
                )

            exported_frustums += 1
            print(
                f"\rFrustums: {exported_frustums}/{total_frustums}, "
                f"camera={data['name']}, frame={frame}",
                end="",
                flush=True,
            )

    print()
    write_ply(
        OUTPUT_PLY,
        np.asarray(vertices),
        np.asarray(colors),
        faces,
    )

    print(f"Saved: {OUTPUT_PLY}")
    print(f"Selected cameras: {', '.join(SELECTED_CAMERAS)}")
    print(f"Frustums: {exported_frustums}")
