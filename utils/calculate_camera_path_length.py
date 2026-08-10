"""Calculate the total camera path length from T_wc poses."""

from math import sqrt
from pathlib import Path


POSE_DIR = Path(r"D:\UE_Render\Factory\Pose\T_wc")
FRAME_STEP = 1


def read_camera_center(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append([float(value) for value in line.split()])

    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"Expected a 4x4 matrix: {path}")

    return rows[0][3], rows[1][3], rows[2][3]


def collect_pose_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Pose directory does not exist: {folder}")

    files = [path for path in folder.glob("*.txt") if path.stem.isdigit()]
    files.sort(key=lambda path: int(path.stem))
    return files[::FRAME_STEP]


def point_distance(point_a, point_b):
    return sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


if __name__ == "__main__":
    pose_files = collect_pose_files(POSE_DIR)

    if len(pose_files) < 2:
        raise RuntimeError(f"At least two pose files are required: {POSE_DIR}")

    camera_centers = [read_camera_center(path) for path in pose_files]
    total_length = sum(
        point_distance(previous, current)
        for previous, current in zip(camera_centers, camera_centers[1:])
    )

    print(f"Pose directory: {POSE_DIR}")
    print(f"Pose count: {len(pose_files)}")
    print(f"Frame range: {pose_files[0].stem} -> {pose_files[-1].stem}")
    print(f"Total camera path length: {total_length:.3f} m")
