"""

Fuse RGB-D frames and camera poses into a point cloud.
Stream Ply I/O code from https://github.com/DengkaiCQ/VGGT-Long

"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np

from exr_depth import describe_channel, read_depth_channel


DATA_DIR = Path(r"D:\UE_Render\Factory")
RGB_DIR = DATA_DIR / "RGB"
DEPTH_DIR = DATA_DIR / "Depth"
POSE_DIR = DATA_DIR / "Pose"
K_DIR = POSE_DIR / "K"
TWC_DIR = POSE_DIR / "T_wc"
OUTPUT_PLY = DATA_DIR / "global_pcd.ply"
TEMP_PLY_DIR = DATA_DIR / "_point_cloud_chunks"

STORAGE_MODE = "memory" # memory or disk
ESTIMATE_FRAMES = 150
LARGE_PLY_GIB = 1.0
WORKER_COUNT = 16

SAMPLE_RATIO = 0.00125
DEPTH_SCALE_TO_METERS = 200.0
DEPTH_MODE = "z"
MIN_DEPTH_METERS = 0.0005
MAX_DEPTH_METERS = 1000.0
RANDOM_SEED = 2026
MAX_FRAMES = None
FRAMES_PER_CHUNK = 20
COPY_BUFFER_SIZE = 10 * 1024 * 1024

RGB_FRAME_OFFSET = 0
DEPTH_FRAME_OFFSET = 0
K_FRAME_OFFSET = 0
POSE_FRAME_OFFSET = 0

RGB_PATTERN = re.compile(r"image_(\d+)\.png$", re.IGNORECASE)
DEPTH_PATTERN = re.compile(r"depth_(\d+)\.exr$", re.IGNORECASE)
TXT_PATTERN = re.compile(r"^(\d+)\.txt$", re.IGNORECASE)

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


def collect_files(folder, pattern, offset, label):
    if not folder.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {folder}")

    files = {}
    for path in folder.iterdir():
        match = pattern.search(path.name)
        if not path.is_file() or match is None:
            continue

        frame = int(match.group(1)) + offset
        if frame in files:
            raise ValueError(
                f"Multiple {label} files for frame {frame}:\n"
                f"{files[frame]}\n{path}"
            )
        files[frame] = path

    if not files:
        raise FileNotFoundError(f"No {label} files found in {folder}")

    print(f"{label}: {len(files)} files, frames [{min(files)}, {max(files)}]")
    return files


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path, show_channels=False):
    depth, channels = read_depth_channel(path)

    if show_channels:
        print(f"Selected EXR channel: {describe_channel(channels[0])}")
        print("Available 2D channels:")
        for channel in channels:
            print(f"  {describe_channel(channel)}")

    return depth * DEPTH_SCALE_TO_METERS


def load_matrix(path, expected_shape, label):
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Invalid {label} shape in {path}: "
            f"expected {expected_shape}, got {matrix.shape}"
        )
    return matrix


def reservoir_sample(values, sample_count, random_generator):
    reservoir = values[:sample_count].copy()
    remaining = values[sample_count:]
    seen_counts = np.arange(sample_count + 1, len(values) + 1)
    replacements = random_generator.integers(0, seen_counts)
    selected = replacements < sample_count
    reservoir[replacements[selected]] = remaining[selected]
    return reservoir


def select_pixels(depth, frame):
    valid = (
        np.isfinite(depth)
        & (depth >= MIN_DEPTH_METERS)
        & (depth <= MAX_DEPTH_METERS)
    )
    valid_indices = np.flatnonzero(valid)
    sample_count = int(valid_indices.size * SAMPLE_RATIO)

    if sample_count >= valid_indices.size:
        return valid_indices
    if sample_count == 0:
        return np.empty(0, dtype=np.int64)

    random_generator = np.random.default_rng(RANDOM_SEED + frame)
    return reservoir_sample(valid_indices, sample_count, random_generator)


def project_pixels(depth, selected, intrinsic, camera_to_world):
    height, width = depth.shape
    rows, columns = np.unravel_index(selected, (height, width))

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (columns - cx) / fx
    y = (rows - cy) / fy
    sampled_depth = depth.ravel()[selected]

    if DEPTH_MODE == "z":
        z = sampled_depth
    elif DEPTH_MODE == "ray":
        z = sampled_depth / np.sqrt(x**2 + y**2 + 1.0)
    else:
        raise ValueError('DEPTH_MODE must be "z" or "ray"')

    camera_points = np.column_stack((x * z, y * z, z))
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return camera_points @ rotation.T + translation


def process_frame(
    frame,
    rgb_path,
    depth_path,
    k_path,
    pose_path,
    show_channels,
):
    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path, show_channels)
    intrinsic = load_matrix(k_path, (3, 3), "K")
    camera_to_world = load_matrix(pose_path, (4, 4), "T_wc")

    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"Frame {frame} size mismatch: RGB={rgb.shape[:2]}, depth={depth.shape}"
        )

    selected = select_pixels(depth, frame)
    if selected.size == 0:
        empty_points = np.empty((0, 3), dtype=np.float64)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        return empty_points, empty_colors, depth

    points = project_pixels(depth, selected, intrinsic, camera_to_world)
    colors = rgb.reshape(-1, 3)[selected]
    return points, colors, depth


def process_frames(frames, rgb_files, depth_files, k_files, pose_files):
    pending = {}
    next_submit = 0

    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        while next_submit < min(WORKER_COUNT, len(frames)):
            frame = frames[next_submit]
            pending[next_submit] = executor.submit(
                process_frame,
                frame,
                rgb_files[frame],
                depth_files[frame],
                k_files[frame],
                pose_files[frame],
                next_submit == 0,
            )
            next_submit += 1

        for frame_index, frame in enumerate(frames):
            points, colors, depth = pending.pop(frame_index).result()

            if next_submit < len(frames):
                next_frame = frames[next_submit]
                pending[next_submit] = executor.submit(
                    process_frame,
                    next_frame,
                    rgb_files[next_frame],
                    depth_files[next_frame],
                    k_files[next_frame],
                    pose_files[next_frame],
                    False,
                )
                next_submit += 1

            yield frame_index + 1, frame, points, colors, depth


def write_ply_header(file, point_count):
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
    file.write(header.encode("ascii"))


def pack_vertices(points, colors):
    vertices = np.empty(len(points), dtype=PLY_DTYPE)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]

    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    return vertices


def write_ply_points(file, points, colors):
    file.write(pack_vertices(points, colors).tobytes())


def write_memory_ply(path, vertex_chunks, point_count):
    with path.open("wb") as file:
        write_ply_header(file, point_count)
        for vertices in vertex_chunks:
            file.write(vertices.tobytes())


def save_chunk(path, point_chunks, color_chunks):
    points = np.concatenate(point_chunks)
    colors = np.concatenate(color_chunks)

    with path.open("wb") as file:
        write_ply_header(file, len(points))
        write_ply_points(file, points, colors)


def read_ply_point_count(path):
    with path.open("rb") as file:
        for line in file:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                break

    raise ValueError(f"PLY vertex count not found: {path}")


def skip_ply_header(file):
    while True:
        line = file.readline()
        if not line:
            raise ValueError("Unexpected end of PLY header")
        if line.startswith(b"end_header"):
            return


def merge_ply_files(chunk_paths, output_path):
    total_points = sum(read_ply_point_count(path) for path in chunk_paths)

    with output_path.open("wb") as output_file:
        write_ply_header(output_file, total_points)

        for index, path in enumerate(chunk_paths, start=1):
            print(
                f"\rMerging: {index}/{len(chunk_paths)} chunks",
                end="",
                flush=True,
            )
            with path.open("rb") as input_file:
                skip_ply_header(input_file)
                shutil.copyfileobj(input_file, output_file, COPY_BUFFER_SIZE)

    print()
    return total_points


def confirm_estimated_size(processed_points, processed_frames, total_frames):
    estimated_points = round(
        processed_points / processed_frames * total_frames
    )
    estimated_bytes = estimated_points * PLY_DTYPE.itemsize + 256
    estimated_gib = estimated_bytes / 1024**3

    print(
        f"\nEstimated PLY size from {processed_frames} frames: "
        f"{estimated_gib:.2f} GiB ({estimated_points:,} points)"
    )

    if estimated_gib <= LARGE_PLY_GIB:
        return

    answer = input(
        f"Estimated size exceeds {LARGE_PLY_GIB:.2f} GiB. "
        "Continue? [y/N]: "
    ).strip().lower()
    if answer not in {"y", "yes"}:
        print("Cancelled.")
        raise SystemExit(0)


def format_duration(seconds):
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    if not 0 < SAMPLE_RATIO <= 1:
        raise ValueError("SAMPLE_RATIO must be in the range (0, 1]")
    if STORAGE_MODE not in {"memory", "disk"}:
        raise ValueError('STORAGE_MODE must be "memory" or "disk"')
    if ESTIMATE_FRAMES < 1:
        raise ValueError("ESTIMATE_FRAMES must be at least 1")
    if WORKER_COUNT < 1:
        raise ValueError("WORKER_COUNT must be at least 1")

    rgb_files = collect_files(RGB_DIR, RGB_PATTERN, RGB_FRAME_OFFSET, "RGB")
    depth_files = collect_files(
        DEPTH_DIR, DEPTH_PATTERN, DEPTH_FRAME_OFFSET, "depth EXR"
    )
    k_files = collect_files(K_DIR, TXT_PATTERN, K_FRAME_OFFSET, "K")
    pose_files = collect_files(TWC_DIR, TXT_PATTERN, POSE_FRAME_OFFSET, "T_wc")

    frames = sorted(
        set(rgb_files) & set(depth_files) & set(k_files) & set(pose_files)
    )
    if not frames:
        raise ValueError("RGB, depth, K, and T_wc have no matching frames")
    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]

    print(f"Matching frames: {len(frames)}, range [{frames[0]}, {frames[-1]}]")

    print(f"PLY storage mode: {STORAGE_MODE}")
    print(f"Worker threads: {WORKER_COUNT}")

    if STORAGE_MODE == "disk":
        TEMP_PLY_DIR.mkdir(parents=True, exist_ok=True)

    chunk_paths = []
    memory_vertices = []
    pending_points = []
    pending_colors = []
    total_points = 0
    estimate_at_frame = min(ESTIMATE_FRAMES, len(frames))
    start_time = time.perf_counter()

    for index, frame, points, colors, depth in process_frames(
        frames,
        rgb_files,
        depth_files,
        k_files,
        pose_files,
    ):
        if points.size:
            total_points += len(points)

            if STORAGE_MODE == "memory":
                memory_vertices.append(pack_vertices(points, colors))
            else:
                pending_points.append(points)
                pending_colors.append(colors)

        if index == 1:
            valid_depth = depth[
                np.isfinite(depth)
                & (depth >= MIN_DEPTH_METERS)
                & (depth <= MAX_DEPTH_METERS)
            ]
            print(
                f"First frame: {depth.shape[1]} x {depth.shape[0]}, "
                f"sample ratio={SAMPLE_RATIO}"
            )
            if valid_depth.size:
                print(
                    f"Depth: min={valid_depth.min():.3f} m, "
                    f"median={np.median(valid_depth):.3f} m, "
                    f"max={valid_depth.max():.3f} m"
                )

        elapsed_seconds = time.perf_counter() - start_time
        eta_seconds = elapsed_seconds / index * (len(frames) - index)

        print(
            f"\rFrames: {index}/{len(frames)}, frame={frame}, "
            f"points={len(points)}, total={total_points}, "
            f"elapsed={format_duration(elapsed_seconds)}, "
            f"ETA={format_duration(eta_seconds)}",
            end="",
            flush=True,
        )

        if STORAGE_MODE == "memory" and index == estimate_at_frame:
            confirm_estimated_size(total_points, index, len(frames))

        if (
            STORAGE_MODE == "disk"
            and index % FRAMES_PER_CHUNK == 0
            and pending_points
        ):
            chunk_path = TEMP_PLY_DIR / f"chunk_{len(chunk_paths):06d}.ply"
            save_chunk(chunk_path, pending_points, pending_colors)
            chunk_paths.append(chunk_path)
            pending_points.clear()
            pending_colors.clear()

    print()

    if total_points == 0:
        raise ValueError("No valid points were generated")

    OUTPUT_PLY.parent.mkdir(parents=True, exist_ok=True)

    if STORAGE_MODE == "memory":
        write_memory_ply(OUTPUT_PLY, memory_vertices, total_points)
        final_point_count = total_points
    else:
        if pending_points:
            chunk_path = TEMP_PLY_DIR / f"chunk_{len(chunk_paths):06d}.ply"
            save_chunk(chunk_path, pending_points, pending_colors)
            chunk_paths.append(chunk_path)

        final_point_count = merge_ply_files(chunk_paths, OUTPUT_PLY)

        for path in chunk_paths:
            path.unlink()

    print(f"Saved: {OUTPUT_PLY}")
    print(f"Final points: {final_point_count}")
