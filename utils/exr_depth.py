"""Read a 2D depth channel from an EXR file."""

import os
from pathlib import Path
import struct

import numpy as np

try:
    import OpenEXR
except ImportError:
    OpenEXR = None

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


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
    if "worlddepth" in full_name:
        score += 100
    if "scene_depth" in full_name or "scene.depth" in full_name:
        score += 90
    return score


def find_channels(exr_file):
    channels = []

    for part_index, part in enumerate(exr_file.parts):
        part_name = str(part.name())

        for channel_name, channel in part.channels.items():
            pixels = np.asarray(channel.pixels)
            if pixels.ndim != 2 or pixels.dtype == object:
                continue

            channels.append(
                {
                    "part_index": part_index,
                    "part_name": part_name,
                    "channel_name": channel_name,
                    "pixels": pixels,
                    "score": channel_score(part_name, channel_name),
                }
            )

    return sorted(channels, key=lambda item: item["score"], reverse=True)


def read_depth_channel(path):
    path = Path(path)

    if not hasattr(OpenEXR, "File"):
        return read_simple_exr(path)

    with OpenEXR.File(str(path), separate_channels=True) as exr_file:
        if not exr_file.parts:
            raise ValueError(f"EXR has no parts: {path}")

        channels = find_channels(exr_file)
        if not channels:
            raise ValueError(f"EXR has no usable 2D channels: {path}")

        depth = channels[0]["pixels"].astype(np.float64, copy=True)

    return depth, channels


def read_simple_exr(path):
    # OpenCV can read standard single-part RGB(A) EXRs but may silently
    # ignore custom channels. Confirm the layout before using that decoder.
    with path.open("rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"Truncated EXR header: {path}")
        magic, version = struct.unpack("<II", header)
        if magic != 20000630:
            raise ValueError(f"Invalid OpenEXR file: {path}")
        if version & 0x1800:
            raise ValueError(f"Multipart or deep EXR requires OpenEXR 3.3+: {path}")

        def read_name():
            chars = bytearray()
            for _ in range(256):
                char = file.read(1)
                if not char:
                    break
                if char == b"\0":
                    return bytes(chars)
                chars.extend(char)
            raise ValueError(f"Invalid EXR header: {path}")

        names = []
        while True:
            attr = read_name()
            if not attr:
                break
            attr_type = read_name()
            size_bytes = file.read(4)
            if len(size_bytes) != 4:
                raise ValueError(f"Truncated EXR header: {path}")
            size, = struct.unpack("<I", size_bytes)
            if attr == b"channels":
                if attr_type != b"chlist" or not 1 <= size <= 65536:
                    raise ValueError(f"Invalid EXR channel list: {path}")
                payload = file.read(size)
                if len(payload) != size:
                    raise ValueError(f"Truncated EXR channel list: {path}")
                offset = 0
                while offset < size and payload[offset]:
                    end = payload.find(b"\0", offset)
                    if end < 0 or end + 17 >= size:
                        raise ValueError(f"Invalid EXR channel list: {path}")
                    names.append(payload[offset:end])
                    offset = end + 17
                break
            file.seek(size, 1)

    if sorted(names) not in ([b"B", b"G", b"R"], [b"A", b"B", b"G", b"R"]):
        raise ValueError(f"Named or non-RGB depth channels require OpenEXR 3.3+: {path}")

    import cv2

    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(f"OpenCV could not decode the RGB(A) depth EXR: {path}")
    names = "BGRA"[:pixels.shape[2]]
    channels = [
        {"part_index": 0, "part_name": "OpenCV", "channel_name": name,
         "pixels": pixels[:, :, i], "score": channel_score("", name)}
        for i, name in enumerate(names)
    ]
    channels.sort(key=lambda item: item["score"], reverse=True)
    return np.ascontiguousarray(channels[0]["pixels"], dtype=np.float64), channels


def describe_channel(channel):
    return (
        f"part={channel['part_index']}, "
        f"part_name={channel['part_name']!r}, "
        f"channel={channel['channel_name']!r}, "
        f"shape={channel['pixels'].shape}, "
        f"dtype={channel['pixels'].dtype}, "
        f"score={channel['score']}"
    )
