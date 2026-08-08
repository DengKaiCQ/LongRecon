"""Read a 2D depth channel from an EXR file."""

from pathlib import Path

import numpy as np
import OpenEXR


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

    with OpenEXR.File(str(path), separate_channels=True) as exr_file:
        if not exr_file.parts:
            raise ValueError(f"EXR has no parts: {path}")

        channels = find_channels(exr_file)
        if not channels:
            raise ValueError(f"EXR has no usable 2D channels: {path}")

        depth = channels[0]["pixels"].astype(np.float64, copy=True)

    return depth, channels


def describe_channel(channel):
    return (
        f"part={channel['part_index']}, "
        f"part_name={channel['part_name']!r}, "
        f"channel={channel['channel_name']!r}, "
        f"shape={channel['pixels'].shape}, "
        f"dtype={channel['pixels'].dtype}, "
        f"score={channel['score']}"
    )
