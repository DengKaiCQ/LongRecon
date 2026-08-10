"""Inspect an EXR depth channel."""

import argparse
from pathlib import Path

import numpy as np

from exr_depth import describe_channel, read_depth_channel


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXR = PROJECT_DIR / "RuinsLong" / "depth" / "depth_0000.exr"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_EXR)
    path = parser.parse_args().path

    depth, channels = read_depth_channel(path)

    print(f"File: {path}")
    print(f"Selected: {describe_channel(channels[0])}")
    print("Candidates:")
    for channel in channels:
        print(f"  {describe_channel(channel)}")

    finite_depth = depth[np.isfinite(depth)]
    print(f"Shape: {depth.shape}") # depth.shape: (height, width)

    if finite_depth.size == 0:
        print("The depth channel has no finite values.")
    else:
        print(
            f"min={finite_depth.min():.6g}, "
            f"median={np.median(finite_depth):.6g}, "
            f"mean={finite_depth.mean():.6g}, "
            f"max={finite_depth.max():.6g}"
        )

'''
Example output:

File: \depth\depth_0000.exr
Selected: part=0, part_name='', channel='R', shape=(720, 1280), dtype=float16, score=100
Candidates:
  part=0, part_name='', channel='R', shape=(720, 1280), dtype=float16, score=100
  part=0, part_name='', channel='A', shape=(720, 1280), dtype=float16, score=0
  part=0, part_name='', channel='B', shape=(720, 1280), dtype=float16, score=0
  part=0, part_name='', channel='G', shape=(720, 1280), dtype=float16, score=0
Shape: (720, 1280)
min=0.019455, median=0.059082, mean=15240.1, max=65504

'''