# LongRecon Benchmark: A Large-Scale Benchmark for Continuous Long-Sequence 3D Reconstruction

Python utilities for inspecting Unreal Engine EXR depth maps and fusing RGB, Pose and Depth frames into a point cloud `.ply` file.

## Installation

```bash
conda create -n longrecon python=3.11 -y
conda activate longrecon
python -m pip install -r requirements.txt
```

## Data layout

```text
DATA_DIR/
├── rgb/                # image_0000.png, ...
├── depth/              # depth_0000.exr, ...
└── pose/
    ├── K/              # 000000.txt, ...
    └── T_wc/           # 000000.txt, ...
```

Set `DATA_DIR` and other parameters at the top of `fuse_pcd.py` before running.

## Usage

Inspect one EXR file:

```bash
python readexr.py "path/to/depth_0000.exr"
```

Fuse all matching frames:

```bash
python -u fuse_pcd.py
```

The point cloud is saved to the path specified by `OUTPUT_PLY`.

Export C2W camera poses as red frustums:

```bash
python export_camera_frustums.py
```
