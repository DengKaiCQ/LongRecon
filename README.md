<p align="center">
<h1 align="center">LongRecon Benchmark: A Large-Scale Benchmark for Continuous Long-Sequence 3D Reconstruction</h1>
</p>
<strong><h4 align="center"><a href="https://huggingface.co/datasets/DengKaiCQ/LongRecon" target="_blank">Huggingface (Uploading)</a></h4></strong>
</strong>

It is a dense 3D recon benchmark rendered by Unreal Engine, will contains large scale scenarios for 10k images each. But it is currently under construction. 

I am working on it! :)

Utilities for inspecting Unreal Engine EXR depth maps and fusing multi-camera RGB, depth and camera poses into a point cloud.

## Installation

```bash
conda create -n longrecon python=3.11 -y
conda activate longrecon
python -m pip install -r requirements.txt
```

## Data layout

```text
<Scenario Name>/
|-- depth_exr/
|   `-- depth_000000.exr, ...
|-- depth_png/
|   `-- depth_000000.png, ...
|-- image_png/
|   `-- image_000000.png, ...
|-- image_jpg/
|   `-- image_000000.jpg, ...
`-- pose/
    |-- K/          # 000000.txt, ...
    |-- T_wc/       # 000000.txt, ...
    `-- T_cw/       # 000000.txt, ...
```

We have provided original precision rendering data (with **depths** in `exr` format and **images** in `png` format). The original precision data is the native rendering result of the UE engine. However, we also know that the current benchmark size may exceed some training datasets. To alleviate the storage pressure, we have also provided compressed format data (**depths** in `png` format and **images** in `jpg` format). Generally speaking, the compressed format data is 1/4 the size of the original precision data.


For depth map, the 16-bit depth PNG stores depth in centimeters with up to 0.5 cm rounding error, but it is quite acceptable for outdoor scenarios, and can reduce the storage pressure by 75%.

The EXR depth format scale ( $\times 200$  ) converts normalized UE depth to meters, whereas the PNG depth format scale ( $\div 100$ ) converts stored centimeters back to meters. Depth PNGs look dark because normal scene depths occupy only a small part of the full 16-bit range, and this creates a less depth information lost.

*Why don't we use inverse depth to PNGs depth? Cause inverse depth greatly improves near-field precision but rapidly loses accuracy at long distances, making linear centimeter depth more suitable for large outdoor scenes.*

## Usage

Here is a quick example for converting them to pointcloud.

If you are using `exr` Depth format, try using

```bash
python -u utils/fuse_pcd.py
```

The streaming pipeline writes temporary PLY chunks and merges them into the path configured by `OUTPUT_PLY`. And the number of pointcloud will be downsample $\times50 $~$100$ due to large amount of pointcloud.

You can also convert just a few frames of depth maps and images into dense, non-downsampled point clouds to examine scene details, by using

```bash
python -u utils\fuse_selected_frames.py
```

Export C2W camera poses as red wireframe frustums

```bash
python utils/export_camera_frustums.py
```

If you would like to use the compressed format (`png` Depths and `jpg` Images), try using

```bash
python -u utils\png_depth\fuse_pcd_from_png.py
python -u utils\png_depth\fuse_selected_frames_png_depth.py
```

## Example scenario

You can try by using the example scenario ([link](https://huggingface.co/datasets/DengKaiCQ/LongRecon/tree/main/LongRecon-Example-Ruins)) to quickly get started and familiarize yourself with the benchmark's data format. It is a short sequence of 2500 frames in a small scene. The example scenario is about 7.5 GB for original precision (4.04 GB for RGB in `png` and 3.50 GB for Depth in `exr`) or 2.21GB for compressed format (0.99 GB for RGB in `jpg` and 1.22 GB for Depth in `png`).

View the `.ply` file using [Meshlab](https://www.meshlab.net/) as below (pointcloud form front 3 views)

![ExampleRender](./assets/example-ply-1.jpg)
![ExampleRender](./assets/example-ply-3.jpg)
![ExampleRender](./assets/example-ply-2.jpg)
![ExampleRender](./assets/example-ply-pose.jpg)

And the actual rendered example scene would be like

![ExampleRender](./assets/render-1.jpg)
![ExampleRender](./assets/render-2.jpg)
![ExampleRender](./assets/topdown-view-example.jpg)