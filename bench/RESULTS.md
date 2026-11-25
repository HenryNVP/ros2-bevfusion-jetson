# Benchmark results

CUDA-BEVFusion (ResNet50, camera + LiDAR) wrapped in ROS 2, evaluated on nuScenes v1.0-mini.

**Platform:** Jetson Orin Nano Engineering Reference Developer Kit Super (8 GB), L4T R35.6.4,
nvpmodel 25W, `jetson_clocks` (GPU 918 MHz). Container `dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1`,
CycloneDDS.

| Config | Engines | LiDAR sparse conv |
|---|---|---|
| **FP16 (baseline)** | `model/resnet50` TensorRT FP16 | FP16 |
| **INT8 (quantized)** | `model/resnet50int8` TensorRT PTQ INT8 | INT8 |

## 1. Model-only (offline, `bench/nusc_bench.py`)

mini_val (81 samples) × 3 passes after 20 warmup; full model input (per-sample calibration,
10 LiDAR sweeps, ~246k points/frame). Power from a 30 s continuous-inference pass.

| Metric | FP16 | INT8 | Δ |
|---|---|---|---|
| Latency p50 / p99 | 150.6 / 162.8 ms | 115.7 / 124.8 ms | **−23%** |
| Throughput | 6.6 FPS | 8.6 FPS | +30% |
| mAP / NDS (official, 10 classes) | 0.577 / 0.582 | 0.574 / 0.581 | −0.3 / −0.1 pt |
| mAP, 7 classes present in mini_val | 0.82 | 0.82 | ≈0 |
| Board power (VDD_IN) | 19.1 W | 17.5 W | −8% |
| Energy / frame | 3.11 J | 2.22 J | **−29%** |
| GPU utilisation | 96% | 96% | |

- mini_val has no trailer / construction_vehicle / barrier boxes, so those classes score 0 and cap
  official mAP at 0.7. Per-class AP (INT8): car 0.92, pedestrian 0.93, bus 0.99, truck 0.80,
  motorcycle 0.73, traffic cone 0.75, bicycle 0.61.
- Reference (NVIDIA, full val 6019 samples, AGX Orin): FP16 67.89 mAP / 70.98 NDS (18 FPS);
  PTQ INT8 67.66 / 70.81 (25 FPS).

## 2. ROS 2 end-to-end (`bench/run_ros_e2e.sh`)

`replay_node` → `bev_node` → `detection_logger_node`, all 404 mini keyframes.
E2E latency = detection received − frame stamp (stamped right before publish). First 5 frames excluded.

| Metric | FP16 | INT8 | Δ |
|---|---|---|---|
| E2E latency p50 / p99 (1 Hz input) | 210 / 240 ms | 176 / 209 ms | **−16%** |
| Inference in node, p50 | 117 ms | 89 ms | −24% |
| Frames delivered | 404/404 (0% drop) | 404/404 (0% drop) | |
| Max throughput* | 3.1 FPS | 3.1 FPS | |
| Board power at max throughput | 13.2 W | 11.9 W | −10% |
| GPU utilisation at max throughput | 33% | 24% | |
| Peak system RAM | 5.2 GB | 4.7 GB | −0.4 GB |

\* Bounded by `replay_node` (JPEG decode + 26 MB/frame publish), not by inference.

Notes:
- The node currently feeds a single LiDAR sweep (~35k points) and fixed `example-data`
  calibration, so in-node inference is faster than the offline harness and accuracy above is
  from the offline harness on the same engines, not from the ROS output.
- Per-frame stage breakdown (INT8, p50): stamp→callback (transport + sync) ~42 ms,
  preprocess ~12 ms, inference ~89 ms, publish ~14 ms.

## Required setup: DDS socket buffer

Without this, ~50% of frames are dropped: each 4.3 MB image is fragmented into thousands of UDP
datagrams and the default 208 KB kernel receive buffer overflows (visible as `RcvbufErrors` in
`/proc/net/snmp`).

```bash
sudo sysctl -w net.core.rmem_max=2147483647            # persist: /etc/sysctl.d/60-dds.conf
export CYCLONEDDS_URI=file:///workspace/bench/cyclonedds.xml   # 64 MB SocketReceiveBufferSize
```

| | Drops (1 Hz) | Throughput (max) | Peak RAM |
|---|---|---|---|
| Default | 50% | 1.6 FPS | 7.1 GB |
| Tuned | 0% | 3.1 FPS | 4.4 GB |

## Demo

`bench/render_demo.py` renders a run's logged detections (6 cameras + LiDAR BEV, with the run's
measured metrics). INT8 1 Hz run, scene-0103 (40 frames): [docs/demo.mp4](../docs/demo.mp4)

## Reproduce (inside the container)

```bash
. bench/env_bench.sh
python3 bench/nusc_bench.py --model resnet50     --precision fp16 --split mini_val
python3 bench/nusc_bench.py --model resnet50int8 --precision int8 --split mini_val
bash bench/run_ros_e2e.sh 1  fp16 resnet50     fp16
SAVE_DETS=true bash bench/run_ros_e2e.sh 1 int8 resnet50int8 int8
bash bench/run_ros_e2e.sh 10 int8 resnet50int8 int8   # max rate
python3 bench/render_demo.py logs/bevfusion_detections/<run_dir> --scene scene-0103
```
