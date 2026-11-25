# Benchmark results — baseline (2026-11-25)

**Platform:** Jetson Orin Nano Engineering Reference Developer Kit Super (8 GB), L4T R35.6.4,
nvpmodel 25W, `jetson_clocks` (GPU 918 MHz). Container `dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1`.
**Model:** CUDA-BEVFusion ResNet50-PTQ (TensorRT INT8 camera/fuser/head). **Data:** nuScenes v1.0-mini.

## 1. Model-only (offline harness `bench/nusc_bench.py`, mini_val, 81 samples × 3 passes, 20 warmup)

Full input: per-sample calibration, 10 LiDAR sweeps (~246k points/frame avg).

| LiDAR sparse-conv | p50 ms | p90 ms | p99 ms | FPS (p50) | mAP | NDS | mAP (7 present classes) |
|---|---|---|---|---|---|---|---|
| FP16 (what bev_node uses) | 129.6 | 135.8 | 141.4 | 7.7 | 0.575 | 0.581 | 0.822 |
| **INT8** | **115.3** | 120.3 | 124.4 | **8.7** | 0.573 | 0.581 | 0.819 |

- INT8 sparse conv: **−11% latency, no accuracy change** (within mini noise).
- mini_val has 0 GT boxes for trailer / construction_vehicle / barrier → those score 0 by
  construction, capping official mAP at 0.7. Per-class AP: car 0.92, ped 0.93, bus 0.99, truck 0.80.
- Reference (NVIDIA, full val, AGX Orin): FP16 67.89 mAP / 70.98 NDS; PTQ INT8 67.66 / 70.81.
- Harness process peak RSS: ~3.2 GB.

## 2. ROS 2 end-to-end, baseline (`bench/run_ros_e2e.sh`, 404 keyframes)

E2E = detection received − frame stamp (stamp set by replay immediately before publish).
First 5 received frames excluded.

| Requested rate | Offered (actual) | Received | Drop rate | Throughput | E2E p50 | E2E p99 | Peak RAM / swap |
|---|---|---|---|---|---|---|---|
| 1 Hz | 1.00 Hz | 202/404 | **50%** | 0.51 FPS | 192 ms | 244 ms | 7.1 GB / 1.4 GB |
| 10 Hz (max) | 2.97 Hz | 215/404 | **47%** | 1.59 FPS | 182 ms | 214 ms | 7.0 GB / 1.3 GB |

Per-frame stage breakdown (10 Hz run, p50): wait (transport+sync) 40 ms · prep 12 ms ·
inference 97 ms · publish 14 ms · callback total 124 ms.

Peak RSS: bev_node 5.6 GB, replay 0.2 GB, logger 0.05 GB.

## Findings

1. **~50% frame loss independent of rate** (1 Hz and 3 Hz both lose half), while bev_node's
   callback takes 124 ms, i.e. compute could sustain ~8 FPS. Drops are irregular and happen
   before the callback → transport (best-effort DDS, ~4 MB images) or 7-way sync, not compute.
2. **Memory pressure:** bev_node 5.6 GB RSS vs 3.2 GB for the same model in the harness;
   board hits 7 GB + swap. Sync queue (200) and subscriber depth (50) × 7 topics of 4 MB images
   are the prime suspects.
3. **Replay is a bottleneck:** cannot exceed ~3 Hz (single-threaded JPEG decode + 26 MB/frame).
4. **ROS input is not the model's training input:** single sweep (~35k pts) + fixed
   `example-data` calibration. ROS inference (97 ms) is faster than the harness (130 ms) only
   because it processes ~7× fewer points. Accuracy of the ROS path is not yet representative.
5. Free win: bev_node uses FP16 sparse conv; INT8 is 11% faster at equal accuracy.

## Next (optimizations, each re-measured with the same scripts)

- Fix drops: reliable QoS / shared-memory transport or intra-process composition; shrink queues.
- Per-frame calibration + 10-sweep accumulation in the ROS path; score ROS output with the harness eval.
- Switch LiDAR backbone to INT8.
- Faster replay (preloaded / threaded decode) so the source isn't the ceiling.
