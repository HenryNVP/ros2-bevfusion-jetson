# ROS2 BEV-Fusion

ROS2 nodes for running BEV-Fusion 3D object detection with CUDA acceleration.

## Project Description

This project provides a ROS2 implementation of **BEV-Fusion**, a state-of-the-art 3D object detection system that fuses camera and LiDAR sensor data. BEV-Fusion performs bird's-eye-view (BEV) feature extraction from multi-camera images and LiDAR point clouds, then combines these features to detect and localize objects in 3D space.

The ROS2 package is platform-agnostic and can run on systems with CUDA support. It requires meeting the [CUDA-BEVFusion requirements](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) (CUDA toolkit, TensorRT, etc.). The setup instructions provided are optimized for NVIDIA Jetson devices, but can be adapted for other CUDA-capable systems.

**Tested Platforms:**
- Jetson Orin Nano (ARM64, JetPack)
- NVIDIA Tesla P100 (x86_64, CUDA)

### What It Achieves

- **Real-time 3D Object Detection**: Detects 10 object classes (cars, trucks, buses, pedestrians, motorcycles, bicycles, etc.) in real-time using CUDA-accelerated inference
- **Multi-Sensor Fusion**: Fuses data from 6 cameras and 1 LiDAR sensor for robust perception
- **Bird's-Eye-View Processing**: Converts camera images to BEV representation for unified feature space with LiDAR
- **ROS2 Integration**: Publishes detections as standard ROS2 messages (`vision_msgs/Detection3DArray`) for integration with other ROS2 systems
- **Performance Metrics**: Publishes inference timing and FPS metrics for performance monitoring
- **NuScenes Dataset Support**: Replays NuScenes dataset for testing and evaluation

The system processes synchronized camera images and LiDAR point clouds, runs inference using TensorRT-optimized models, and publishes 3D bounding boxes with class labels and confidence scores.

## Quick Start

> **Note**: The following setup instructions are optimized for NVIDIA Jetson devices using Docker. For other CUDA-capable systems, adapt the Docker image and paths accordingly.

### 1. Launch Docker Container

> **Note**: ROS2 containers for Jetson can be found at [jetson-containers](https://github.com/dusty-nv/jetson-containers).

```bash
docker run -it --rm --runtime nvidia --network host \
    -v ~/ros2-bevfusion-jetson:/workspace \
    --name <container_name> \
    <ros2_docker_image>
```

### 2. Install Dependencies (Inside Container)

```bash
apt-get update
apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken -y
apt-get install -y libprotobuf-dev protobuf-compiler
pip3 install onnx onnx-graphsurgeon pycuda --index-url https://pypi.org/simple
```

### 3. Setup CUDA-BEVFusion

```bash
cd /workspace
source /opt/ros/humble/install/setup.bash

# Clone Lidar_AI_Solution if not already present
git clone --recursive https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution

# Setup environment
cd Lidar_AI_Solution/CUDA-BEVFusion
# Need to configure environment.sh
source tool/environment.sh

# Regenerate protobuf files
bash src/onnx/make_pb.sh
```

### 4. Build CUDA-BEVFusion Library

```bash
cd /workspace/Lidar_AI_Solution/CUDA-BEVFusion
rm -rf build && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DProtobuf_INCLUDE_DIR=/usr/include \
  -DProtobuf_LIBRARY=/usr/lib/aarch64-linux-gnu/libprotobuf.so \
  -DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc

# For x86_64 systems, adjust the Protobuf library path:
# -DProtobuf_LIBRARY=/usr/lib/x86_64-linux-gnu/libprotobuf.so

make -j$(nproc)
```

### 5. Build TensorRT Model Engines

```bash
cd /workspace/Lidar_AI_Solution/CUDA-BEVFusion
bash tool/build_trt_engine.sh
```

This creates `.plan` files in `model/<variant>/build/` (e.g., `model/resnet50int8/build/`).

### 6. Setup Environment Variables

Source the environment script before running nodes:

```bash
cd /workspace
source env.sh
```

> **Note**: Edit `env.sh` and adjust the `libspconv` library path according to your system architecture

### 7. Build ROS2 Package

```bash
cd /workspace/ros2_ws
colcon build --packages-select bevfusion_pkg --symlink-install
source install/setup.bash
```

### 8. Run Nodes

**Terminal 1 - BEV-Fusion Node:**
```bash
cd /workspace/ros2_ws
source install/setup.bash
ros2 run bevfusion_pkg bev_node
```

**Terminal 2 - Replay Node:**
```bash
cd /workspace/ros2_ws
source install/setup.bash
ros2 run bevfusion_pkg replay_node
```

## Data Requirements

- **NuScenes dataset**: Place dataset at `/workspace/data/nuscenes/`
- **Calibration data**: Uses `example-data` from CUDA-BEVFusion repository (automatically detected)

## Topics

### Subscribed (bev_node)
- `/cam_front/image_raw` - Front camera image
- `/cam_front_right/image_raw` - Front-right camera image
- `/cam_front_left/image_raw` - Front-left camera image
- `/cam_back/image_raw` - Back camera image
- `/cam_back_left/image_raw` - Back-left camera image
- `/cam_back_right/image_raw` - Back-right camera image
- `/lidar_top/points` - LiDAR point cloud

### Published (bev_node)
- `/bevfusion/detections` - `vision_msgs/Detection3DArray` - 3D object detections
- `/bevfusion/metrics` - `std_msgs/Float64MultiArray` - Performance metrics [prep_time_ms, inference_time_ms, publish_time_ms, total_time_ms, fps]

> **Note**: You can observe these topics using ROS2 tools:
> - `ros2 topic echo /bevfusion/detections` - View detections in real-time
> - `ros2 topic echo /bevfusion/metrics` - View performance metrics
> - `ros2 bag record` - Record topics for offline analysi
## Project Structure

```
ros2-bevfusion-jetson/
├── ros2_ws/
│   └── src/
│       └── bevfusion_pkg/
│           └── bevfusion_pkg/
│               ├── bev_node.py      # BEV-Fusion inference node
│               └── replay_node.py   # NuScenes data replay node
├── data/
│   └── nuscenes/                    # NuScenes dataset
├── env.sh                           # Environment variables setup
└── Lidar_AI_Solution/               # CUDA-BEVFusion repository
    └── CUDA-BEVFusion/
```

## License

This project is based on [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) from NVIDIA AI-IOT.

