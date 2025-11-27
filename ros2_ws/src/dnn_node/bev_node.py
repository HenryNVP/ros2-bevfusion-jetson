import sys
import os
from typing import List
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import message_filters
import importlib.util

# --- PATH SETUP ---
# Find the repo root dynamically
script_dir = os.path.dirname(os.path.realpath(__file__))
# We assume structure: src/dnn_node/dnn_node/bev_node.py -> repo is in ros2_ws/Lidar_AI_Solution
# Adjust logic to find the Lidar_AI_Solution folder
REPO_ROOT = "/root/ros2_ws/Lidar_AI_Solution/CUDA-BEVFusion"

# 1. Add Build Library Path
build_path = os.path.join(REPO_ROOT, "build")
if build_path not in sys.path:
    sys.path.append(build_path)

# 2. Potential helper locations (tensor.py was moved in recent drops)
helper_paths = [
    os.path.join(REPO_ROOT, "tool"),
    os.path.join(REPO_ROOT, "src/common"),
]
for path in helper_paths:
    if path not in sys.path:
        sys.path.append(path)

# Import C++ Library
try:
    import libpybev as pybev
except ImportError as e:
    print(f"FATAL: Could not import libpybev.so from {build_path}.")
    raise e

# Import Tensor Helper (check both tool/ and src/common/)
try:
    tensor_mod = None
    for candidate in helper_paths:
        tensor_file = os.path.join(candidate, "tensor.py")
        if os.path.exists(tensor_file):
            tensor_spec = importlib.util.spec_from_file_location("tensor_helper", tensor_file)
            tensor_mod = importlib.util.module_from_spec(tensor_spec)
            tensor_spec.loader.exec_module(tensor_mod)
            load_tensor = tensor_mod.load
            break
    if tensor_mod is None:
        raise FileNotFoundError("tensor.py not found under tool/ or src/common/")
except Exception as e:
    print(f"FATAL: Could not load tensor.py: {e}")
    raise e

class BEVFusionNode(Node):
    CLASS_NAMES = [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
    ]

    def __init__(self):
        super().__init__('bevfusion_node')
        
        # Parameters
        self.declare_parameter("confidence_threshold", 0.01)  # Lowered temporarily to debug low scores
        self.conf_threshold = self.get_parameter("confidence_threshold").value
        
        # Model Config
        model_variant = "resnet50"  # Match the model you built
        model_root = os.path.join(REPO_ROOT, "model", model_variant, "build")
        # Use converted nuScenes data (or example-data for quick start)
        # Option 1: Your converted data (recommended if you have nuScenes)
        converted_data_dir = "/home/student/ros2/ros2_ws/data/bev_sequence"
        # Option 2: Example data (fallback)
        example_data_dir = os.path.join(REPO_ROOT, "example-data")
        # Use converted data if it exists, otherwise fall back to example-data
        if os.path.exists(converted_data_dir) and os.path.exists(os.path.join(converted_data_dir, "camera2lidar.tensor")):
            calib_dir = converted_data_dir
            self.get_logger().info(f"Using converted calibration data: {calib_dir}")
        else:
            calib_dir = example_data_dir
            self.get_logger().info(f"Using example calibration data: {calib_dir}")
        
        self.get_logger().info(f"Loading Model from: {model_root}")
        
        # Initialize Core (FP16)
        self.core = pybev.load_bevfusion(
            os.path.join(model_root, "camera.backbone.plan"),
            os.path.join(model_root, "camera.vtransform.plan"),
            os.path.join(REPO_ROOT, "model", model_variant, "lidar.backbone.xyz.onnx"),
            os.path.join(model_root, "fuser.plan"),
            os.path.join(model_root, "head.bbox.plan"),
            "fp16"
        )
        
        if self.core is None:
             raise RuntimeError("Core Init Failed")
        
        self._load_calibration(calib_dir)

        # Setup ROS
        self.bridge = CvBridge()
        self.subs = []
        topics = [
            '/cam_front/image_raw', '/cam_front_right/image_raw', '/cam_front_left/image_raw',
            '/cam_back/image_raw', '/cam_back_left/image_raw', '/cam_back_right/image_raw'
        ]
        
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        for t in topics:
            self.subs.append(message_filters.Subscriber(self, Image, t, qos_profile=qos))
        
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar_top/points', qos_profile=qos)

        # Sync Policy
        self.ts = message_filters.ApproximateTimeSynchronizer(self.subs + [self.lidar_sub], 10, 0.1)
        self.ts.registerCallback(self._sync_callback)

        self.pub = self.create_publisher(Detection3DArray, '/bevfusion/detections', 10)
        self.get_logger().info("Node Ready! Waiting for synced data...")

    def _load_calibration(self, directory):
        self.get_logger().info(f"Loading calibration from {directory}...")
        try:
            cam2lidar = load_tensor(os.path.join(directory, "camera2lidar.tensor"))
            intrinsics = load_tensor(os.path.join(directory, "camera_intrinsics.tensor"))
            lidar2img = load_tensor(os.path.join(directory, "lidar2image.tensor"))
            img_aug = load_tensor(os.path.join(directory, "img_aug_matrix.tensor"))
            
            # Debug: Log calibration shapes
            self.get_logger().info(f"Calibration loaded - cam2lidar: {cam2lidar.shape}, intrinsics: {intrinsics.shape}, "
                                 f"lidar2img: {lidar2img.shape}, img_aug: {img_aug.shape}")
            
            self.core.update(cam2lidar, intrinsics, lidar2img, img_aug)
            self.get_logger().info("Calibration updated successfully")
        except Exception as e:
            self.get_logger().error(f"Calibration Error: {e}", exc_info=True)
            raise

    def _prepare_images(self, images: List[Image]) -> np.ndarray:
        processed = []
        # Target size should match your logs/model requirement
        # Your logs showed 1600x900 input coming from ReplayNode
        target_w, target_h = 1600, 900 

        for msg in images:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            if cv_image.shape[0] != target_h or cv_image.shape[1] != target_w:
                cv_image = cv2.resize(cv_image, (target_w, target_h))
            
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # CRITICAL: Keep as uint8 (0-255). The C++ library handles normalization on GPU.
            processed.append(np.ascontiguousarray(rgb_image, dtype=np.uint8))

        # Stack: (1, 6, H, W, 3)
        batch = np.stack(processed, axis=0)
        batch = batch[np.newaxis, ...] 
        return np.ascontiguousarray(batch)

    def _prepare_points(self, cloud: PointCloud2) -> np.ndarray:
        # Read x,y,z,intensity
        # We use a manual buffer read for speed and to ensure float32 first
        raw_data = np.frombuffer(cloud.data, dtype=np.float32)
        point_step = cloud.point_step // 4
        num_points = cloud.width * cloud.height
        points = raw_data.reshape(num_points, point_step)
        
        # Extract first 4 columns
        xyz_i = points[:, :4]
        
        # Add padding for 5th column if needed by C++
        padding = np.zeros((num_points, 1), dtype=np.float32)
        final_points = np.hstack((xyz_i, padding))
        
        # CRITICAL: Cast to Float16 for the FP16 model
        return np.ascontiguousarray(final_points.astype(np.float16))

    def _sync_callback(self, *msgs):
        # msgs[0-5] = Images, msgs[6] = Lidar
        img_msgs = msgs[:6]
        lidar_msg = msgs[6]
        
        try:
            # 1. Prepare Inputs
            images = self._prepare_images(img_msgs)
            points = self._prepare_points(lidar_msg)
            
            if points.size == 0:
                self.get_logger().warn("Empty point cloud, skipping")
                return

            # Debug: Log input shapes
            self.get_logger().debug(f"Images shape: {images.shape}, dtype: {images.dtype}, range: [{images.min()}, {images.max()}]")
            self.get_logger().debug(f"Points shape: {points.shape}, dtype: {points.dtype}, num_points: {points.shape[0]}")

            # 2. Inference
            # forward(images, points, with_normalization=True, with_dlpack=False)
            detections = self.core.forward(images, points, True, False)
            
            # Debug: Log raw detections
            if len(detections) > 0:
                # Check for NaN/inf in detections
                nan_count = np.sum(~np.isfinite(detections))
                valid_scores = detections[:, 10] if detections.shape[1] > 10 else []
                if len(valid_scores) > 0:
                    finite_scores = valid_scores[np.isfinite(valid_scores)]
                    if len(finite_scores) > 0:
                        # Score distribution analysis
                        above_01 = np.sum(finite_scores >= 0.01)
                        above_05 = np.sum(finite_scores >= 0.05)
                        above_10 = np.sum(finite_scores >= 0.10)
                        above_25 = np.sum(finite_scores >= 0.25)
                        score_stats = f"score range: [{np.min(finite_scores):.3f}, {np.max(finite_scores):.3f}], valid: {len(finite_scores)}/{len(valid_scores)}, above thresholds: 0.01={above_01}, 0.05={above_05}, 0.10={above_10}, 0.25={above_25}"
                    else:
                        score_stats = "all NaN"
                else:
                    score_stats = "no scores"
                self.get_logger().info(f"Raw detections: {len(detections)} total, {nan_count} NaN/inf values, {score_stats}")
                if len(detections) > 0 and detections.shape[1] >= 11:
                    # Log top 5 detections by score
                    sorted_indices = np.argsort(detections[:, 10])[::-1] if len(valid_scores) > 0 else range(min(5, len(detections)))
                    for idx, i in enumerate(sorted_indices[:5]):
                        det = detections[i]
                        self.get_logger().info(f"  Top[{idx}]: xyz=({det[0]:.2f},{det[1]:.2f},{det[2]:.2f}), size=({det[3]:.2f},{det[4]:.2f},{det[5]:.2f}), score={det[10]:.3f}, class={int(det[9])}")
            else:
                self.get_logger().info("No detections returned from model")
            
            # 3. Publish
            self._publish_detections(detections, lidar_msg.header)
            
        except Exception as e:
            self.get_logger().error(f"Inference Fail: {e}", exc_info=True)

    def _publish_detections(self, detections, header):
        out_msg = Detection3DArray()
        out_msg.header = header
        
        if len(detections) == 0:
            self.pub.publish(out_msg)
            return
        
        valid_count = 0
        for det in detections:
            # Handle NaN/inf values
            if not isinstance(det, np.ndarray) or len(det) < 11:
                continue
                
            score = float(det[10])
            
            # Skip invalid scores (NaN, inf, or below threshold)
            if not np.isfinite(score) or score < self.conf_threshold:
                continue
            
            # Check for NaN in position/size
            if not all(np.isfinite([det[0], det[1], det[2], det[3], det[4], det[5]])):
                continue
            
            ros_det = Detection3D()
            ros_det.header = header
            
            # BBox: x, y, z, w, l, h, rot
            ros_det.bbox.center.position.x = float(det[0])
            ros_det.bbox.center.position.y = float(det[1])
            ros_det.bbox.center.position.z = float(det[2])
            ros_det.bbox.size.x = float(det[4])  # width
            ros_det.bbox.size.y = float(det[3])  # length
            ros_det.bbox.size.z = float(det[5])  # height
            
            yaw = float(det[6])
            if np.isfinite(yaw):
                ros_det.bbox.center.orientation.z = np.sin(yaw / 2.0)
                ros_det.bbox.center.orientation.w = np.cos(yaw / 2.0)
            else:
                ros_det.bbox.center.orientation.w = 1.0  # Default orientation
            
            # Class
            class_id = int(det[9]) if np.isfinite(det[9]) else 0
            if 0 <= class_id < len(self.CLASS_NAMES):
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = self.CLASS_NAMES[class_id]
                hyp.hypothesis.score = score
                ros_det.results.append(hyp)
            else:
                # Unknown class, skip
                continue
                
            out_msg.detections.append(ros_det)
            valid_count += 1
            
        self.get_logger().info(f"Published {valid_count}/{len(detections)} valid detections (threshold: {self.conf_threshold})")
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BEVFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()