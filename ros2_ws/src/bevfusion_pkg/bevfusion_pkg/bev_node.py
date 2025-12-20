import sys
import os
import time
import traceback
from typing import List
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
import message_filters
import importlib.util

# --- PATH SETUP ---
# Resolve paths relative to this ROS2 workspace
script_dir = os.path.dirname(os.path.realpath(__file__))

# Expected directory layout:
#   /home/<user>/ros2-bevfusion-jetson
#       ├── Lidar_AI_Solution/CUDA-BEVFusion
#       └── ros2_ws/src/bevfusion_pkg/bevfusion_pkg/bev_node.py  (this file)
#
# Walk up to project root containing Lidar_AI_Solution
WORKSPACE_ROOT = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
DEFAULT_REPO_ROOT = os.path.join(WORKSPACE_ROOT, "Lidar_AI_Solution", "CUDA-BEVFusion")

# Allow overriding repository root with environment variable
REPO_ROOT = os.environ.get("CUDA_BEVFUSION_ROOT", DEFAULT_REPO_ROOT)

if not os.path.isdir(REPO_ROOT):
    raise RuntimeError(
        f"CUDA-BEVFusion repo not found at '{REPO_ROOT}'. "
        f"Set CUDA_BEVFUSION_ROOT env var if your layout is different."
    )

# Add build directory to Python path for libpybev.so
build_path = os.path.join(REPO_ROOT, "build")
if build_path not in sys.path:
    sys.path.append(build_path)

# Add helper directories for tensor.py (may be in tool/ or src/common/)
helper_paths = [
    os.path.join(REPO_ROOT, "tool"),
    os.path.join(REPO_ROOT, "src/common"),
]
for path in helper_paths:
    if path not in sys.path:
        sys.path.append(path)

# Import C++ BEV-Fusion library
try:
    import libpybev as pybev
except ImportError as e:
    print(f"FATAL: Could not import libpybev.so from {build_path}.")
    raise e

# Import tensor loading helper (searches tool/ and src/common/)
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
        self.declare_parameter("confidence_threshold", 0.01)
        self.conf_threshold = self.get_parameter("confidence_threshold").value
        
        # Initialize pre-allocated image buffers for zero-copy processing
        self._init_buffers()
        
        # Model configuration
        model_variant = "resnet50int8"
        model_root = os.path.join(REPO_ROOT, "model", model_variant, "build")

        # Load calibration data from example-data directory
        # Contains camera intrinsics, camera-to-lidar transforms, and augmentation matrices
        calib_dir = os.path.join(REPO_ROOT, "example-data")
        self.get_logger().info(f"Using calibration data from: {calib_dir}")
        
        self.get_logger().info(f"Loading Model from: {model_root}")
        
        # Initialize BEV-Fusion model core
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
        
        # Print model architecture information
        self.core.print()
        
        # Enable C++ timing if available (optional, requires libpybev with timer support)
        try:
            self.core.set_timer(True)
            self.get_logger().info("C++ timing enabled")
        except AttributeError:
            pass  # C++ timing not available, using Python timing only
        
        self._load_calibration(calib_dir)

        # Setup ROS2 subscribers for synchronized camera and lidar data
        self.bridge = CvBridge()
        self.subs = []
        topics = [
            '/cam_front/image_raw', '/cam_front_right/image_raw', '/cam_front_left/image_raw',
            '/cam_back/image_raw', '/cam_back_left/image_raw', '/cam_back_right/image_raw'
        ]
        
        # QoS profile: large queue depth to buffer messages during processing delays
        qos = QoSProfile(depth=50, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        for t in topics:
            self.subs.append(message_filters.Subscriber(self, Image, t, qos_profile=qos))
        
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar_top/points', qos_profile=qos)

        # Synchronize 6 camera images + 1 lidar point cloud by timestamp
        # Queue size 200: buffers messages during processing delays
        # Time tolerance 0.5s: allows for slight timing differences between message arrivals
        self.ts = message_filters.ApproximateTimeSynchronizer(self.subs + [self.lidar_sub], 200, 0.5)
        self.ts.registerCallback(self._sync_callback)

        self.pub = self.create_publisher(Detection3DArray, '/bevfusion/detections', 10)
        self.metrics_pub = self.create_publisher(Float64MultiArray, '/bevfusion/metrics', 10)
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
            self.get_logger().error(f"Calibration Error: {e}\n{traceback.format_exc()}")
            raise

    def _init_buffers(self):
        """Pre-allocate image buffer for zero-copy processing.
        
        Creates a single contiguous memory block for all 6 camera images.
        Shape: (1, 6, 900, 1600, 3) - batch=1, cameras=6, height=900, width=1600, channels=3
        """
        self.image_buffer = np.zeros((1, 6, 900, 1600, 3), dtype=np.uint8)
        # Create memory views for each camera (no copy, points to same buffer)
        self.cam_views = [self.image_buffer[0, i, ...] for i in range(6)]

    def _prepare_images(self, images: List[Image]) -> np.ndarray:
        """Convert ROS Image messages to numpy array format expected by model.
        
        Args:
            images: List of 6 ROS Image messages (one per camera)
            
        Returns:
            Pre-allocated buffer with shape (1, 6, 900, 1600, 3) in RGB format
        """
        target_w, target_h = 1600, 900 

        for i, msg in enumerate(images):
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Resize if needed (replay_node can pre-resize to save time)
            if cv_image.shape[0] != target_h or cv_image.shape[1] != target_w:
                temp = cv2.resize(cv_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                cv2.cvtColor(temp, cv2.COLOR_BGR2RGB, dst=self.cam_views[i])
            else:
                # Already correct size, just convert BGR to RGB
                cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, dst=self.cam_views[i])

        return self.image_buffer

    def _prepare_points(self, cloud: PointCloud2) -> np.ndarray:
        """Extract point cloud data from ROS message and convert to model format.
        
        Args:
            cloud: ROS PointCloud2 message with x, y, z, intensity fields
            
        Returns:
            Numpy array with shape (N, 5) in float16: [x, y, z, intensity, padding]
        """
        raw_data = np.frombuffer(cloud.data, dtype=np.float32)
        point_step = cloud.point_step // 4  # 4 bytes per float32
        num_points = cloud.width * cloud.height
        points = raw_data.reshape(num_points, point_step)
        
        # Extract x, y, z, intensity (first 4 columns)
        xyz_i = points[:, :4]
        
        # Add padding column (5th column) required by C++ backend
        padding = np.zeros((num_points, 1), dtype=np.float32)
        final_points = np.hstack((xyz_i, padding))
        
        # Convert to float16 for FP16 model (reduces memory and improves performance)
        return np.ascontiguousarray(final_points.astype(np.float16))

    def _sync_callback(self, *msgs):
        """Callback when ApproximateTimeSynchronizer matches messages from all 7 topics.
        
        Args:
            *msgs: 7 messages - msgs[0-5] are Image messages, msgs[6] is PointCloud2
        """
        img_msgs = msgs[:6]
        lidar_msg = msgs[6]
        
        try:
            # 1. Prepare Inputs (with timing)
            prep_start = time.perf_counter()
            images = self._prepare_images(img_msgs)
            points = self._prepare_points(lidar_msg)
            prep_time = (time.perf_counter() - prep_start) * 1000  # Convert to ms
            
            if points.size == 0:
                self.get_logger().warn("Empty point cloud, skipping")
                return

            # 2. Inference (with timing)
            inf_start = time.perf_counter()
            detections = self.core.forward(images, points, True, False)
            inf_time = (time.perf_counter() - inf_start) * 1000  # Convert to ms
            
            # 3. Publish (with timing)
            pub_start = time.perf_counter()
            self._publish_detections(detections, lidar_msg.header)
            pub_time = (time.perf_counter() - pub_start) * 1000  # Convert to ms
            
            # Calculate and publish performance metrics
            total_time = prep_time + inf_time + pub_time
            fps = 1000 / total_time if total_time > 0 else 0
            
            self.get_logger().info(
                f"Processed frame: Inference={inf_time:.1f}ms, Total={total_time:.1f}ms, FPS={fps:.1f}"
            )
            
            # Publish metrics: [prep_time_ms, inference_time_ms, publish_time_ms, total_time_ms, fps]
            metrics_msg = Float64MultiArray()
            metrics_msg.layout = MultiArrayLayout()
            metrics_msg.layout.dim = [MultiArrayDimension()]
            metrics_msg.layout.dim[0].label = "metrics"
            metrics_msg.layout.dim[0].size = 5
            metrics_msg.layout.dim[0].stride = 5
            metrics_msg.data = [prep_time, inf_time, pub_time, total_time, fps]
            metrics_msg.layout.data_offset = 0
            self.metrics_pub.publish(metrics_msg)
            
        except Exception as e:
            self.get_logger().error(f"Inference Fail: {e}\n{traceback.format_exc()}")

    def _publish_detections(self, detections, header):
        """Convert model detections to ROS Detection3DArray message and publish.
        
        Args:
            detections: Raw detection array from model [N, 11] where columns are:
                       [x, y, z, length, width, height, yaw, vx, vy, class_id, score]
            header: ROS message header to copy to output
        """
        out_msg = Detection3DArray()
        out_msg.header = header
        
        if len(detections) == 0:
            self.pub.publish(out_msg)
            self.get_logger().info("Detections: 0")
            return
        
        valid_detections = []
        for det in detections:
            # Validate detection format
            if not isinstance(det, np.ndarray) or len(det) < 11:
                continue
                
            score = float(det[10])
            
            # Filter by confidence threshold
            if not np.isfinite(score) or score < self.conf_threshold:
                continue
            
            # Validate position and size values
            if not all(np.isfinite([det[0], det[1], det[2], det[3], det[4], det[5]])):
                continue
            
            ros_det = Detection3D()
            ros_det.header = header
            
            # Bounding box center position
            ros_det.bbox.center.position.x = float(det[0])
            ros_det.bbox.center.position.y = float(det[1])
            ros_det.bbox.center.position.z = float(det[2])
            
            # Bounding box dimensions
            ros_det.bbox.size.x = float(det[4])  # width
            ros_det.bbox.size.y = float(det[3])  # length
            ros_det.bbox.size.z = float(det[5])  # height
            
            # Orientation: convert yaw angle to quaternion
            yaw = float(det[6])
            if np.isfinite(yaw):
                ros_det.bbox.center.orientation.x = 0.0
                ros_det.bbox.center.orientation.y = 0.0
                ros_det.bbox.center.orientation.z = np.sin(yaw / 2.0)
                ros_det.bbox.center.orientation.w = np.cos(yaw / 2.0)
            else:
                # Default: identity quaternion (no rotation)
                ros_det.bbox.center.orientation.x = 0.0
                ros_det.bbox.center.orientation.y = 0.0
                ros_det.bbox.center.orientation.z = 0.0
                ros_det.bbox.center.orientation.w = 1.0
            
            # Class label and confidence score
            class_id = int(det[9]) if np.isfinite(det[9]) else 0
            if 0 <= class_id < len(self.CLASS_NAMES):
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = self.CLASS_NAMES[class_id]
                hyp.hypothesis.score = score
                ros_det.results.append(hyp)
            else:
                continue
                
            out_msg.detections.append(ros_det)
            valid_detections.append({
                'class': self.CLASS_NAMES[class_id],
                'score': score,
                'position': [float(det[0]), float(det[1]), float(det[2])],
                'size': [float(det[4]), float(det[3]), float(det[5])],  # width, length, height
                'yaw': float(det[6]) if np.isfinite(det[6]) else 0.0
            })
            
        self.pub.publish(out_msg)
        
        # Log detections summary
        if len(valid_detections) > 0:
            det_summary = f"Detections: {len(valid_detections)}"
            for i, det_info in enumerate(valid_detections[:5]):  # Log top 5 detections
                det_summary += (
                    f" | {det_info['class']}({det_info['score']:.2f}) "
                    f"@({det_info['position'][0]:.1f},{det_info['position'][1]:.1f},{det_info['position'][2]:.1f})"
                )
            if len(valid_detections) > 5:
                det_summary += f" ... (+{len(valid_detections) - 5} more)"
            self.get_logger().info(det_summary)
        else:
            self.get_logger().info("Detections: 0")

def main(args=None):
    rclpy.init(args=args)
    node = BEVFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()