import os
import time
import json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge

# Resolve workspace root and NuScenes data location
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

# NuScenes dataset root directory (can be overridden with NUSCENES_ROOT env var)
DEFAULT_NUSCENES_ROOT = os.path.join(WORKSPACE_ROOT, "data", "nuscenes")
NUSCENES_ROOT = os.environ.get("NUSCENES_ROOT", DEFAULT_NUSCENES_ROOT)


class ReplayNode(Node):
    def __init__(self):
        super().__init__('replay_node')
        
        # Parameters for pre-resizing optimization
        self.declare_parameter("pre_resize", True)  # Enable pre-resizing at source
        self.declare_parameter("target_width", 1600)  # Target image width
        self.declare_parameter("target_height", 900)  # Target image height
        
        self.pre_resize = self.get_parameter("pre_resize").value
        self.target_w = self.get_parameter("target_width").value
        self.target_h = self.get_parameter("target_height").value
        
        self.bridge = CvBridge()
        self.pubs = {}
        
        # ROS2 topics for 6 camera images (must match bev_node.py subscriber topics)
        self.topics = [
            "/cam_front/image_raw", 
            "/cam_front_right/image_raw", 
            "/cam_front_left/image_raw", 
            "/cam_back/image_raw", 
            "/cam_back_left/image_raw", 
            "/cam_back_right/image_raw"
        ]
        
        # NuScenes camera channel names (used to map from dataset to topics)
        self.cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        # Create publishers with large queue depth to prevent message drops
        for topic in self.topics:
            self.pubs[topic] = self.create_publisher(Image, topic, 50)
        
        self.lidar_pub = self.create_publisher(PointCloud2, "/lidar_top/points", 50)
        
        # Parameters
        self.declare_parameter("publish_rate", 5)  # Hz (frames per second)
        
        publish_rate = self.get_parameter("publish_rate").value
        self.publishing_started = False  # Will be set to True after startup delay
        
        # Build an index of keyframes from the original nuScenes layout
        self.frames = self._build_nuscenes_index(NUSCENES_ROOT)
        self.num_frames = len(self.frames)
        self.frame_idx = 0
        self.finished = self.num_frames == 0

        if self.num_frames == 0:
            self.get_logger().error(
                f"No playable frames found under '{NUSCENES_ROOT}'. "
                "Check that data/nuscenes/v1.0-mini is present."
            )

        # Log pre-resize configuration
        if self.pre_resize:
            self.get_logger().info(f"Pre-resize ENABLED: Images will be resized to {self.target_w}x{self.target_h} at source")
        else:
            self.get_logger().info("Pre-resize DISABLED: Images will be published at original size")

        # Startup delay: wait for subscribers (bev_node) to connect before publishing
        self.declare_parameter("startup_delay", 3.0)  # seconds
        startup_delay = self.get_parameter("startup_delay").value
        
        # Create timer based on publish rate
        timer_period = 1.0 / publish_rate
        if startup_delay > 0:
            # Delay publishing to allow subscribers to connect
            self.startup_timer = self.create_timer(startup_delay, self._start_publishing)
            self.publishing_started = False
            self.timer = None  # Will be created after startup delay
        else:
            self.publishing_started = True
            self.timer = self.create_timer(timer_period, self.publish_frame)
        
        self.get_logger().info(
            f"Replay Node reading {self.num_frames} keyframes from {NUSCENES_ROOT} at {publish_rate} Hz"
        )
        if startup_delay > 0:
            self.get_logger().info(f"Waiting {startup_delay}s for subscribers to connect before publishing...")
    
    def _start_publishing(self):
        """Callback to start publishing after startup delay"""
        if not self.publishing_started:
            self.startup_timer.cancel()
            self.publishing_started = True
            # Recreate the main timer
            publish_rate = self.get_parameter("publish_rate").value
            timer_period = 1.0 / publish_rate
            self.timer = self.create_timer(timer_period, self.publish_frame)
            self.get_logger().info("Startup delay complete. Starting to publish frames...")

    def publish_frame(self):
        # Don't publish until startup delay is complete
        if not self.publishing_started:
            return
        
        # Stop if finished
        if self.finished:
            return

        # Bounds check
        if self.frame_idx >= self.num_frames:
            self.get_logger().info(
                f"End of sequence reached at frame {self.frame_idx}. Exiting."
            )
            self.finished = True
            if self.timer is not None:
                self.timer.cancel()
            return

        # Capture single timestamp for all messages in this frame
        # Ensures all 7 messages (6 images + 1 lidar) have identical timestamps
        timestamp = self.get_clock().now().to_msg()
        
        frame = self.frames[self.frame_idx]

        # Pre-load all data before publishing to ensure identical timestamps
        # 1. Load LiDAR point cloud
        pc_msg = None
        try:
            points = self._load_lidar_points(frame["lidar"])
            pc_msg = self.create_cloud(points, timestamp)
        except Exception as e:
            self.get_logger().error(f"Failed to load LiDAR '{frame['lidar']}': {e}")

        # 2. Pre-load all images
        img_msgs = []
        for topic, cam_name in zip(self.topics, self.cam_names):
            img_path = frame["cams"].get(cam_name)
            if img_path is None:
                continue

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Pre-resize images if enabled (saves processing time in bev_node)
            if self.pre_resize:
                h, w = img.shape[:2]
                if h != self.target_h or w != self.target_w:
                    img = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header.stamp = timestamp
            msg.header.frame_id = "base_link"
            img_msgs.append((topic, msg))

        # ApproximateTimeSynchronizer requires all 7 messages (6 images + 1 lidar)
        # Skip frame if any message is missing
        required_msgs = len(self.topics) + 1  # 6 images + 1 lidar = 7
        actual_msgs = len(img_msgs) + (1 if pc_msg is not None else 0)
        
        if actual_msgs < required_msgs:
            self.frame_idx += 1
            return

        # Publish all messages with identical timestamp for synchronization
        if pc_msg is not None:
            self.lidar_pub.publish(pc_msg)
        
        for topic, msg in img_msgs:
            self.pubs[topic].publish(msg)

        self.get_logger().info(f"Published frame {self.frame_idx}")
        self.frame_idx += 1

    def create_cloud(self, points, timestamp):
        """Create ROS PointCloud2 message from numpy array.
        
        Args:
            points: Numpy array with shape (N, 5) containing [x, y, z, intensity, padding]
            timestamp: ROS timestamp for the point cloud
            
        Returns:
            PointCloud2 message ready to publish
        """
        msg = PointCloud2()
        msg.header.stamp = timestamp
        msg.header.frame_id = "lidar_top"
        
        msg.height = 1
        msg.width = points.shape[0]
        
        # Define point fields: x, y, z, intensity (4 fields, 16 bytes)
        # Note: points have 5 columns but only 4 are exposed as fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 20  # 5 floats * 4 bytes (includes padding column)
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        
        msg.data = points.astype(np.float32).tobytes()
        return msg

    # --------- Internal helpers for NuScenes indexing & IO ----------

    def _build_nuscenes_index(self, root):
        """Build index of keyframes from NuScenes dataset.
        
        Reads sample_data.json to find all keyframes and groups them by sample_token.
        Each frame contains LIDAR_TOP and up to 6 camera views.
        
        Args:
            root: Path to NuScenes dataset root directory
            
        Returns:
            List of frame dictionaries with 'lidar', 'cams', and metadata
        """
        v1_root = os.path.join(root, "v1.0-mini")
        sd_file = os.path.join(v1_root, "sample_data.json")

        if not os.path.exists(sd_file):
            self.get_logger().error(f"NuScenes sample_data.json not found at {sd_file}")
            return []

        # Load all necessary JSON files for mapping
        sensor_file = os.path.join(v1_root, "sensor.json")
        calibrated_sensor_file = os.path.join(v1_root, "calibrated_sensor.json")
        
        if not os.path.exists(sensor_file) or not os.path.exists(calibrated_sensor_file):
            self.get_logger().error(f"NuScenes sensor or calibrated_sensor.json not found")
            return []

        # Build mapping: sensor_token -> channel
        with open(sensor_file, "r") as f:
            sensors = json.load(f)
        sensor_to_channel = {s["token"]: s["channel"] for s in sensors}

        # Build mapping: calibrated_sensor_token -> sensor_token
        with open(calibrated_sensor_file, "r") as f:
            calibrated_sensors = json.load(f)
        calib_to_sensor = {cs["token"]: cs["sensor_token"] for cs in calibrated_sensors}

        # Load sample data and build frame index
        with open(sd_file, "r") as f:
            sample_data = json.load(f)

        # Group keyframes by sample_token: {sample_token -> {channel -> (filepath, timestamp)}}
        per_sample = {}
        for sd in sample_data:
            if not sd.get("is_key_frame", False):
                continue
            
            # Map through calibrated_sensor -> sensor -> channel
            calib_token = sd.get("calibrated_sensor_token")
            if not calib_token:
                continue
            
            sensor_token = calib_to_sensor.get(calib_token)
            if not sensor_token:
                continue
            
            channel = sensor_to_channel.get(sensor_token, "")
            if channel not in (
                "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                "LIDAR_TOP",
            ):
                continue

            sample_token = sd["sample_token"]
            sample_entry = per_sample.setdefault(sample_token, {})
            sample_entry[channel] = (
                os.path.join(root, sd["filename"]),
                sd["timestamp"],
            )

        # Build frame list: each frame must have lidar + at least one camera
        frames = []
        for sample_token, chans in per_sample.items():
            if "LIDAR_TOP" not in chans:
                continue

            lidar_path, lidar_ts = chans["LIDAR_TOP"]
            cams = {}
            for cam in self.cam_names:
                if cam in chans:
                    cams[cam] = chans[cam][0]

            if len(cams) == 0:
                continue

            frames.append({
                "sample_token": sample_token,
                "lidar": lidar_path,
                "lidar_ts": lidar_ts,
                "cams": cams,
            })

        # Sort frames by lidar timestamp to maintain temporal order
        frames.sort(key=lambda x: x["lidar_ts"])

        self.get_logger().info(
            f"Indexed {len(frames)} keyframes from NuScenes under {root}"
        )
        return frames

    def _load_lidar_points(self, lidar_file):
        """Load NuScenes lidar binary file into numpy array.
        
        Args:
            lidar_file: Path to .pcd.bin file
            
        Returns:
            Numpy array with shape (N, 5): [x, y, z, intensity, padding]
            Handles both 4-field and 5-field formats, padding to 5 if needed
        """
        data = np.fromfile(lidar_file, dtype=np.float32)
        if data.size % 5 != 0:
            # Fallback: try 4 fields (x, y, z, intensity)
            n = data.size // 4
            points = data[: n * 4].reshape(-1, 4)
            # Pad to 5D for compatibility
            pad = np.zeros((points.shape[0], 1), dtype=np.float32)
            return np.hstack((points, pad))

        points = data.reshape(-1, 5)
        return points

def main(args=None):
    rclpy.init(args=args)
    node = ReplayNode()
    try:
        # Spin until node finishes or interrupted
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass  # Ignore if already shut down

if __name__ == '__main__':
    main()