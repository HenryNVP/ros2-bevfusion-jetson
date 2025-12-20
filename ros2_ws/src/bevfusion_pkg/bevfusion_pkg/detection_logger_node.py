import os
import csv
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

class DetectionLoggerNode(Node):
    def __init__(self):
        super().__init__("detection_logger")

        # Parameters
        self.declare_parameter("log_dir", "logs/bevfusion_detections")
        self.declare_parameter("save_metrics", True)
        self.declare_parameter("save_detections", True)

        base_dir = self.get_parameter("log_dir").value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_writer = None
        if self.get_parameter("save_metrics").value:
            metrics_path = os.path.join(self.log_dir, "metrics.csv")
            f = open(metrics_path, "w", newline="")
            self.metrics_file = f
            self.metrics_writer = csv.writer(f)
            self.metrics_writer.writerow(
                ["timestamp", "frame_id", "num_detections"]
            )
            self.get_logger().info(f"Logging metrics to {metrics_path}")
        else:
            self.metrics_file = None

        self.detections_fh = None
        if self.get_parameter("save_detections").value:
            det_path = os.path.join(self.log_dir, "detections.jsonl")
            self.detections_fh = open(det_path, "w")
            self.get_logger().info(f"Logging detections to {det_path}")

        # Subscriber
        self.sub = self.create_subscription(
            Detection3DArray,
            "/bevfusion/detections",
            self.callback,
            10,
        )

    def callback(self, msg: Detection3DArray):
        stamp = msg.header.stamp
        frame_time = f"{stamp.sec}.{stamp.nanosec:09d}"
        frame_id = msg.header.frame_id or ""

        # Simple per-frame metrics
        if self.metrics_writer is not None:
            self.metrics_writer.writerow(
                [frame_time, frame_id, len(msg.detections)]
            )
            self.metrics_file.flush()

        # Detailed per-detection log
        if self.detections_fh is not None:
            record = {
                "timestamp": frame_time,
                "frame_id": frame_id,
                "detections": [],
            }
            for det in msg.detections:
                classes = [
                    {
                        "class_id": r.hypothesis.class_id,
                        "score": float(r.hypothesis.score),
                    }
                    for r in det.results
                ]
                record["detections"].append(
                    {
                        "position": [
                            float(det.bbox.center.position.x),
                            float(det.bbox.center.position.y),
                            float(det.bbox.center.position.z),
                        ],
                        "size": [
                            float(det.bbox.size.x),
                            float(det.bbox.size.y),
                            float(det.bbox.size.z),
                        ],
                        "orientation": {
                            "x": float(det.bbox.center.orientation.x),
                            "y": float(det.bbox.center.orientation.y),
                            "z": float(det.bbox.center.orientation.z),
                            "w": float(det.bbox.center.orientation.w),
                        },
                        "classes": classes,
                    }
                )
            self.detections_fh.write(json.dumps(record) + "\n")
            self.detections_fh.flush()

    def destroy_node(self):
        if self.metrics_file:
            self.metrics_file.close()
        if self.detections_fh:
            self.detections_fh.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DetectionLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()