import os
import csv
import json
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Header
from vision_msgs.msg import Detection3DArray


def _key(sec, nanosec):
    """Microsecond stamp key shared by detections, frame markers and metrics."""
    return int(sec) * 1_000_000 + int(nanosec) // 1000


def _stats(xs):
    if not xs:
        return None
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


class DetectionLoggerNode(Node):
    """Logs detections plus end-to-end benchmark data.

    End-to-end latency = logger receive time - frame stamp, where the stamp is
    set by replay_node immediately before publishing the 6 images + LiDAR.
    Drop rate is computed against /replay/frame markers (one per frame sent).

    Outputs in <log_dir>/<timestamp>/:
      metrics.csv       per received frame (legacy columns + e2e)
      detections.jsonl  per received frame, with sample_token when known
      frames.csv        per sent frame: received?, e2e, stage timings
      summary.json      aggregate stats (rewritten every few seconds + on exit)
    """

    def __init__(self):
        super().__init__("detection_logger")

        self.declare_parameter("log_dir", "logs/bevfusion_detections")
        self.declare_parameter("save_metrics", True)
        self.declare_parameter("save_detections", True)
        # First N received frames are excluded from stats (TensorRT/CUDA warmup).
        self.declare_parameter("warmup_frames", 5)
        self.declare_parameter("run_label", "")

        base_dir = self.get_parameter("log_dir").value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = self.get_parameter("run_label").value
        self.log_dir = os.path.join(base_dir, f"{timestamp}_{label}" if label else timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.warmup = int(self.get_parameter("warmup_frames").value)

        self.metrics_writer = None
        self.metrics_file = None
        if self.get_parameter("save_metrics").value:
            metrics_path = os.path.join(self.log_dir, "metrics.csv")
            self.metrics_file = open(metrics_path, "w", newline="")
            self.metrics_writer = csv.writer(self.metrics_file)
            self.metrics_writer.writerow(
                ["timestamp", "frame_id", "num_detections", "e2e_ms"]
            )
            self.get_logger().info(f"Logging metrics to {metrics_path}")

        self.detections_fh = None
        if self.get_parameter("save_detections").value:
            det_path = os.path.join(self.log_dir, "detections.jsonl")
            self.detections_fh = open(det_path, "w")
            self.get_logger().info(f"Logging detections to {det_path}")

        # Benchmark state, keyed by microsecond stamp.
        self.sent = {}       # key -> sample_token
        self.sent_order = []
        self.recv = {}       # key -> {"t": recv_s, "e2e_ms": .., "n": ..}
        self.recv_order = []
        self.stages = {}     # key -> [prep, inf, pub, total, fps, wait]

        self.sub = self.create_subscription(
            Detection3DArray, "/bevfusion/detections", self.callback, 10)
        self.frame_sub = self.create_subscription(
            Header, "/replay/frame", self._on_frame, 50)
        self.metrics_sub = self.create_subscription(
            Float64MultiArray, "/bevfusion/metrics", self._on_metrics, 50)
        self.summary_timer = self.create_timer(5.0, self.write_summary)
        self.get_logger().info(f"Benchmark logs -> {self.log_dir}")

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_frame(self, msg: Header):
        k = _key(msg.stamp.sec, msg.stamp.nanosec)
        if k not in self.sent:
            self.sent[k] = msg.frame_id
            self.sent_order.append(k)

    def _on_metrics(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) < 8:  # older bev_node without exact stamp; cannot join
            return
        self.stages[_key(d[6], d[7])] = d[:6]

    def callback(self, msg: Detection3DArray):
        recv_s = self._now_s()
        stamp = msg.header.stamp
        k = _key(stamp.sec, stamp.nanosec)
        e2e_ms = (recv_s - (stamp.sec + stamp.nanosec * 1e-9)) * 1000
        frame_time = f"{stamp.sec}.{stamp.nanosec:09d}"
        frame_id = msg.header.frame_id or ""

        if k not in self.recv:
            self.recv[k] = {"t": recv_s, "e2e_ms": e2e_ms, "n": len(msg.detections)}
            self.recv_order.append(k)

        if self.metrics_writer is not None:
            self.metrics_writer.writerow(
                [frame_time, frame_id, len(msg.detections), f"{e2e_ms:.2f}"]
            )
            self.metrics_file.flush()

        if self.detections_fh is not None:
            record = {
                "timestamp": frame_time,
                "frame_id": frame_id,
                "sample_token": self.sent.get(k),
                "e2e_ms": e2e_ms,
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

    def write_summary(self):
        if not self.sent_order and not self.recv_order:
            return
        counted = self.recv_order[self.warmup:]
        e2e = [self.recv[k]["e2e_ms"] for k in counted]
        cols = ["prep_ms", "inference_ms", "publish_ms", "callback_total_ms",
                "fps_callback", "wait_ms"]
        stage_stats = {}
        for i, c in enumerate(cols):
            vals = [self.stages[k][i] for k in counted if k in self.stages]
            stage_stats[c] = _stats(vals)

        throughput = None
        if len(counted) >= 2:
            span = self.recv[counted[-1]]["t"] - self.recv[counted[0]]["t"]
            if span > 0:
                throughput = (len(counted) - 1) / span

        # Rate replay actually achieved (may be below the requested publish_rate).
        offered = None
        if len(self.sent_order) >= 2:
            span = (self.sent_order[-1] - self.sent_order[0]) * 1e-6
            if span > 0:
                offered = (len(self.sent_order) - 1) / span

        # Drop rate only over frames whose marker we saw (sent after logger start).
        n_sent = len(self.sent_order)
        n_recv_of_sent = sum(1 for k in self.sent_order if k in self.recv)
        summary = {
            "frames_sent": n_sent,
            "frames_received": len(self.recv_order),
            "frames_received_of_sent": n_recv_of_sent,
            "drop_rate": (1 - n_recv_of_sent / n_sent) if n_sent else None,
            "warmup_frames_excluded": self.warmup,
            "e2e_ms": _stats(e2e),
            "offered_fps": offered,
            "throughput_fps": throughput,
            "stages": stage_stats,
        }
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(self.log_dir, "frames.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stamp_us", "sample_token", "received", "e2e_ms",
                        "num_detections"] + cols)
            for k in self.sent_order or self.recv_order:
                r = self.recv.get(k)
                st = self.stages.get(k, [""] * len(cols))
                w.writerow([k, self.sent.get(k, ""), int(r is not None),
                            f"{r['e2e_ms']:.2f}" if r else "",
                            r["n"] if r else ""] + list(st))

    def destroy_node(self):
        self.write_summary()
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
