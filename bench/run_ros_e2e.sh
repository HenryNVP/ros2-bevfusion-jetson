#!/bin/bash
# End-to-end ROS 2 benchmark: replay -> bev_node -> detection_logger, plus system sampling.
# Usage (inside container, as root for power rails):
#   bash /workspace/bench/run_ros_e2e.sh <rate_hz> <label> [model_variant] [lidar_precision]
# Result: /workspace/logs/bevfusion_detections/<time>_<label>_<rate>hz/{summary,system}.json
# DDS socket-buffer fix is on by default (BENCH_DDS_TUNED=0 to disable); host needs
#   sudo sysctl -w net.core.rmem_max=2147483647
RATE=${1:?publish rate in Hz}
LABEL=${2:-run}_${RATE}hz
MODEL=${3:-resnet50int8}
LIDAR=${4:-fp16}
export BENCH_DDS_TUNED=${BENCH_DDS_TUNED:-1}
. /workspace/bench/env_bench.sh
BIN=/workspace/ros2_ws/install/bevfusion_pkg/lib/bevfusion_pkg
LOGS=/workspace/bench/ros_logs/${LABEL}
mkdir -p "$LOGS"
cd /workspace

python3 /workspace/bench/sysmon.py record "$LOGS/sysmon.csv" 0.2 & MON=$!
cleanup() { kill $MON 2>/dev/null; kill -INT $LOGGER $BEV $REPLAY 2>/dev/null; sleep 2; kill -9 $BEV $REPLAY 2>/dev/null; }
trap cleanup EXIT

"$BIN/bev_node" --ros-args -p model_variant:=$MODEL -p lidar_precision:=$LIDAR \
  > "$LOGS/bev_node.log" 2>&1 & BEV=$!
for i in $(seq 180); do grep -q "Node Ready" "$LOGS/bev_node.log" && break; sleep 1; done
grep -q "Node Ready" "$LOGS/bev_node.log" || { echo "bev_node failed to start"; tail "$LOGS/bev_node.log"; exit 1; }

"$BIN/detection_logger_node" --ros-args -p run_label:=$LABEL -p save_detections:=${SAVE_DETS:-false} \
  -p log_dir:=/workspace/logs/bevfusion_detections > "$LOGS/logger.log" 2>&1 & LOGGER=$!
sleep 2
UDP0=$(awk '/^Udp:/{getline; print $6}' /proc/net/snmp)
"$BIN/replay_node" --ros-args -p publish_rate:=$RATE > "$LOGS/replay.log" 2>&1 & REPLAY=$!
T0=$(date +%s.%N)

echo "[$LABEL] model=$MODEL lidar=$LIDAR dds=${CYCLONEDDS_URI:-default} replaying..."
while ! grep -q "End of sequence" "$LOGS/replay.log"; do
  kill -0 $REPLAY 2>/dev/null || { echo "replay died"; tail "$LOGS/replay.log"; exit 1; }
  sleep 2
done
T1=$(date +%s.%N)
sleep 8   # drain in-flight frames
kill -INT $LOGGER; wait $LOGGER 2>/dev/null
UDP1=$(awk '/^Udp:/{getline; print $6}' /proc/net/snmp)
kill $MON; wait $MON 2>/dev/null

DIR=$(grep -o "Benchmark logs -> .*" "$LOGS/logger.log" | sed 's/Benchmark logs -> //')
cp "$LOGS/sysmon.csv" "$DIR/"
python3 - "$DIR" "$LOGS/sysmon.csv" "$T0" "$T1" "$((UDP1-UDP0))" "$MODEL" "$LIDAR" <<'EOF'
import json, sys
sys.path.insert(0, "/workspace/bench")
import sysmon
d, csv, t0, t1, udp, model, lidar = sys.argv[1:]
s = json.load(open(f"{d}/summary.json"))
sysm = sysmon.summarize(csv, s.get("throughput_fps"), float(t0), float(t1))
sysm.update({"udp_rcvbuf_errors": int(udp), "model_variant": model, "lidar_precision": lidar})
json.dump(sysm, open(f"{d}/system.json", "w"), indent=2)
e = s["e2e_ms"] or {}
print(json.dumps({"dir": d, "received": f'{s["frames_received_of_sent"]}/{s["frames_sent"]}',
  "drop_rate": s["drop_rate"], "offered_fps": s["offered_fps"], "throughput_fps": s["throughput_fps"],
  "e2e_p50": e.get("p50"), "e2e_p99": e.get("p99"),
  "inference_p50": (s["stages"]["inference_ms"] or {}).get("p50"), **sysm}, indent=2))
EOF
