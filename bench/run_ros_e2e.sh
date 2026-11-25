#!/bin/bash
# End-to-end ROS 2 benchmark: replay -> bev_node -> detection_logger.
# Usage (inside container): bash /workspace/bench/run_ros_e2e.sh <rate_hz> [label]
# Result: /workspace/logs/bevfusion_detections/<time>_<label>/summary.json
# (no set -u: ROS setup scripts reference unset vars)
RATE=${1:?publish rate in Hz}
LABEL=${2:-baseline}_${RATE}hz
. /workspace/bench/env_bench.sh
BIN=/workspace/ros2_ws/install/bevfusion_pkg/lib/bevfusion_pkg
LOGS=/workspace/bench/ros_logs/${LABEL}
mkdir -p "$LOGS"
cd /workspace

( while true; do echo "$(date +%s),$(free -m | awk '/Mem:/{print $3}'),$(free -m | awk '/Swap:/{print $3}')"; sleep 1; done ) > "$LOGS/mem.csv" & MEM=$!

cleanup() { kill $MEM 2>/dev/null; kill -INT $LOGGER $BEV $REPLAY 2>/dev/null; sleep 2; kill -9 $BEV $REPLAY 2>/dev/null; }
trap cleanup EXIT

"$BIN/bev_node" > "$LOGS/bev_node.log" 2>&1 & BEV=$!
for i in $(seq 120); do grep -q "Node Ready" "$LOGS/bev_node.log" && break; sleep 1; done
grep -q "Node Ready" "$LOGS/bev_node.log" || { echo "bev_node failed to start"; tail "$LOGS/bev_node.log"; exit 1; }

"$BIN/detection_logger_node" --ros-args -p run_label:=$LABEL -p save_detections:=false \
  -p log_dir:=/workspace/logs/bevfusion_detections > "$LOGS/logger.log" 2>&1 & LOGGER=$!
sleep 2
"$BIN/replay_node" --ros-args -p publish_rate:=$RATE > "$LOGS/replay.log" 2>&1 & REPLAY=$!

echo "[$LABEL] replaying..."
while ! grep -q "End of sequence" "$LOGS/replay.log"; do
  kill -0 $REPLAY 2>/dev/null || { echo "replay died"; tail "$LOGS/replay.log"; exit 1; }
  sleep 2
done
sleep 8   # drain in-flight frames
kill -INT $LOGGER; wait $LOGGER 2>/dev/null
DIR=$(grep -o "Benchmark logs -> .*" "$LOGS/logger.log" | sed 's/Benchmark logs -> //')
echo "[$LABEL] results in $DIR"
cat "$DIR/summary.json"
awk -F, '$2>m{m=$2} $3>s{s=$3} END{print "peak RAM used MB:",m,"  peak swap MB:",s}' "$LOGS/mem.csv" | tee "$DIR/mem_peak.txt"
cp "$LOGS/mem.csv" "$DIR/"
