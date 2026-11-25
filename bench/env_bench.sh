# Source inside the container: . /workspace/bench/env_bench.sh
source /opt/ros/humble/install/setup.bash
source /workspace/env.sh
[ -f /workspace/ros2_ws/install/setup.bash ] && source /workspace/ros2_ws/install/setup.bash
# aarch64: sklearn's bundled libgomp fails with "cannot allocate memory in static TLS block"
export LD_PRELOAD=$(ls /usr/local/lib/python3.8/dist-packages/scikit_learn.libs/libgomp-*.so* 2>/dev/null | head -1)
export MPLBACKEND=Agg
# Opt-in large DDS socket buffer (needs: sudo sysctl -w net.core.rmem_max=2147483647)
[ "${BENCH_DDS_TUNED:-0}" = "1" ] && export CYCLONEDDS_URI=file:///workspace/bench/cyclonedds.xml
