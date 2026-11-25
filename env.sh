export LD_LIBRARY_PATH=/workspace/Lidar_AI_Solution/CUDA-BEVFusion/build:$LD_LIBRARY_PATH
# libspconv: pick the folder matching your arch/CUDA (aarch64_cuda11.4 = JetPack 5, aarch64_cuda12.8 = JetPack 6)
export LD_LIBRARY_PATH=/workspace/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/aarch64_cuda11.4:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/Lidar_AI_Solution/CUDA-BEVFusion/build:$PYTHONPATH
