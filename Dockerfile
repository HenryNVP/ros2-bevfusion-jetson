# Custom Docker image for ROS2 BEV-Fusion
# Base image: ROS2 Humble for Jetson (L4T r36.4.0)
FROM dustynv/ros:humble-desktop-l4t-r36.4.0

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Fix expired ROS2 GPG key and install system dependencies
# The ROS2 repository GPG key may expire, so we update it first
RUN apt-get update --allow-insecure-repositories 2>/dev/null || apt-get update || true && \
    apt-get install -y --allow-unauthenticated curl gnupg2 lsb-release ca-certificates && \
    (curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - 2>/dev/null || \
     curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - 2>/dev/null || \
     apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F42ED6FBAB17C654 2>/dev/null || \
     apt-key adv --keyserver hkp://pgp.mit.edu:80 --recv-keys F42ED6FBAB17C654 2>/dev/null || true) && \
    apt-get update && \
    apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken -y && \
    apt-get install -y \
    libprotobuf-dev \
    protobuf-compiler \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    onnx \
    onnx-graphsurgeon \
    pycuda \
    --index-url https://pypi.org/simple

# Setup ROS2 environment (will be sourced in entrypoint)
RUN echo "source /opt/ros/humble/install/setup.bash" >> /root/.bashrc

# Create workspace directory structure
RUN mkdir -p /workspace/ros2_ws/src

# Copy project files (excluding build artifacts)
COPY env.sh /workspace/
COPY ros2_ws/src /workspace/ros2_ws/src/

# Note: Lidar_AI_Solution will be cloned at runtime or can be mounted as volume
# This allows flexibility - either build everything in image or mount at runtime

# Set default command
CMD ["/bin/bash"]

