#!/bin/bash

# Alternative Docker build method for systems with iptables issues
# This script builds the image by running commands inside a container
# instead of using docker build, which avoids the iptables raw table requirement

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="${IMAGE_NAME:-ros2-bevfusion}"
IMAGE_TAG="${IMAGE_TAG:-jetson-l4t-r36.4.0}"
BASE_IMAGE="dustynv/ros:humble-desktop-l4t-r36.4.0"
TEMP_CONTAINER="bevfusion-builder-temp"

echo -e "${GREEN}Alternative Docker build method (for iptables issues)...${NC}"
echo -e "Base image: ${YELLOW}${BASE_IMAGE}${NC}"
echo -e "Target image: ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo ""

# Check if base image exists
echo -e "${YELLOW}Checking if base image exists...${NC}"
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${BASE_IMAGE}$"; then
    echo -e "${YELLOW}Base image not found. Pulling ${BASE_IMAGE}...${NC}"
    docker pull "${BASE_IMAGE}"
else
    echo -e "${GREEN}Base image found.${NC}"
fi

# Clean up any existing temp container
echo -e "${YELLOW}Cleaning up any existing temporary container...${NC}"
docker rm -f "${TEMP_CONTAINER}" 2>/dev/null || true

# Create a temporary container from the base image
echo -e "${YELLOW}Creating temporary container...${NC}"
docker create --name "${TEMP_CONTAINER}" \
    --network=host \
    "${BASE_IMAGE}" \
    /bin/bash -c "tail -f /dev/null"

# Start the container
docker start "${TEMP_CONTAINER}"

echo -e "${YELLOW}Installing dependencies in container...${NC}"

# Install dependencies
docker exec "${TEMP_CONTAINER}" /bin/bash -c "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    
    # Update package lists (ignore ROS2 GPG warnings)
    apt-get update 2>&1 | grep -v 'EXPKEYSIG' || true
    
    # Install system packages (from Ubuntu repos, not ROS2)
    apt-get install -y \
        libprotobuf-dev \
        protobuf-compiler \
        build-essential \
        cmake \
        git \
        wget
    
    # Install Python packages
    pip3 install --no-cache-dir \
        onnx \
        onnx-graphsurgeon \
        pycuda \
        --index-url https://pypi.org/simple
    
    # Setup ROS2 environment
    echo 'source /opt/ros/humble/install/setup.bash' >> /root/.bashrc
    
    # Create workspace structure
    mkdir -p /workspace/ros2_ws/src
"

# Copy project files
echo -e "${YELLOW}Copying project files to container...${NC}"
docker cp env.sh "${TEMP_CONTAINER}:/workspace/" 2>/dev/null || true
docker cp ros2_ws/src "${TEMP_CONTAINER}:/workspace/ros2_ws/" 2>/dev/null || true

# Commit the container as a new image with proper CMD
echo -e "${YELLOW}Committing container as new image...${NC}"
docker commit --change='CMD ["/bin/bash"]' "${TEMP_CONTAINER}" "${IMAGE_NAME}:${IMAGE_TAG}"

# Clean up
echo -e "${YELLOW}Cleaning up temporary container...${NC}"
docker rm -f "${TEMP_CONTAINER}"

echo ""
echo -e "${GREEN}✓ Successfully built ${IMAGE_NAME}:${IMAGE_TAG}${NC}"


