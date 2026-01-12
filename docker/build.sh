#!/bin/bash
# Copyright Ioannis Tsikelis
#
# Script to build HURo docker container

# Usage: ./build.sh OR ./build.sh opensot

IMAGE_TYPE=$1

if [ "$IMAGE_TYPE" == "opensot" ]; then
    echo "Building OpenSoT HURo image..."
    docker build -f Dockerfile.opensot -t huro_opensot:latest .
else
    echo "Building the base HURo image..."
    docker build -t huro:latest .
fi
