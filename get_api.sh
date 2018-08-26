#!/bin/bash

# Taken from https://www.tensorflow.org/install/install_c

TF_TYPE="cpu" # Change to "gpu" for GPU support
OS="darwin" # Change to "linux" for linux
TARGET_DIRECTORY="/opt/tensorflow"
curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.10.1.tar.gz" |
   sudo tar -C $TARGET_DIRECTORY -xz
