#!/bin/bash
set -e
CUDA_VISIBLE_DEVICE=1 python train_drive.py
CUDA_VISIBLE_DEVICE=1 python test_drive.py
