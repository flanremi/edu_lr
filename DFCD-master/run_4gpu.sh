#!/bin/bash
# 4 路 3090 分布式训练入口（Ubuntu）
# 在 DFCD-master 根目录执行: bash run_4gpu.sh
# 或: chmod +x run_4gpu.sh && ./run_4gpu.sh

cd "$(dirname "$0")"
python run_train_2020.py
