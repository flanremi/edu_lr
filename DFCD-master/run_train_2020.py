# -*- coding: utf-8 -*-
"""
2020 数据集训练入口：按学生划分 85% 训练 / 15% 测试，使用 embedding_remote.pkl。

数据目录：data/2020/（TotalData.csv、q.csv、embedding_remote.pkl）

运行方式（Ubuntu）：
  单卡：  python run_train_2020.py
  4 卡：  python run_train_2020.py   # 自动检测多 GPU 并使用 DDP
  指定卡：CUDA_VISIBLE_DEVICES=0,1,2,3 python run_train_2020.py

物理分割（训练/测试集落盘）：
  - 首次运行会在 data/2020/split_stu_s0_pt15/ 下生成：
    TotalData_train.csv、TotalData_test.csv、split_info.json
  - 之后运行会直接从此目录加载，训练集与测试集在物理上隔绝、划分固定。
  - 若需重新划分，删除上述 split_stu_s0_pt15 目录即可。
"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "models"))

import torch
import run_config

# 覆盖为 2020 数据集；split/test_size 见下方
# 使用 100% 训练集时请改用 run_train_2020_full.py
run_config.CONFIG.update({
    "data_type": "2020",
    "data_root": _SCRIPT_DIR,
    "test_size": 0.15,          # 按学生 85/15 时用 0.15；合并后改用 run_train_2020_full.py
    "split": "Stu",             # Stu=按学生划分 | Original+test_size=0=全量（见 run_train_2020_full）
    "text_embedding_model": "remote",
    "epoch": 20,
    "lr": 1e-4,
    "batch_size": 1024,
    "seed": 0,
    "device": "cuda:0",
})

def _ddp_entry(rank, world_size, config):
    from multi_gpu_train import main_ddp
    main_ddp(rank, world_size, config)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        run_config.main()
    else:
        import torch.multiprocessing as mp
        config = run_config._build_config()
        mp.spawn(_ddp_entry, nprocs=n_gpus, args=(n_gpus, config), join=True)
