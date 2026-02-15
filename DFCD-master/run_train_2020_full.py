# -*- coding: utf-8 -*-
"""
2020 训练入口：使用 100% 训练集，不划分验证集。
用于合并 85/15 分割后，将全部数据用于训练；验证在训练结束后用外部验证集单独执行。

运行方式：
  单卡：  python run_train_2020_full.py
  多卡：  python run_train_2020_full.py   # 自动检测 GPU 数并用 DDP
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "models"))

import run_config
from utils import load_data, construct_data_geometric
from models.dfcd import DFCD

# 覆盖为 2020 + 100% 训练（不划分验证集）
run_config.CONFIG.update({
    "data_type": "2020",
    "data_root": _SCRIPT_DIR,
    "test_size": 0.0,            # 0% 验证，全部用于训练
    "split": "Original",         # 使用 train_test_split，test_size=0 即全量训练
    "text_embedding_model": "remote",
    "epoch": 20,
    "lr": 1e-4,
    "batch_size": 1024,
    "seed": 0,
    "device": "cuda:0",
})


def train_only_loop(model, config):
    """仅训练，不执行 test_step（单卡）。"""
    model.train()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
    )
    for epoch_i in range(config["epoch"]):
        epoch_losses = []
        for batch_data in tqdm(config["train_dataloader"], desc=f"Epoch {epoch_i}"):
            student_id, exercise_id, knowledge_point, y = [
                d.to(config["device"]) for d in batch_data
            ]
            pred = model(student_id, exercise_id, knowledge_point)
            loss = nn.BCELoss()(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.monotonicity()
            epoch_losses.append(loss.item())
        print(f"[{epoch_i:03d}/{config['epoch']}] Loss: {np.mean(epoch_losses):.4f}")
    print("训练完成。请使用外部验证集进行评估。")


def _ddp_entry(rank, world_size, config):
    from multi_gpu_train import main_ddp
    main_ddp(rank, world_size, config)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        config = run_config._build_config()
        load_data(config)
        config["in_channels_init"] = (
            config["stu_num"] + config["prob_num"] + config["know_num"]
        )
        train_data = construct_data_geometric(config, data=config["np_train"])
        config["train_data"] = train_data.to(config["device"])
        config["full_data"] = train_data.to(config["device"])
        model = DFCD(config)
        train_only_loop(model, config)
    else:
        import torch.multiprocessing as mp
        config = run_config._build_config()
        mp.spawn(_ddp_entry, nprocs=n_gpus, args=(n_gpus, config), join=True)
