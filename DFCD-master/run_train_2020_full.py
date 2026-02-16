# -*- coding: utf-8 -*-
"""
2020 训练入口：使用 100% 训练集 + 外部测试集，每 epoch 评估。

训练集：data/2020/TotalData.csv（全量）
测试集：data/2020/TotalData_test.csv（由 convert_to_dfcd.py 从 test_public_task_4_more_splits.csv 转换）

运行方式：
  单卡：  python run_train_2020_full.py
  多卡：  python run_train_2020_full.py   # 自动检测 GPU 数并用 DDP

模型保存：训练结束后权重写入本脚本同目录下的 dfcd_2020_full.pt（可由 SAVE_PATH 修改）。
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

# 训练结束后保存权重的路径（None 则不保存）
SAVE_PATH = os.path.join(_SCRIPT_DIR, "dfcd_2020_full.pt")

# 覆盖为 2020 + 100% 训练 + 外部测试集
run_config.CONFIG.update({
    "data_type": "2020",
    "data_root": _SCRIPT_DIR,
    "test_size": 0.0,            # 不从训练集划分
    "split": "Original",
    "text_embedding_model": "remote",
    "test_csv": "TotalData_test.csv",  # 外部测试集（在 data/2020/ 下）
    "epoch": 20,
    "lr": 1e-4,
    "batch_size": 512,
    "seed": 0,
    "device": "cuda:0",
    "dtype": torch.float32,
    "save_path": SAVE_PATH,
})


def train_with_test_loop(model, config):
    """训练 + 每 epoch 评估（单卡）。有测试集时调 test_step，无则只打印 loss。"""
    has_test = config.get("np_test") is not None and len(config["np_test"]) > 0
    all_epoch = config["epoch"]
    model.train()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
    )
    best_auc = 0
    for epoch_i in range(all_epoch):
        model.train()
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
        avg_loss = np.mean(epoch_losses)

        if has_test:
            auc, ap, acc, rmse, f1, doa = model.test_step()
            print(f"[{epoch_i:03d}/{all_epoch}] | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | "
                  f"ACC: {acc:.4f} | RMSE: {rmse:.4f} | F1: {f1:.4f} | DOA@10: {doa:.4f}")
            if auc > best_auc:
                best_auc = auc
        else:
            print(f"[{epoch_i:03d}/{all_epoch}] Loss: {avg_loss:.4f}")

    save_path = config.get("save_path")
    if save_path:
        torch.save(model.state_dict(), save_path)
        print("模型已保存:", os.path.abspath(save_path))
    if has_test:
        print(f"Best AUC: {best_auc:.4f}")
    print("训练完成。")


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
        train_with_test_loop(model, config)
    else:
        import torch.multiprocessing as mp
        config = run_config._build_config()
        mp.spawn(_ddp_entry, nprocs=n_gpus, args=(n_gpus, config), join=True)
