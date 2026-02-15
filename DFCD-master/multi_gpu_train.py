# -*- coding: utf-8 -*-
"""
4 路 3090 分布式训练（DDP）模块。
由 run_train_2020.py 在检测到多 GPU 时调用，使用 torch.distributed 实现数据并行。
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    mean_squared_error,
    f1_score,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "models"))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def _make_ddp_dataloader(config, world_size):
    """用 DistributedSampler 替换 train_dataloader。"""
    old_loader = config["train_dataloader"]
    dataset = old_loader.dataset
    batch_size = config["batch_size"]
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )


def train_epoch_ddp(model, optimizer, config, epoch):
    model.train()
    loader = config["train_dataloader"]
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    device = config["device"]
    epoch_losses = []
    desc = f"Epoch {epoch}" if dist.get_rank() == 0 else None
    for batch_data in tqdm(loader, desc=desc, disable=(dist.get_rank() != 0)):
        student_id, exercise_id, knowledge_point, y = [
            d.to(device, non_blocking=True) for d in batch_data
        ]
        pred = model(student_id, exercise_id, knowledge_point)
        loss = nn.BCELoss()(pred, y.to(pred.dtype))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.module.monotonicity()
        epoch_losses.append(loss.item())
    return np.mean(epoch_losses)


@torch.no_grad()
def test_step_ddp(model, config):
    """仅 rank 0 执行测试。"""
    if dist.get_rank() != 0:
        return None
    from utils import get_doa

    model.eval()
    new_preds, preds, new_ys, ys = [], [], [], []
    loader = config["test_dataloader"]
    device = config["device"]
    for batch_data in tqdm(loader, desc="Testing"):
        student_id, exercise_id, knowledge_point, y = [
            d.to(device) for d in batch_data
        ]
        pred = model(student_id, exercise_id, knowledge_point, mode="eval")
        pred = pred.cpu().numpy().tolist()
        new_pred, new_y = [], []
        if config["split"] == "Stu":
            for idx, student in enumerate(student_id):
                if student.detach().cpu().numpy() in config["new_idx"]:
                    new_pred.append(pred[idx])
                    new_y.append(y.cpu().numpy().tolist()[idx])
        elif config["split"] in ("Exer", "Know"):
            for idx, exercise in enumerate(exercise_id):
                if exercise.detach().cpu().numpy() in config["new_idx"]:
                    new_pred.append(pred[idx])
                    new_y.append(y.cpu().numpy().tolist()[idx])
        preds.extend(pred)
        ys.extend(y.cpu().numpy().tolist())
        new_preds.extend(new_pred)
        new_ys.extend(new_y)

    if config["split"] in ("Stu", "Exer", "Know"):
        auc = roc_auc_score(new_ys, new_preds)
        ap = average_precision_score(new_ys, new_preds)
        acc = accuracy_score(new_ys, np.array(new_preds) >= 0.5)
        rmse = np.sqrt(mean_squared_error(new_ys, new_preds))
        f1 = f1_score(new_ys, np.array(new_preds) >= 0.5)
        doa = get_doa(config, model.module.get_mastery_level())
    else:
        auc = roc_auc_score(ys, preds)
        ap = average_precision_score(ys, preds)
        acc = accuracy_score(ys, np.array(preds) >= 0.5)
        rmse = np.sqrt(mean_squared_error(ys, preds))
        f1 = f1_score(ys, np.array(preds) >= 0.5)
        doa = get_doa(config, model.module.get_mastery_level())
    return auc, ap, acc, rmse, f1, doa


def main_ddp(rank, world_size, config_template):
    """
    单进程入口：rank 为当前进程 GPU 编号，world_size 为总 GPU 数。
    config_template: 从 run_config 构建的 config 字典（不含 device，此处会覆盖）。
    """
    setup(rank, world_size)
    config = dict(config_template)
    config["device"] = f"cuda:{rank}"

    # 多卡时线性缩放学习率
    base_lr = config.get("lr", 1e-4)
    config["lr"] = base_lr * world_size

    from utils import load_data, construct_data_geometric
    from models.dfcd import DFCD

    load_data(config)
    config["in_channels_init"] = (
        config["stu_num"] + config["prob_num"] + config["know_num"]
    )

    if config["split"] in ("Stu", "Exer"):
        train_data = construct_data_geometric(config, data=config["np_train_old"])
        full_data = construct_data_geometric(config, data=config["np_train"])
    else:
        train_data = construct_data_geometric(config, data=config["np_train"])
        full_data = train_data

    config["train_data"] = train_data.to(config["device"])
    config["full_data"] = full_data.to(config["device"])

    config["train_dataloader"] = _make_ddp_dataloader(config, world_size)

    model = DFCD(config)
    model = model.to(config["device"])
    # GNN 中 TransformerConv 等使用 in_channels=-1 延迟初始化，DDP 要求参数已初始化，先跑一次 dummy forward
    model.train()
    with torch.no_grad():
        one_batch = next(iter(config["train_dataloader"]))
        student_id, exercise_id, knowledge_point, _ = [t.to(config["device"]) for t in one_batch]
        _ = model(student_id, exercise_id, knowledge_point)
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
    )

    n_epochs = config["epoch"]
    total_auc, total_ap, total_acc, total_rmse, total_f1, total_doa = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    if rank == 0:
        print(f"[DDP] 使用 {world_size} 张 GPU 训练，每卡 batch_size={config['batch_size']}")

    has_test = config.get("np_test") is not None and len(config["np_test"]) > 0

    for epoch in range(n_epochs):
        loss = train_epoch_ddp(model, optimizer, config, epoch)
        dist.barrier()

        if has_test:
            metrics = test_step_ddp(model, config)
            if rank == 0 and metrics is not None:
                auc, ap, acc, rmse, f1, doa = metrics
                total_auc.append(auc)
                total_ap.append(ap)
                total_acc.append(acc)
                total_rmse.append(rmse)
                total_f1.append(f1)
                total_doa.append(doa)
                print(
                    f"[{epoch:03d}/{n_epochs}] | Loss: {loss:.4f} | AUC: {auc:.4f} | "
                    f"ACC: {acc:.4f} | RMSE: {rmse:.4f} | F1: {f1:.4f} | DOA@10: {doa:.4f}"
                )
        elif rank == 0:
            print(f"[{epoch:03d}/{n_epochs}] Loss: {loss:.4f}")

    if rank == 0 and has_test and total_auc:
        print(
            f"Best AUC: {max(total_auc)}, Best AP: {max(total_ap)}, "
            f"Best ACC: {max(total_acc)}, Best RMSE: {min(total_rmse)}, "
            f"Best F1: {max(total_f1)}, Best DOA: {max(total_doa)}"
        )
    elif rank == 0 and not has_test:
        print("训练完成。请使用外部验证集进行评估。")

    cleanup()
