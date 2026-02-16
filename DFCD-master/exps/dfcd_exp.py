"""
dfcd_exp.py — DFCD 实验入口
功能：
  1. 统一数据加载与 4 模式划分
  2. 构建图数据 (训练图 + 完整图)
  3. 多 GPU 支持 (DataParallel)
  4. 训练循环: 每个 epoch 训练 + 4 模式测试
  5. 模型保存

用法:
  # 单卡
  python dfcd_exp.py --gpus 0 --data_type 2020
  # 多卡
  python dfcd_exp.py --gpus 0,1,2 --data_type 2020
  # 指定嵌入模型
  python dfcd_exp.py --gpus 0 --text_embedding_model remote
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pprint import pprint
from datetime import datetime


# ===========================================================================
# 路径设置
# ===========================================================================
def import_paths():
    import warnings
    warnings.filterwarnings("ignore")
    current_path = os.path.abspath('.')
    tmp = os.path.dirname(current_path)
    sys.path.insert(0, tmp)
    sys.path.insert(0, tmp + '/models')


import_paths()

from models.dfcd import DFCD
from models.base import BaseModel, get_raw_model
from utils import load_data_unified, set_common_args, construct_data_geometric

# ===========================================================================
# 嵌入维度映射
# ===========================================================================
EMBEDDING_DIM_MAP = {
    'openai': 1536,
    'BAAI': 1024,
    'm3e': 768,
    'instructor': 768,
    'remote': 768,  # gte-multilingual-base
}


# ===========================================================================
# Lazy GNN 初始化（DataParallel 前必须完成，否则 UninitializedParameter 报错）
# ===========================================================================
def _init_lazy_gnn(model, train_dataloader, device):
    """
    Transformer/GAT 等 GNN 使用 in_channels=-1 懒初始化，需 dummy forward 后才可被 DataParallel 复制。
    """
    model.train()
    batch = next(iter(train_dataloader))
    student_id, exercise_id, knowledge_point, _ = [d.to(device) for d in batch]
    with torch.no_grad():
        _ = model(student_id, exercise_id, knowledge_point, mode='train')


# ===========================================================================
# 多 GPU 设置
# ===========================================================================
def setup_device_and_model(model, gpu_str, train_dataloader=None):
    """
    根据 --gpus 参数配置设备与 DataParallel。

    参数:
      model:   原始模型
      gpu_str: GPU 编号字符串，如 "0" 或 "0,1,2"
      train_dataloader: 用于懒初始化 GNN 的 dummy forward，多 GPU 时必传

    返回:
      model:   可能被 DataParallel 包裹的模型
      device:  主设备 (primary GPU)
    """
    gpu_ids = [int(g.strip()) for g in gpu_str.split(',')]
    if not torch.cuda.is_available():
        print("[setup] CUDA 不可用，使用 CPU")
        return model, torch.device('cpu')

    primary_device = torch.device(f'cuda:{gpu_ids[0]}')
    torch.cuda.set_device(primary_device)
    model = model.to(primary_device)

    # 多 GPU 时，DataParallel 会 replicate 模型，懒初始化 GNN 需在此前完成
    if len(gpu_ids) > 1:
        if train_dataloader is None:
            raise ValueError("多 GPU 模式下需传入 train_dataloader 以初始化懒 GNN 层")
        _init_lazy_gnn(model, train_dataloader, primary_device)
        print(f"[setup] 多 GPU 模式: {gpu_ids} (primary: cuda:{gpu_ids[0]})")
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    else:
        print(f"[setup] 单 GPU 模式: cuda:{gpu_ids[0]}")

    return model, primary_device


# ===========================================================================
# 打印测试结果
# ===========================================================================
def print_epoch_results(epoch, total_epochs, loss, results):
    """格式化打印每个 epoch 的训练 loss 和 4 模式测试指标"""
    print(f"\n{'=' * 80}")
    print(f"[Epoch {epoch:03d}/{total_epochs}] Loss: {loss:.4f}")
    print(f"{'-' * 80}")
    for mode_name in ['Overall', 'Stu', 'Exer', 'Know']:
        metrics = results.get(mode_name)
        if metrics is None:
            print(f"  {mode_name:8s} | 样本不足, 跳过")
        else:
            print(f"  {mode_name:8s} | AUC: {metrics['auc']:.4f} | "
                  f"ACC: {metrics['acc']:.4f} | RMSE: {metrics['rmse']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | DOA@10: {metrics['doa']:.4f}")
    print(f"{'=' * 80}\n")


# ===========================================================================
# 打印最终汇总
# ===========================================================================
def print_best_results(history):
    """打印所有 epoch 中各模式的最佳指标"""
    print(f"\n{'#' * 80}")
    print(f"训练结束 — 各模式最佳指标:")
    print(f"{'#' * 80}")
    for mode_name in ['Overall', 'Stu', 'Exer', 'Know']:
        mode_history = [h[mode_name] for h in history if h.get(mode_name) is not None]
        if not mode_history:
            print(f"  {mode_name:8s} | 无有效结果")
            continue
        best_auc = max(m['auc'] for m in mode_history)
        best_acc = max(m['acc'] for m in mode_history)
        best_rmse = min(m['rmse'] for m in mode_history)
        best_f1 = max(m['f1'] for m in mode_history)
        best_doa = max(m['doa'] for m in mode_history)
        print(f"  {mode_name:8s} | Best AUC: {best_auc:.4f} | "
              f"Best ACC: {best_acc:.4f} | Best RMSE: {best_rmse:.4f} | "
              f"Best F1: {best_f1:.4f} | Best DOA: {best_doa:.4f}")
    print(f"{'#' * 80}\n")


# ===========================================================================
# 主函数
# ===========================================================================
def main(config):
    # ================================================================
    # Step 1: 统一数据加载与 4 模式划分
    # ================================================================
    print("[main] Step 1: 数据加载与统一划分")
    load_data_unified(config)

    # ================================================================
    # Step 2: 设置嵌入维度
    # ================================================================
    emb_model = config['text_embedding_model']
    if emb_model in EMBEDDING_DIM_MAP:
        config['in_channels_llm'] = EMBEDDING_DIM_MAP[emb_model]
    else:
        raise ValueError(f"未知嵌入模型: {emb_model}, 请在 EMBEDDING_DIM_MAP 中添加维度映射")
    config['in_channels_init'] = config['stu_num'] + config['prob_num'] + config['know_num']
    print(f"[main] Step 2: 嵌入维度 — LLM: {config['in_channels_llm']}, "
          f"Init: {config['in_channels_init']}")

    # ================================================================
    # Step 3: 构建图数据
    # ================================================================
    print("[main] Step 3: 构建图数据")
    # 训练图: 仅含已知实体的边和交互
    train_data = construct_data_geometric(config, data=config['np_train_final'])
    # 完整图: 含所有实体的边和交互（eval 时使用，使新实体可通过图获得表示）
    full_data = construct_data_geometric(config, data=config['np_train'])
    print(f"  训练图 — 节点: {train_data.x_llm.shape[0]}, 边: {train_data.edge_index.shape[1]}")
    print(f"  完整图 — 节点: {full_data.x_llm.shape[0]}, 边: {full_data.edge_index.shape[1]}")

    # ================================================================
    # Step 4: 创建模型
    # ================================================================
    print("[main] Step 4: 创建 DFCD 模型")
    # 先在主设备上创建模型
    primary_gpu = int(config['gpus'].split(',')[0]) if torch.cuda.is_available() else 0
    config['device'] = f'cuda:{primary_gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(config['device'])

    model = DFCD(config)

    # 注册图数据为 buffer (支持多 GPU 自动复制)
    model.set_graph_data(
        train_data.to(device),
        full_data.to(device)
    )

    # ================================================================
    # Step 5: 多 GPU 配置（多卡时需先 dummy forward 初始化懒 GNN 层）
    # ================================================================
    print(f"[main] Step 5: GPU 配置")
    model, device = setup_device_and_model(model, config['gpus'], config['train_dataloader'])

    # ================================================================
    # Step 6: 优化器
    # ================================================================
    raw_model = get_raw_model(model)
    optimizer = torch.optim.Adam(
        raw_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    print(f"[main] Step 6: 优化器 — Adam(lr={config['lr']}, wd={config['weight_decay']})")

    # ================================================================
    # Step 7: 训练循环 — 每个 epoch 训练 + 4 模式测试
    # ================================================================
    print(f"[main] Step 7: 开始训练 — 共 {config['epoch']} 个 epoch\n")
    history = []

    for epoch_i in range(1, config['epoch'] + 1):
        # ---- 训练 ----
        loss = BaseModel.train_one_epoch(
            model, optimizer, config['train_dataloader'], device
        )

        # ---- 测试 (4 模式) ----
        results = BaseModel.test_all_modes(
            model, config['test_dataloader'], config['splits_info'], config, device
        )
        history.append(results)

        # ---- 打印结果 ----
        print_epoch_results(epoch_i, config['epoch'], loss, results)

    # ================================================================
    # Step 8: 打印最佳结果
    # ================================================================
    print_best_results(history)

    # ================================================================
    # Step 9: 保存最终模型
    # ================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(
        config['save_dir'],
        config['name'],
        f"model_epoch{config['epoch']}_{timestamp}.pt"
    )
    BaseModel.save_model(model, optimizer, config['epoch'], save_path, config)

    print("[main] 全部完成!")
    return history


# ===========================================================================
# 入口
# ===========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFCD 训练实验')
    parser.add_argument('--encoder_type', default='transformer', type=str,
                        help='GNN 编码器类型: gcn, gat, gatv2, transformer')
    parser.add_argument('--decoder_type', default='simplecd', type=str,
                        help='解码器类型: simplecd, kancd, ncd')
    parser.add_argument('--out_channels', default=128, type=int,
                        help='编码器输出维度')
    parser.add_argument('--mode', default=2, type=int,
                        help='特征模式: 0=仅响应, 1=仅文本, 2=混合')
    set_common_args(parser)

    config_dict = vars(parser.parse_args())

    # ---- 实验名称 ----
    mode_suffix = {0: '-response', 1: '-text', 2: '-hybrid'}.get(config_dict['mode'], '')
    config_dict['name'] = (f"{config_dict['method']}{mode_suffix}-"
                           f"{config_dict['data_type']}-seed{config_dict['seed']}")

    # ---- 打印配置 ----
    print("\n" + "=" * 80)
    print("实验配置:")
    print("=" * 80)
    pprint(config_dict)
    print("=" * 80 + "\n")

    # ---- 执行 ----
    main(config_dict)
