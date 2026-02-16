"""
base.py — 基础模型类
功能：
  1. 定义模型抽象接口 (forward, get_mastery_level, monotonicity)
  2. 模块化训练: train_one_epoch
  3. 统一 4 模式测试: test_all_modes
  4. 指标计算: compute_metrics
  5. 模型保存: save_model
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_squared_error, f1_score
from utils import get_doa


def get_raw_model(model):
    """获取原始模型，若被 DataParallel 包裹则解包"""
    return model.module if isinstance(model, nn.DataParallel) else model


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def forward(self, student_id, exercise_id, knowledge_point, mode='train'):
        raise NotImplementedError

    def get_mastery_level(self, mode='eval'):
        raise NotImplementedError

    def monotonicity(self):
        raise NotImplementedError

    # ===================================================================
    # 训练一个 epoch
    # ===================================================================
    @staticmethod
    def train_one_epoch(model, optimizer, train_dataloader, device):
        """
        训练一个 epoch。

        流程：
          1. 遍历 train_dataloader 的每个 batch
          2. 前向传播 → BCELoss
          3. 反向传播 → 参数更新
          4. 单调性约束 (monotonicity)

        返回: 平均 loss
        """
        model.train()
        epoch_losses = []
        raw_model = get_raw_model(model)

        for batch_data in tqdm(train_dataloader, desc="Training"):
            # ---- 数据搬运到设备 ----
            student_id, exercise_id, knowledge_point, y = [
                data.to(device) for data in batch_data
            ]

            # ---- 前向传播 ----
            pred = model(student_id, exercise_id, knowledge_point, mode='train')
            loss = nn.BCELoss()(pred, y)

            # ---- 反向传播 ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- 单调性约束: 裁剪 decoder 权重为非负 ----
            raw_model.monotonicity()

            epoch_losses.append(loss.item())

        return np.mean(epoch_losses)

    # ===================================================================
    # 统一 4 模式测试
    # ===================================================================
    @staticmethod
    @torch.no_grad()
    def test_all_modes(model, test_dataloader, splits_info, config, device):
        """
        在同一个测试集上评估 4 种模式。

        模式说明:
          - Overall: 所有测试样本
          - Stu:     仅陌生学生的预测
          - Exer:    仅陌生习题的预测
          - Know:    仅陌生知识点关联习题的预测

        流程：
          1. 遍历 test_dataloader，获取所有预测
          2. 按模式分类收集 (pred, label)
          3. 分别计算各模式指标
          4. DOA 统一计算一次

        返回: dict[mode_name] -> metrics_dict
        """
        model.eval()
        raw_model = get_raw_model(model)

        new_stu_set = splits_info['new_stu_set']
        new_exer_set = splits_info['new_exer_set']
        new_know_exer_set = splits_info['new_know_exer_set']

        # ---- 按模式收集预测和标签 ----
        collectors = {
            'Overall': {'preds': [], 'ys': []},
            'Stu':     {'preds': [], 'ys': []},
            'Exer':    {'preds': [], 'ys': []},
            'Know':    {'preds': [], 'ys': []},
        }

        for batch_data in tqdm(test_dataloader, desc="Testing"):
            student_id, exercise_id, knowledge_point, y = [
                data.to(device) for data in batch_data
            ]

            # ---- 前向传播（eval 模式使用 full_data 图） ----
            pred = model(student_id, exercise_id, knowledge_point, mode='eval')
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            stu_np = student_id.cpu().numpy()
            exer_np = exercise_id.cpu().numpy()

            for i in range(len(pred_np)):
                p, label = float(pred_np[i]), float(y_np[i])
                sid, eid = int(stu_np[i]), int(exer_np[i])

                # Overall: 所有样本
                collectors['Overall']['preds'].append(p)
                collectors['Overall']['ys'].append(label)

                # Stu: 陌生学生
                if sid in new_stu_set:
                    collectors['Stu']['preds'].append(p)
                    collectors['Stu']['ys'].append(label)

                # Exer: 陌生习题
                if eid in new_exer_set:
                    collectors['Exer']['preds'].append(p)
                    collectors['Exer']['ys'].append(label)

                # Know: 陌生知识点关联习题
                if eid in new_know_exer_set:
                    collectors['Know']['preds'].append(p)
                    collectors['Know']['ys'].append(label)

        # ---- DOA: 统一计算一次（基于全局 mastery level 和 r_matrix） ----
        mastery_level = raw_model.get_mastery_level(mode='eval')
        doa = get_doa(config, mastery_level)

        # ---- 分模式计算指标 ----
        results = {}
        for mode_name, data in collectors.items():
            if len(data['preds']) < 2:
                print(f"  [WARNING] {mode_name} 模式测试样本不足 ({len(data['preds'])}), 跳过")
                results[mode_name] = None
                continue
            metrics = BaseModel.compute_metrics(data['ys'], data['preds'], doa)
            results[mode_name] = metrics

        return results

    # ===================================================================
    # 指标计算
    # ===================================================================
    @staticmethod
    def compute_metrics(ys, preds, doa=0.0):
        """
        计算单个模式的评估指标。

        返回: dict with keys: auc, ap, acc, rmse, f1, doa
        """
        ys_arr = np.array(ys)
        preds_arr = np.array(preds)
        return {
            'auc':  roc_auc_score(ys_arr, preds_arr),
            'ap':   average_precision_score(ys_arr, preds_arr),
            'acc':  accuracy_score(ys_arr, preds_arr >= 0.5),
            'rmse': np.sqrt(mean_squared_error(ys_arr, preds_arr)),
            'f1':   f1_score(ys_arr, preds_arr >= 0.5),
            'doa':  doa,
        }

    # ===================================================================
    # 模型保存 / 加载
    # ===================================================================
    @staticmethod
    def save_model(model, optimizer, epoch, save_path, config=None):
        """
        保存模型参数、优化器状态、epoch 信息。

        参数:
          model:     模型（可能被 DataParallel 包裹）
          optimizer: 优化器
          epoch:     当前 epoch
          save_path: 保存路径
          config:    可选配置信息
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        raw_model = get_raw_model(model)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if config is not None:
            # 只保存可序列化的配置项
            save_config = {k: v for k, v in config.items()
                          if isinstance(v, (int, float, str, bool, list, tuple, dict))}
            checkpoint['config'] = save_config
        torch.save(checkpoint, save_path)
        print(f"[save_model] 模型已保存: {save_path}")

    @staticmethod
    def load_model(model, save_path, optimizer=None):
        """加载模型参数"""
        checkpoint = torch.load(save_path, map_location='cpu')
        raw_model = get_raw_model(model)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[load_model] 模型已加载: {save_path} (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint.get('epoch', 0)
