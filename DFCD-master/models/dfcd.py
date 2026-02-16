"""
dfcd.py — DFCD 主模型
功能：
  1. 多模态编码器 (LLM嵌入 + 初始化特征 → 注意力融合 → GNN)
  2. 认知诊断解码器
  3. 使用 register_buffer 存储图数据，支持多 GPU (DataParallel)
  4. 训练/测试模式切换 (train_data vs full_data)
"""

import torch
import numpy as np
from base import BaseModel
from utils import Weighted_Summation
from decoders import get_decoder, get_mlp_encoder, GNNEncoder


class DFCD(BaseModel):
    def __init__(self, config):
        super(DFCD, self).__init__(config)
        device = config['device']

        # ---- GNN 编码器 ----
        self.encoder_GNN = GNNEncoder(
            layer=config['encoder_type'],
            in_channels=config['out_channels'],
            hidden_channels=config['out_channels'],
            out_channels=config['out_channels']
        ).to(device)

        # ---- 注意力融合模块 (用于 mode=2 混合模式) ----
        self.attn_S = Weighted_Summation(config['out_channels'], attn_drop=0.2).to(device)
        self.attn_E = Weighted_Summation(config['out_channels'], attn_drop=0.2).to(device)
        self.attn_K = Weighted_Summation(config['out_channels'], attn_drop=0.2).to(device)

        # ---- LLM 嵌入编码器: 将文本嵌入映射到统一维度 ----
        self.encoder_student_llm = get_mlp_encoder(
            in_channels=config['in_channels_llm'], out_channels=config['out_channels']
        ).to(device)
        self.encoder_exercise_llm = get_mlp_encoder(
            in_channels=config['in_channels_llm'], out_channels=config['out_channels']
        ).to(device)
        self.encoder_knowledge_llm = get_mlp_encoder(
            in_channels=config['in_channels_llm'], out_channels=config['out_channels']
        ).to(device)

        # ---- 初始化特征编码器: 将答题邻接特征映射到统一维度 ----
        self.encoder_student_init = get_mlp_encoder(
            in_channels=config['in_channels_init'], out_channels=config['out_channels']
        ).to(device)
        self.encoder_exercise_init = get_mlp_encoder(
            in_channels=config['in_channels_init'], out_channels=config['out_channels']
        ).to(device)
        self.encoder_knowledge_init = get_mlp_encoder(
            in_channels=config['in_channels_init'], out_channels=config['out_channels']
        ).to(device)

        # ---- 解码器 ----
        self.decoder = get_decoder(config).to(device)

    # ===================================================================
    # 图数据注册 — 使用 register_buffer 以支持 DataParallel 自动复制
    # ===================================================================
    def set_graph_data(self, train_data, full_data):
        """
        将图数据注册为 buffer，DataParallel 会自动将 buffer 复制到每张 GPU。

        参数:
          train_data: 训练图 (仅含已知实体的边和 x_init)
          full_data:  完整图 (含所有实体的边和 x_init，用于 eval)
        """
        # ---- 训练图 ----
        self.register_buffer('train_x_llm', train_data.x_llm)
        self.register_buffer('train_x_init', train_data.x_init)
        self.register_buffer('train_edge_index', train_data.edge_index)
        # ---- 完整图 (eval 用) ----
        self.register_buffer('full_x_llm', full_data.x_llm)
        self.register_buffer('full_x_init', full_data.x_init)
        self.register_buffer('full_edge_index', full_data.edge_index)

    # ===================================================================
    # 获取图数据 — 根据 mode 选择训练图或完整图
    # ===================================================================
    def get_data(self, mode='train'):
        """
        mode='train': 使用训练图（不含陌生实体的边/x_init）
        mode='eval':  使用完整图（含所有实体，用于冷启动预测）
        """
        if mode == 'train':
            return self.train_x_llm, self.train_x_init, self.train_edge_index
        else:
            return self.full_x_llm, self.full_x_init, self.full_edge_index

    # ===================================================================
    # 节点掩码 — 训练时随机 mask 部分节点以增强泛化
    # ===================================================================
    def mask_nodes(self, x_init, ratio=0.2):
        total_rows = self.config['stu_num'] + self.config['prob_num'] + self.config['know_num']
        mask_rows = np.random.choice(total_rows, int(ratio * total_rows), replace=False)
        x_init[mask_rows] = 0
        return x_init, mask_rows

    # ===================================================================
    # 编码器: LLM / Init / 混合 → GNN
    # ===================================================================
    def get_x(self, x_llm, x_init, edge_index):
        """
        根据 config['mode'] 选择编码方式:
          mode=0: 仅使用 x_init (响应特征)
          mode=1: 仅使用 x_llm (文本嵌入)
          mode=2: 混合模式 — 注意力融合 init + llm
        """
        stu_num = self.config['stu_num']
        prob_num = self.config['prob_num']

        if self.config['mode'] == 0:
            # ---- 仅响应特征 ----
            student_factor = self.encoder_student_init(x_init[:stu_num])
            exercise_factor = self.encoder_exercise_init(x_init[stu_num:stu_num + prob_num])
            knowledge_factor = self.encoder_knowledge_init(x_init[stu_num + prob_num:])

        elif self.config['mode'] == 1:
            # ---- 仅文本嵌入 ----
            student_factor = self.encoder_student_llm(x_llm[:stu_num])
            exercise_factor = self.encoder_exercise_llm(x_llm[stu_num:stu_num + prob_num])
            knowledge_factor = self.encoder_knowledge_llm(x_llm[stu_num + prob_num:])

        elif self.config['mode'] == 2:
            # ---- 混合模式: init + llm → 注意力融合 ----
            student_factor_init = self.encoder_student_init(x_init[:stu_num])
            exercise_factor_init = self.encoder_exercise_init(x_init[stu_num:stu_num + prob_num])
            knowledge_factor_init = self.encoder_knowledge_init(x_init[stu_num + prob_num:])

            student_factor_llm = self.encoder_student_llm(x_llm[:stu_num])
            exercise_factor_llm = self.encoder_exercise_llm(x_llm[stu_num:stu_num + prob_num])
            knowledge_factor_llm = self.encoder_knowledge_llm(x_llm[stu_num + prob_num:])

            student_factor = self.attn_S([student_factor_init, student_factor_llm])
            exercise_factor = self.attn_E([exercise_factor_init, exercise_factor_llm])
            knowledge_factor = self.attn_K([knowledge_factor_init, knowledge_factor_llm])

        # ---- 拼接全部节点表示 → GNN 编码 ----
        final_x = torch.cat([student_factor, exercise_factor, knowledge_factor], dim=0)

        if self.training:
            x_mask, mask_rows = self.mask_nodes(final_x, ratio=0.2)
            return self.encoder_GNN.forward(x_mask, edge_index), mask_rows
        else:
            return self.encoder_GNN.forward(final_x, edge_index), None

    # ===================================================================
    # 前向传播
    # ===================================================================
    def forward(self, student_id, exercise_id, knowledge_point, mode='train'):
        """
        完整前向传播: 图数据 → 编码 → 解码 → 预测

        参数:
          student_id:      学生 ID (batch,)
          exercise_id:     习题 ID (batch,)
          knowledge_point: Q矩阵行 (batch, know_num)
          mode:            'train' 使用训练图, 'eval' 使用完整图
        """
        x_llm, x_init, edge_index = self.get_data(mode)
        rep, _ = self.get_x(x_llm, x_init, edge_index)
        return self.decoder.forward(rep, student_id, exercise_id, knowledge_point)

    # ===================================================================
    # 获取 mastery level (用于 DOA 评估)
    # ===================================================================
    def get_mastery_level(self, mode='eval'):
        x_llm, x_init, edge_index = self.get_data(mode)
        rep, _ = self.get_x(x_llm, x_init, edge_index)
        return self.decoder.get_mastery_level(rep)

    # ===================================================================
    # 单调性约束
    # ===================================================================
    def monotonicity(self):
        self.decoder.monotonicity()
