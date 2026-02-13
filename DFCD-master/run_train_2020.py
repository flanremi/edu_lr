# -*- coding: utf-8 -*-
"""
2020 数据集训练入口：按学生划分 85% 训练 / 15% 测试，使用 embedding_remote.pkl。

数据目录：data/2020/（TotalData.csv、q.csv、embedding_remote.pkl）
运行方式：在 DFCD-master 根目录下执行
  python run_train_2020.py

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
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'models'))

# 复用 run_config 的完整流程，仅覆盖 2020 相关配置
import run_config

# 覆盖为 2020 数据集 + 按学生 85/15 划分 + remote 嵌入
run_config.CONFIG.update({
    "data_type": "2020",
    "data_root": _SCRIPT_DIR,
    "test_size": 0.15,           # 15% 学生作为测试集
    "split": "Stu",              # 按学生划分：85% 学生训练，15% 学生测试
    "text_embedding_model": "remote",  # 使用 data/2020/embedding_remote.pkl
    "epoch": 20,
    "lr": 1e-4,
    "batch_size": 1024,
    "seed": 0,
    "device": "cuda:0",
})

if __name__ == "__main__":
    run_config.main()
