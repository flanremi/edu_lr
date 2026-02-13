# -*- coding: utf-8 -*-
"""
DFCD 可配置训练/测试脚本
- 所有需要修改的变量集中在下方 CONFIG 中，并配有中文注释
- 支持：引入训练集训练、使用部分数据做测试集
"""
import os
import sys
import warnings
from pprint import pprint

# 将项目根目录加入路径，便于从任意位置运行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'models'))
warnings.filterwarnings("ignore")

# =============================================================================
# 【请在此修改】所有可配置变量（修改后直接运行本脚本即可）
# =============================================================================

CONFIG = {
    # ----- 数据相关 -----
    # 数据集名称，需与 data/<名称> 目录及 data_params_dict.py 中一致
    "data_type": "XES3G5M",

    # 项目根目录（一般无需改，用于定位 data/ 下的数据；从项目根运行则用脚本所在目录）
    "data_root": _SCRIPT_DIR,

    # 测试集比例 (0~1)，例如 0.2 表示 20% 作为测试集，80% 训练
    "test_size": 0.2,

    # 划分方式：Original=标准随机划分 | Stu=未见学生 | Exer=未见题目 | Know=未见知识点
    "split": "Original",

    # ----- 训练超参数 -----
    "epoch": 20,
    "lr": 1e-4,
    "batch_size": 1024,
    "weight_decay": 0,
    "seed": 0,

    # ----- 设备与精度 -----
    "device": "cuda:0",
    "dtype": None,  # 保持 None 时使用 torch.float64

    # ----- 模型结构 -----
    "encoder_type": "transformer",  # GNN 类型: transformer | gcn | gat | gatv2
    "decoder_type": "simplecd",    # 解码器: simplecd | kancd | ncd
    "out_channels": 128,           # 隐层维度
    "mode": 2,  # 0=仅答题行为 1=仅文本 2=双融合(hybrid)

    # ----- 文本嵌入模型（需与已生成的 embedding_*.pkl 对应） -----
    "text_embedding_model": "openai",  # openai | BAAI | m3e | instructor | remote(2020)
}

# 若未设置 dtype，使用 float64（与原文一致）
if CONFIG.get("dtype") is None:
    import torch
    CONFIG["dtype"] = torch.float64

# =============================================================================
# 完整实行方案说明（如何引入训练集、如何使用部分数据测试）
# =============================================================================
"""
【一、准备数据】
  1. 在 data/<data_type>/ 下放置：
     - TotalData.csv（学生ID, 题目ID, 0/1得分，无表头）
     - q.csv（题目数×知识点数 的 0/1 矩阵，无表头）
  2. 在 data/data_params_dict.py 中为该 data_type 添加/修改：
     stu_num, prob_num, know_num, batch_size
  3. 生成文本嵌入：进入 data_preprocess，运行
     python main_embedding.py --dataset <data_type> --llm <OpenAI|BAAI|m3e|Instructor>
     得到 data/<data_type>/embedding_<小写>.pkl（如 embedding_openai.pkl）
     CONFIG 中 text_embedding_model 需与文件名一致（小写：openai, BAAI, m3e, instructor）

【二、引入训练集进行训练】
  - 直接运行本脚本：python run_config.py
  - 脚本会按 CONFIG["test_size"] 将总答题记录划分为训练集与测试集
  - 训练集用于更新模型，每个 epoch 结束后在测试集上计算 AUC/ACC/RMSE/F1/DOA

【三、使用部分数据做测试集】
  - 调大 test_size 即可用“更少数据训练、更多数据测试”，例如 test_size=0.3 或 0.4
  - 或希望“只用部分数据做训练”：
    先自行从 TotalData.csv 中采样/截取一份子集保存为新 CSV，在 data_params_dict 中
    可保持不变（仍用全集维度），但将 data/<data_type>/TotalData.csv 临时替换为该子集；
    或复制一份新数据集目录，子集 TotalData 放入新目录并修改 CONFIG["data_type"] 与 data_params_dict
  - 划分方式 split：
    - Original：按条随机划分，常用
    - Stu/Exer/Know：按“未见学生/题目/知识点”划分，评估时只报告未见部分指标
"""


def _build_config():
    """从 CONFIG 构造完整 config（补全 name、in_channels 等）。"""
    import torch
    c = dict(CONFIG)
    c["method"] = "dfcd"
    c["name"] = f"dfcd-{c['data_type']}-seed{c['seed']}"
    if c["mode"] == 1:
        c["method"] = c["method"] + "-text"
    elif c["mode"] == 2:
        c["method"] = c["method"] + "-hybrid"
    else:
        c["method"] = c["method"] + "-response"
    if c["text_embedding_model"] == "openai":
        c["in_channels_llm"] = 1536
    elif c["text_embedding_model"] == "BAAI":
        c["in_channels_llm"] = 1024
    elif c["text_embedding_model"] in ("m3e", "instructor", "remote"):
        c["in_channels_llm"] = 768
    else:
        raise ValueError("未支持的 text_embedding_model: " + str(c["text_embedding_model"]))
    return c


def main():
    import torch
    from utils import load_data, set_common_args, construct_data_geometric
    from models.dfcd import DFCD

    config = _build_config()
    # 从 data_params 得到 stu_num, prob_num, know_num 后再算 in_channels_init
    load_data(config)
    config["in_channels_init"] = config["stu_num"] + config["prob_num"] + config["know_num"]

    if config["split"] in ("Stu", "Exer"):
        train_data = construct_data_geometric(config, data=config["np_train_old"])
        full_data = construct_data_geometric(config, data=config["np_train"])
    else:
        train_data = construct_data_geometric(config, data=config["np_train"])
        full_data = train_data

    config["train_data"] = train_data.to(config["device"])
    config["full_data"] = full_data.to(config["device"])

    dfcd = DFCD(config)
    pprint({k: v for k, v in config.items() if k not in ("np_data", "np_train", "np_test", "train_dataloader", "test_dataloader", "train_data", "full_data", "q", "r", "s_embed", "e_embed", "k_embed", "se_map")})
    dfcd.train_step()


if __name__ == "__main__":
    main()
