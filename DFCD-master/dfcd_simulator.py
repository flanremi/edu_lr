# -*- coding: utf-8 -*-
"""
DFCD 独立模拟器脚本
仅提供两个功能出口：
  1. 训练 (train)
  2. 正向传播 (forward)：含预测答题正确率、知识点掌握度
"""
import os
import sys
import warnings

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'models'))
warnings.filterwarnings("ignore")


def _default_config():
    """返回默认 config 字典，便于外部只改少量参数后调用 train。"""
    import torch
    return {
        "data_type": "XES3G5M",
        "data_root": _SCRIPT_DIR,
        "test_size": 0.2,
        "split": "Original",
        "epoch": 20,
        "lr": 1e-4,
        "batch_size": 1024,
        "weight_decay": 0,
        "seed": 0,
        "device": "cuda:0",
        "dtype": torch.float64,
        "encoder_type": "transformer",
        "decoder_type": "simplecd",
        "out_channels": 128,
        "mode": 2,
        "text_embedding_model": "openai",
    }


def _build_full_config(config):
    """补全 config（name、in_channels、data 加载等）。"""
    from utils import load_data, construct_data_geometric

    c = dict(config)
    c.setdefault("method", "dfcd")
    c.setdefault("name", f"dfcd-{c['data_type']}-seed{c.get('seed', 0)}")
    c.setdefault("data_root", _SCRIPT_DIR)
    mode = c.get("mode", 2)
    if mode == 1:
        c["method"] = c.get("method", "dfcd") + "-text"
    elif mode == 2:
        c["method"] = c.get("method", "dfcd") + "-hybrid"
    else:
        c["method"] = c.get("method", "dfcd") + "-response"

    text_emb = c.get("text_embedding_model", "openai")
    if text_emb == "openai":
        c["in_channels_llm"] = 1536
    elif text_emb == "BAAI":
        c["in_channels_llm"] = 1024
    elif text_emb in ("m3e", "instructor"):
        c["in_channels_llm"] = 768
    else:
        c["in_channels_llm"] = 1536

    load_data(c)
    c["in_channels_init"] = c["stu_num"] + c["prob_num"] + c["know_num"]

    if c.get("split") in ("Stu", "Exer"):
        train_data = construct_data_geometric(c, data=c["np_train_old"])
        full_data = construct_data_geometric(c, data=c["np_train"])
    else:
        train_data = construct_data_geometric(c, data=c["np_train"])
        full_data = train_data

    c["train_data"] = train_data.to(c["device"])
    c["full_data"] = full_data.to(c["device"])
    return c


# =============================================================================
# 出口 1：训练
# =============================================================================
def train(config=None, save_path=None):
    """
    训练 DFCD 模型。

    :param config: 配置字典；为 None 时使用 _default_config()
    :param save_path: 训练结束后保存模型权重的路径（.pt 或 .pth）；None 则不保存
    :return: 训练好的模型（DFCD 实例）
    """
    import torch
    from models.dfcd import DFCD

    if config is None:
        config = _default_config()
    if config.get("dtype") is None:
        config["dtype"] = torch.float64

    full_config = _build_full_config(config)
    model = DFCD(full_config)
    model.train_step()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")
    return model


# =============================================================================
# 出口 2：正向传播（预测 + 掌握度）
# =============================================================================
def forward(model, config, student_id, exercise_id, knowledge_point=None, mode="eval"):
    """
    正向传播：对给定的 (学生, 题目) 批量预测答题正确率。

    :param model: DFCD 模型实例（已 train 或从 load_model 加载）
    :param config: 与训练时一致的完整 config（需含 device、stu_num、prob_num 等）
    :param student_id: 学生 ID，shape (batch,) 或标量，LongTensor 或 numpy
    :param exercise_id: 题目 ID，shape (batch,) 或标量
    :param knowledge_point: 由 Q 矩阵得到的知识点 0/1，shape (batch, know_num)；None 时从 config['q'] 按 exercise_id 取
    :param mode: 'eval' 或 'train'
    :return: 预测正确率，shape (batch,)，numpy
    """
    import torch
    import numpy as np

    device = config["device"]
    if not isinstance(student_id, torch.Tensor):
        student_id = torch.tensor(student_id, dtype=torch.long, device=device)
    if student_id.dim() == 0:
        student_id = student_id.unsqueeze(0)
    if not isinstance(exercise_id, torch.Tensor):
        exercise_id = torch.tensor(exercise_id, dtype=torch.long, device=device)
    if exercise_id.dim() == 0:
        exercise_id = exercise_id.unsqueeze(0)
    if knowledge_point is None:
        q = config["q"].to(device)
        knowledge_point = q[exercise_id]
    elif not isinstance(knowledge_point, torch.Tensor):
        knowledge_point = torch.tensor(knowledge_point, dtype=torch.float64, device=device)

    model.eval()
    with torch.no_grad():
        pred = model.forward(student_id, exercise_id, knowledge_point, mode=mode)
    return pred.cpu().numpy()


def get_mastery_level(model, config, mode="eval"):
    """
    正向传播：得到所有学生在各知识点上的掌握度矩阵。

    :param model: DFCD 模型实例
    :param config: 完整 config
    :param mode: 'eval' 或 'train'
    :return: numpy 数组，shape (stu_num, know_num)，取值 0~1
    """
    model.eval()
    with __import__("torch").no_grad():
        mastery = model.get_mastery_level(mode=mode)
    return mastery


def load_model(config=None, checkpoint_path=None):
    """
    加载已保存的模型（用于仅做正向传播时）。

    :param config: 与训练时一致的 config；None 时用 _default_config() 并只做数据与结构初始化
    :param checkpoint_path: 权重文件路径（.pt/.pth）
    :return: (model, full_config)
    """
    import torch
    from models.dfcd import DFCD

    if config is None:
        config = _default_config()
    if config.get("dtype") is None:
        config["dtype"] = torch.float64

    full_config = _build_full_config(config)
    model = DFCD(full_config)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=full_config["device"]))
    return model, full_config


# =============================================================================
# 【代码内变量控制】直接改下面变量后运行本脚本即可
# =============================================================================
RUN_MODE = "train"           # "train" = 训练；"predict" = 仅正向传播
SAVE_PATH = "dfcd_model.pt"  # 训练结束后保存权重的路径；None 则不保存
LOAD_PATH = "dfcd_model.pt"  # predict 时加载的权重路径（RUN_MODE=="predict" 时必填）

# 以下覆盖默认 config，按需修改（不写的项用默认值）
CONFIG_OVERRIDE = {
    "data_type": "XES3G5M",
    "device": "cuda:0",
    "epoch": 20,
    "test_size": 0.2,
    "split": "Original",
    "text_embedding_model": "openai",
    # "lr": 1e-4,
    # "batch_size": 1024,
    # "encoder_type": "transformer",
    # "decoder_type": "simplecd",
    # "out_channels": 128,
    # "mode": 2,
}

# =============================================================================
# 主入口：根据 RUN_MODE 执行训练或正向传播
# =============================================================================
if __name__ == "__main__":
    config = _default_config()
    config.update(CONFIG_OVERRIDE)

    if RUN_MODE == "train":
        train(config=config, save_path=SAVE_PATH)
    elif RUN_MODE == "predict":
        if not LOAD_PATH or not os.path.isfile(LOAD_PATH):
            print("predict 模式需要 LOAD_PATH 指向已有的权重文件")
            sys.exit(1)
        model, full_config = load_model(config=config, checkpoint_path=LOAD_PATH)
        mastery = get_mastery_level(model, full_config)
        print(f"掌握度矩阵 shape: {mastery.shape}")
        # 示例：对某条 (学生, 题目) 做预测
        # pred = forward(model, full_config, student_id=[0], exercise_id=[10])
        # print("预测正确率:", pred)
    else:
        print("RUN_MODE 只能是 'train' 或 'predict'")
        sys.exit(1)
