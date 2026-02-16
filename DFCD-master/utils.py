"""
utils.py — 工具函数与数据加载模块
功能：
  1. 随机种子、数据转换等基础工具
  2. 统一数据加载与 4 模式划分 (load_data_unified)
  3. 图数据构建 (construct_data_geometric)
  4. 评估指标 (DOA, r_matrix 等)
  5. 模型组件 (Weighted_Summation, NoneNegClipper)
"""

import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


# ===========================================================================
# 1. 基础工具函数
# ===========================================================================

class NoneNegClipper(object):
    """将线性层权重裁剪为非负，保证解码器单调性"""
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


def transform(q, user, item, score, batch_size, dtype=torch.float64):
    """将 numpy 数据打包为 PyTorch DataLoader"""
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        q[item, :],
        torch.tensor(score, dtype=dtype)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def set_seed(seed: int):
    """固定所有随机种子，保证实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# ===========================================================================
# 2. 统一数据加载与 4 模式划分
# ===========================================================================

def load_data_unified(config):
    """
    统一数据加载与划分，一次完成 4 种测试模式的数据准备。

    划分流程：
      Step 1: 从 TotalData 中随机划分 20% 样本作为测试集 (np_test)，80% 作为初始训练集 (np_train)
      Step 2: 从所有学生中剥离 20% 作为陌生学生 (new_stu)
      Step 3: 从所有习题中剥离 10% 作为陌生习题 (new_exer)
      Step 4: 从所有知识点中剥离 10% 作为陌生知识点 (new_know)，
              找出仅涉及陌生知识点的习题 (new_know_exer)
      Step 5: 从 np_train 中移除 new_stu、new_exer、new_know_exer 相关记录，
              得到最终训练集 (np_train_final)

    config 中新增的关键字段：
      - np_train:       80% 初始训练集（含所有实体）
      - np_train_final: 最终训练集（剥离陌生实体后）
      - np_test:        20% 测试集
      - splits_info:    4 种测试模式所需的索引集合
      - q:              Q 矩阵 (tensor)
      - r:              测试集响应矩阵
      - train_dataloader / test_dataloader
    """
    # ---- 环境初始化 ----
    set_seed(config['seed'])
    torch.set_default_dtype(config['dtype'])

    # ---- 加载数据集参数 ----
    from data.data_params_dict import data_params
    config.update({
        'stu_num': data_params[config["data_type"]]['stu_num'],
        'prob_num': data_params[config["data_type"]]['prob_num'],
        'know_num': data_params[config["data_type"]]['know_num'],
    })
    print(f"[load_data] 数据集: {config['data_type']} | "
          f"学生: {config['stu_num']} | 习题: {config['prob_num']} | 知识点: {config['know_num']}")

    # ---- 加载嵌入 ----
    import pickle
    embed_path = '../data/{}/embedding_{}.pkl'.format(config["data_type"], config["text_embedding_model"])
    print(f"[load_data] 加载嵌入: {embed_path}")
    with open(embed_path, 'rb') as file:
        embeddings = pickle.load(file)
        config['s_embed'] = embeddings['student_embeddings']
        config['e_embed'] = embeddings['exercise_embeddings']
        config['k_embed'] = embeddings['knowledge_embeddings']

    # ---- 加载 Q 矩阵 ----
    q_np = pd.read_csv(f'../data/{config["data_type"]}/q.csv', header=None).to_numpy()
    q_tensor = torch.tensor(q_np)

    # ---- 加载 TotalData ----
    np_data = pd.read_csv(f'../data/{config["data_type"]}/TotalData.csv', header=None).to_numpy()
    print(f"[load_data] 总记录数: {len(np_data)}")

    # ---- 构建 se_map: student_id -> {exer_id: embedding_index} ----
    TotalData = pd.DataFrame(np_data, columns=['stu', 'exer', 'answervalue'])
    se_map = {}
    for stu_id in range(config['stu_num']):
        student_logs = TotalData.loc[TotalData['stu'] == stu_id]
        se_map[stu_id] = {}
        cnt = 0
        for log in student_logs.values:
            se_map[stu_id][log[1]] = cnt
            cnt += 1
    config['se_map'] = se_map

    # ================================================================
    # Step 1: 样本级随机划分 — 80% 训练 / 20% 测试
    # ================================================================
    np_train, np_test = train_test_split(
        np_data, test_size=config['test_size'], random_state=config['seed']
    )
    print(f"[Step 1] 样本划分 — 训练集: {len(np_train)} | 测试集: {len(np_test)}")

    # ================================================================
    # Step 2: 剥离 20% 学生作为陌生学生
    # ================================================================
    new_stu_size = int(config.get('new_stu_ratio', 0.2) * config['stu_num'])
    new_stu = np.random.choice(
        np.arange(config['stu_num']), size=new_stu_size, replace=False
    )
    new_stu_set = set(new_stu.tolist())
    print(f"[Step 2] 陌生学生: {len(new_stu)} / {config['stu_num']} "
          f"({len(new_stu)/config['stu_num']*100:.1f}%)")

    # ================================================================
    # Step 3: 剥离 10% 习题作为陌生习题
    # ================================================================
    new_exer_size = int(config.get('new_exer_ratio', 0.1) * config['prob_num'])
    new_exer = np.random.choice(
        np.arange(config['prob_num']), size=new_exer_size, replace=False
    )
    new_exer_set = set(new_exer.tolist())
    print(f"[Step 3] 陌生习题: {len(new_exer)} / {config['prob_num']} "
          f"({len(new_exer)/config['prob_num']*100:.1f}%)")

    # ================================================================
    # Step 4: 剥离 10% 知识点作为陌生知识点
    # ================================================================
    new_know_size = int(config.get('new_know_ratio', 0.1) * config['know_num'])
    new_know = np.random.choice(
        np.arange(config['know_num']), size=new_know_size, replace=False
    )
    exist_know = np.setdiff1d(np.arange(config['know_num']), new_know)
    # 找出至少涉及一个已知知识点的习题
    know_exer_exist = np.where((q_np[:, exist_know] == 1).any(axis=1))[0]
    # 仅涉及陌生知识点的习题（不涉及任何已知知识点）
    new_know_exer = np.setdiff1d(np.arange(config['prob_num']), know_exer_exist)
    new_know_exer_set = set(new_know_exer.tolist())
    print(f"[Step 4] 陌生知识点: {len(new_know)} / {config['know_num']} "
          f"({len(new_know)/config['know_num']*100:.1f}%) "
          f"→ 关联陌生习题: {len(new_know_exer)}")

    # ================================================================
    # Step 5: 构建最终训练集 — 移除所有陌生实体的记录
    # ================================================================
    mask_keep_stu = ~np.isin(np_train[:, 0], new_stu)        # 移除陌生学生的记录
    mask_keep_exer = ~np.isin(np_train[:, 1], new_exer)       # 移除陌生习题的记录
    mask_keep_know = ~np.isin(np_train[:, 1], new_know_exer)  # 移除陌生知识点习题的记录
    np_train_final = np_train[mask_keep_stu & mask_keep_exer & mask_keep_know]

    removed = len(np_train) - len(np_train_final)
    print(f"[Step 5] 最终训练集: {len(np_train_final)} "
          f"(移除 {removed} 条, {removed/len(np_train)*100:.1f}%)")

    # ---- 汇总 splits_info ----
    splits_info = {
        'new_stu_set': new_stu_set,
        'new_exer_set': new_exer_set,
        'new_know_exer_set': new_know_exer_set,
    }

    # ---- 写入 config ----
    config.update({
        'np_data': np_data,
        'np_train': np_train,               # 80% 完整训练集（含陌生实体）
        'np_train_final': np_train_final,    # 最终训练集（剥离陌生实体后）
        'np_test': np_test,                  # 20% 测试集
        'q': q_tensor,
        'q_np': q_np,
        'r': get_r_matrix(np_test, config['stu_num'], config['prob_num']),
        'splits_info': splits_info,
    })

    # ---- 构建 DataLoader ----
    config['train_dataloader'] = transform(
        q_tensor, np_train_final[:, 0], np_train_final[:, 1],
        np_train_final[:, 2], config['batch_size']
    )
    config['test_dataloader'] = transform(
        q_tensor, np_test[:, 0], np_test[:, 1],
        np_test[:, 2], config['batch_size']
    )

    print(f"[load_data] 数据加载完成\n")


# ===========================================================================
# 3. 图数据构建
# ===========================================================================

def construct_data_geometric(config, data):
    """
    构建 PyTorch Geometric 图数据。

    参数:
      config: 包含 np_train, s_embed, e_embed, k_embed, se_map, q 等
      data:   用于构建边和 x_init 的样本数据（np_train_final 或 np_train）

    返回:
      graph_data: Data(x_llm, x_init, edge_index)
    """
    from torch_geometric.data import Data

    sek = config['stu_num'] + config['prob_num'] + config['know_num']
    se_source_index, ek_source_index = [], []
    se_target_index, ek_target_index = [], []
    se_label = []
    s_train_embed = []

    # ---- 计算学生 LLM 嵌入（使用 np_train 完整训练集） ----
    TotalData_for_embed = pd.DataFrame(config['np_train'], columns=['stu', 'exer', 'answervalue'])
    embed_dim = len(config['e_embed'][0])  # 嵌入维度

    for stu_id in range(config['stu_num']):
        student_logs = TotalData_for_embed.loc[TotalData_for_embed['stu'] == stu_id]
        tmp_embed = []
        for log in student_logs.values:
            exer_id = log[1]
            if stu_id in config['se_map'] and exer_id in config['se_map'][stu_id]:
                tmp_embed.append(config['s_embed'][stu_id][config['se_map'][stu_id][exer_id]])
        if not tmp_embed:
            # 该学生在训练集中无记录，用零向量占位
            tmp_embed.append(np.zeros(embed_dim))
        s_train_embed.append(tmp_embed)

    # ---- 构建 x_llm: [学生嵌入均值, 题目嵌入, 知识点嵌入] ----
    node_features_llm = torch.tensor(
        [np.array(s_train_embed[stu_id]).mean(axis=0) for stu_id in range(config['stu_num'])] +
        [config['e_embed'][exer_id] for exer_id in range(config['prob_num'])] +
        [config['k_embed'][know_id] for know_id in range(config['know_num'])],
        dtype=torch.float64)

    # ---- 构建 x_init: 基于答题正确/错误的邻接矩阵 ----
    node_features_init = torch.zeros(size=(sek, sek), dtype=torch.float64)
    for _, (stu_id, exer_id, label) in enumerate(data):
        stu_id, exer_id = int(stu_id), int(exer_id)
        val = 1 if label == 1 else -1
        node_features_init[stu_id, exer_id + config['stu_num']] = val
        node_features_init[exer_id + config['stu_num'], stu_id] = val

    # ---- 构建边: 学生-题目 ----
    for _, (stu_id, exer_id, label) in enumerate(data):
        stu_id, exer_id = int(stu_id), int(exer_id)
        se_source_index.append(stu_id)
        se_target_index.append(exer_id + config['stu_num'])
        se_label.append(label)

    # ---- 构建边: 题目-知识点 ----
    q_np = config['q'].detach().cpu().numpy() if isinstance(config['q'], torch.Tensor) else config['q_np']
    for exer_id, know_id in zip(*np.where(q_np != 0)):
        ek_source_index.append(exer_id + config['stu_num'])
        ek_target_index.append(know_id + config['stu_num'] + config['prob_num'])

    edge_index = torch.tensor(
        [se_source_index + ek_source_index, se_target_index + ek_target_index],
        dtype=torch.long
    )
    graph_data = Data(x_llm=node_features_llm, x_init=node_features_init, edge_index=edge_index)
    return graph_data


# ===========================================================================
# 4. 评估指标相关
# ===========================================================================

def get_r_matrix(np_test, stu_num, prob_num):
    """构建测试集响应矩阵 r[stu, prob] = score, 未作答为 -1"""
    r = -1 * np.ones(shape=(stu_num, prob_num))
    for i in range(np_test.shape[0]):
        s = int(np_test[i, 0])
        p = int(np_test[i, 1])
        score = np_test[i, 2]
        r[s, p] = int(score)
    return r


def get_top_k_concepts(datatype: str, topk: int = 10):
    """按出现频率获取 top-k 知识点"""
    q = pd.read_csv('../data/{}/q.csv'.format(datatype), header=None).to_numpy()
    a = pd.read_csv('../data/{}/TotalData.csv'.format(datatype), header=None).to_numpy()
    skill_dict = {k: 0 for k in range(q.shape[1])}
    for k in range(a.shape[0]):
        prob_id = a[k, 1]
        skills = np.where(q[int(prob_id), :] != 0)[0].tolist()
        for skill in skills:
            skill_dict[skill] += 1
    sorted_dict = dict(sorted(skill_dict.items(), key=lambda x: x[1], reverse=True))
    return list(sorted_dict.keys())[:topk]


def get_doa(config, mastery_level):
    """计算 DOA@10 指标 (Degree of Agreement)"""
    q_matrix = config['q'].detach().cpu().numpy() if isinstance(config['q'], torch.Tensor) else config['q_np']
    r_matrix = config['r']
    from metrics.DOA import calculate_doa_k_block
    check_concepts = get_top_k_concepts(config['data_type'])
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in check_concepts)
    return np.mean(doa_k_list)


# ===========================================================================
# 5. 模型组件
# ===========================================================================

class Weighted_Summation(nn.Module):
    """注意力加权融合模块：将多个嵌入通过注意力权重加权求和"""
    def __init__(self, hidden_dim, attn_drop, dtype=torch.float64):
        super(Weighted_Summation, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim), dtype=dtype), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


# ===========================================================================
# 6. 命令行参数
# ===========================================================================

def set_common_args(parser):
    """设置公共命令行参数"""
    parser.add_argument('--method', default='idcd', type=str, help='方法名')
    parser.add_argument('--data_type', default='NeurIPS2020', type=str, help='数据集')
    parser.add_argument('--test_size', default=0.2, type=float, help='测试集比例')
    parser.add_argument('--new_stu_ratio', default=0.2, type=float, help='陌生学生比例')
    parser.add_argument('--new_exer_ratio', default=0.1, type=float, help='陌生习题比例')
    parser.add_argument('--new_know_ratio', default=0.1, type=float, help='陌生知识点比例')
    parser.add_argument('--epoch', default=20, type=int, help='训练轮数')
    parser.add_argument('--seed', default=0, type=int, help='随机种子')
    parser.add_argument('--dtype', default=torch.float64, help='张量类型')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=1024, help='批大小')
    parser.add_argument('--text_embedding_model', default='openai', type=str, help='嵌入模型')
    parser.add_argument('--weight_decay', default=0, type=float, help='权重衰减')
    parser.add_argument('--gpus', default='0', type=str, help='GPU编号，多卡用逗号分隔，如 0,1,2')
    parser.add_argument('--save_dir', default='../checkpoints', type=str, help='模型保存目录')
    return parser
