# DFCD 数据格式与使用说明

本文档说明 KDD 2025 论文 **"A Dual-Fusion Cognitive Diagnosis Framework for Open Student Learning Environments"** 参考代码所接收的数据集格式、各列含义以及输出格式，便于自行准备数据与复现实验。

---

## 一、论文与代码作用简述

- **论文**：双融合认知诊断框架（DFCD），面向开放学习环境，同时利用**答题行为**与**文本语义**（题目/知识点/学生记录描述）进行认知诊断。
- **代码作用**：
  - 使用 **Q 矩阵**（题目-知识点）、**答题记录**（学生-题目-对错）和 **预计算好的文本嵌入** 构建学生-题目-知识点的异构图；
  - 通过 GNN + 双路融合（行为初始化表示 + LLM 文本表示）得到学生/题目/知识点的表示；
  - 用解码器预测答题正确率，并输出**知识点掌握度（mastery level）**；
  - 支持多种划分方式：按学生 / 按题目 / 按知识点 / 标准随机划分，用于“未见学生/题目/知识点”等场景。

---

## 二、接收的数据集格式

代码假定每个数据集对应一个**目录**（如 `XES3G5M`、`NeurIPS2020`、`MOOCRadar`），目录下包含以下文件。

### 2.1 必需文件一览

| 文件 | 说明 |
|------|------|
| `TotalData.csv` | 答题记录（学生-题目-得分） |
| `q.csv` | Q 矩阵（题目-知识点 0/1） |
| `embedding_{模型名}.pkl` | 预计算的文本嵌入（学生/题目/知识点） |
| `config.json`（可选） | 数据集元信息，与 `data_params_dict.py` 一致即可 |

### 2.2 TotalData.csv（答题记录）

- **格式**：CSV，**无表头**，每行一条答题记录。
- **列数与含义**：

| 列索引 | 含义 | 类型与说明 |
|--------|------|------------|
| 第 1 列 | 学生 ID | 整数，从 **0** 开始连续编号，范围 `[0, stu_num-1]` |
| 第 2 列 | 题目 ID | 整数，从 **0** 开始连续编号，范围 `[0, prob_num-1]` |
| 第 3 列 | 答题结果（得分） | 0 或 1；**0=错误，1=正确**（二分类） |

- **示例**（前几行）：

```text
0,329,1
0,558,1
0,543,1
0,52,1
0,14,1
```

表示：学生 0 在题目 329、558、543、52、14 上均答对（值为 1）。

- **注意**：
  - 学生 ID、题目 ID 必须与 `q.csv` 的题目维度、以及嵌入中的索引一致；
  - 同一 (学生, 题目) 一般只保留一条记录（如最后一次或聚合后的 0/1）。

### 2.3 q.csv（Q 矩阵）

- **格式**：CSV，**无表头**，数值为 0/1 矩阵。
- **形状**：行数 = **题目数（prob_num）**，列数 = **知识点数（know_num）**。
- **含义**：
  - `q[i][j] = 1` 表示题目 `i` 考查知识点 `j`；
  - `q[i][j] = 0` 表示题目 `i` 不考查知识点 `j`。

- **示例**（片段）：  
  若题目 0 考查知识点 0 和 2，则第 0 行可能为：`1,0,1,0,...`（共 know_num 列）。

- **注意**：题目 ID 与 `TotalData.csv` 第 2 列、以及 `embedding_*.pkl` 中题目顺序一致。

### 2.4 embedding_{模型名}.pkl（文本嵌入）

- **生成方式**：通过 `data_preprocess` 下的 `main_embedding.py` + 各数据集的 `embedding.py` 生成（需先有 `TotalData.csv`、`q.csv` 及原始题目/知识点文本）。
- **模型名**：与运行实验时的 `--text_embedding_model` 对应，如 `openai`、`BAAI`、`m3e`、`instructor`；文件名即为 `embedding_openai.pkl`、`embedding_BAAI.pkl` 等。
- **内容**：一个 Python `dict`，需包含以下三个键（与 `utils.load_data` 中一致）：

| 键 | 含义 | 形状/结构 |
|----|------|-----------|
| `student_embeddings` | 学生文本嵌入 | **list of array**：长度为 stu_num；`student_embeddings[i]` 为学生 i 的**多条**记录对应的嵌入，形状为 `(该学生答题条数, embed_dim)`；代码中会对每学生沿 axis=0 取 mean 得到一条向量 |
| `exercise_embeddings` | 题目文本嵌入 | **2D array**：形状 `(prob_num, embed_dim)`；`exercise_embeddings[j]` 为题目 j 的向量 |
| `knowledge_embeddings` | 知识点文本嵌入 | **2D array**：形状 `(know_num, embed_dim)` |

- **embed_dim** 与 `text_embedding_model` 对应（代码中写死）：
  - `openai` → 1536
  - `BAAI` → 1024
  - `m3e` / `instructor` → 768

- **注意**：训练/推理前必须已生成并放在 `data/<数据集名>/embedding_<模型名>.pkl`，否则会报错。

### 2.5 config.json（可选）

- 用于记录数据集元信息，与 `data/data_params_dict.py` 中该数据集的 `stu_num`、`prob_num`、`know_num` 等一致即可。
- 示例：

```json
{
  "dataset": "MOOCRadar",
  "files": {
    "q_matrix": "q.csv",
    "response": "TotalData.csv"
  },
  "info": {
    "student_num": 2000,
    "exercise_num": 915,
    "knowledge_num": 696
  }
}
```

**实际加载时以 `data_params_dict.py` 为准**；若使用自己的数据集，需在该文件中添加同名条目（stu_num、prob_num、know_num、batch_size）。

---

## 三、数据维度与 data_params_dict.py

- 代码从 `data/data_params_dict.py` 读取每个数据集的学生数、题目数、知识点数（及 batch_size），例如：

```python
data_params = {
    'MOOCRadar': {
        'stu_num': 2000,
        'prob_num': 915,
        'know_num': 696,
        'batch_size': 16
    },
    # ...
}
```

- **约束**：
  - `TotalData.csv` 中学生 ID ∈ [0, stu_num-1]，题目 ID ∈ [0, prob_num-1]；
  - `q.csv` 形状为 (prob_num, know_num)；
  - 嵌入中 student / exercise / knowledge 的数量与 stu_num、prob_num、know_num 一致。

---

## 四、数据预处理流程（如何得到上述文件）

1. **原始数据**：放在 `data_preprocess/<数据集名>/data/`（如先解压 `data.zip`），需包含题目文本、知识点标注、学生答题序列等（具体见各数据集下的 `preprocess.py`）。
2. **过滤与构造 Q 矩阵、答题表**：在 `data_preprocess` 目录下执行：
   ```bash
   python main_filter.py --dataset MOOCRadar --seed 0 --stu_num 2000 --exer_num 1500 --know_num 500 --least_respone_num 50
   ```
   会生成并写入 `../data/<数据集名>/TotalData.csv` 和 `../data/<数据集名>/q.csv`，以及 result 下的副本。
3. **生成嵌入**：同一目录下执行：
   ```bash
   python main_embedding.py --dataset MOOCRadar --llm OpenAI
   ```
   会生成 `../data/<数据集名>/embedding_openai.pkl`（若 llm 为 BAAI 则生成 `embedding_BAAI.pkl`）。需配置 `OPENAI_API_KEY` 等。
4. **同步参数**：在 `data/data_params_dict.py` 中填写或修改该数据集的 `stu_num`、`prob_num`、`know_num`（与预处理后实际维度一致）。

---

## 五、模型输出格式

### 5.1 训练阶段

- **输入**：上述 `TotalData.csv`、`q.csv`、`embedding_*.pkl`；以及命令行或配置中的 `test_size`、`split` 等。
- **划分**：按 `test_size` 将答题记录划分为 train/test；若 `split` 为 `Stu`/`Exer`/`Know`，会进一步按“未见学生/题目/知识点”划分，评估时只在“未见”部分算指标。
- **输出**：每个 epoch 在控制台打印 loss 以及测试集上的：
  - **AUC**、**AP**、**ACC**、**RMSE**、**F1**、**DOA@10**（知识点掌握度与答题一致性的 DOA 指标）。

### 5.2 预测与知识点掌握度

- **答题正确率预测**：  
  `model.forward(student_id, exercise_id, knowledge_point, mode='eval')`  
  返回 shape `(batch_size,)` 的 Tensor，表示每条 (学生, 题目) 在对应 Q 矩阵知识点上的正确率（0~1）。
- **知识点掌握度（mastery level）**：  
  `model.get_mastery_level(mode='eval')`  
  返回 `numpy` 数组，形状 `(stu_num, know_num)`，表示每个学生在每个知识点上的掌握度（0~1）。

上述两者即为代码的“输出格式”：标量指标 + 预测概率 + 掌握度矩阵。

---

## 六、小结表

| 项目 | 说明 |
|------|------|
| **TotalData.csv** | 3 列：学生ID, 题目ID, 0/1 得分；无表头 |
| **q.csv** | (prob_num × know_num) 0/1 矩阵；无表头 |
| **embedding_*.pkl** | dict：`student_embeddings`（list of array）、`exercise_embeddings`、`knowledge_embeddings`（2D array） |
| **data_params_dict.py** | 每个数据集：stu_num、prob_num、know_num、batch_size |
| **训练输出** | 每 epoch：Loss, AUC, ACC, RMSE, F1, DOA@10 |
| **推理输出** | 正确率预测（batch 向量）；掌握度矩阵（stu_num × know_num） |

按上述格式准备数据并生成嵌入后，可使用以下任一方式运行：

- **原实验脚本**：`cd exps` 后 `python dfcd_exp.py --method=dfcd --data_type=...`（见 README）。
- **可配置脚本**：在项目根目录运行 `python run_config.py`，所有可调参数在脚本顶部 `CONFIG` 中集中修改（见脚本内“完整实行方案”注释）。
- **独立模拟器**：`python dfcd_simulator.py --mode train` 或 `--mode predict --load <权重路径>`；或作为模块调用 `train(config)`、`forward(model, config, ...)`、`get_mastery_level(model, config)`（仅两个功能出口：训练、正向传播）。
