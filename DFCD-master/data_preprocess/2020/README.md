# 2020 数据集预处理与嵌入

本目录与 **data/data/convert_to_dfcd.py** 兼容：既支持「已用 convert 生成并放到 data/2020/」的流程，也支持「从原始 CSV 在 2020/data/ 下重新过滤」的流程。

---

## 一、推荐流程（已用 convert_to_dfcd.py 生成过）

若你已经在项目里跑过 **data/data/convert_to_dfcd.py**，并把输出目录（如 `data/data/dfcd_format`）里的文件**拷贝到了 DFCD 的 data/2020/**，按下面做即可。

### 1. 确认 data/2020/ 下已有这些文件

- `TotalData.csv`、`q.csv`、`map.pkl`（convert 必出）
- `question_texts.csv`（convert 在提供 questions_result.json 时会出，嵌入必需）
- 可选：`subject_metadata.csv`（知识点名称，用于嵌入里的 knowledge/学生 prompt；可从 `data/data/metadata/subject_metadata.csv` 拷过来）

### 2. 可选：用 preprocess 做一次“同步”

若你是把 convert 的输出放在**别的目录**（例如仍是 `E:\...\data\data\dfcd_format`），希望同步到 `data/2020/` 和 `2020/result/data/`，可在 **preprocess.py 顶部**设置：

```python
CONVERT_OUTPUT_DIR = r"E:\programming\edu_LPR\data\data\dfcd_format"
```

然后在 `data_preprocess` 下执行：

```bash
python main_filter.py --dataset 2020 --seed 0 --stu_num 4918 --exer_num 948 --know_num 86 --least_respone_num 50
```

脚本会从 `CONVERT_OUTPUT_DIR` 拷贝 `TotalData.csv`、`q.csv`、`map.pkl`、`question_texts.csv` 等到 `../data/2020/` 和 `2020/result/data/`，**不会**再读原始训练集 CSV。

若你已经**手动**把 convert 的输出拷进了 `DFCD-master/data/2020/`，可以**不跑** preprocess，直接做第 3 步。

### 3. 运行嵌入（只读 data/2020/）

嵌入脚本**只从 `../data/2020/` 读**：TotalData、q、map、question_texts.csv、可选的 subject_metadata.csv。

在 `data_preprocess` 下执行：

```bash
python main_embedding.py --dataset 2020 --llm OpenAI
```

（也可选 BAAI / m3e / Instructor。）

生成结果：

- `../data/2020/embedding_openai.pkl`（或对应模型名）
- `2020/result/embedding/embedding_*.pkl`

### 4. 核对 data_params_dict.py 并训练

在 `data/data_params_dict.py` 中确认 `2020` 的 `stu_num`、`prob_num`、`know_num` 与 convert 输出一致（convert 结束时会在终端打印可复制的条目）。训练时 `data_type=2020`、`text_embedding_model` 与上面使用的模型名一致即可。

---

## 二、备选流程（从原始数据在 2020 下做一遍）

若**没有**用 convert_to_dfcd，而是希望像其他数据集一样在 DFCD 里从原始 CSV 做过滤和重映射：

1. 在 **2020/data/** 下按原说明放置：  
   `train_data/train_task_3_4.csv`、`metadata/question_metadata_task_3_4.csv`、`metadata/subject_metadata.csv`、`questions_result.json`。
2. 保持 **preprocess.py 里 `CONVERT_OUTPUT_DIR = None`**。
3. 执行：  
   `python main_filter.py --dataset 2020 --seed 0 --stu_num ... --exer_num ... --know_num ... --least_respone_num 50`  
   会从 2020/data/ 读原始数据，写出 TotalData、q、map 到 `../data/2020/` 和 `2020/result/data/`。  
   注意：此路径**不会**生成 `question_texts.csv`，后续嵌入会报错；若要走此路径，需要自己在 result/data 或 data/2020 下按 convert 的格式提供 `question_texts.csv`（列：exercise_id, QuestionId, text）。

因此**更推荐**：统一用 **convert_to_dfcd 生成 → 拷贝到 data/2020/ → 只跑 embedding**（或再按需跑一次 preprocess 做同步）。

---

## 三、嵌入所需文件小结（data/2020/）

| 文件 | 来源 | 说明 |
|------|------|------|
| TotalData.csv | convert 或 preprocess | 必选 |
| q.csv | convert 或 preprocess | 必选 |
| map.pkl | convert 或 preprocess | 必选，含 reverse_concept_map 等 |
| question_texts.csv | convert（或自建） | 必选，列 exercise_id, QuestionId, text |
| subject_metadata.csv | 从 data/data/metadata 拷入 | 可选，用于知识点名称 |

---

## 四、简要流程小结（你当前情况）

1. 已用 **convert_to_dfcd.py** 得到 `dfcd_format/`。
2. 把 `dfcd_format/` 下的 **TotalData.csv、q.csv、map.pkl、question_texts.csv** 拷到 **DFCD-master/data/2020/**；可选把 **subject_metadata.csv** 也拷到 **data/2020/**。
3. （可选）在 preprocess 里设 `CONVERT_OUTPUT_DIR` 指向 `dfcd_format`，再运行一次 `main_filter.py --dataset 2020 ...` 做同步；若已手动拷好可跳过。
4. 在 `data_preprocess` 下运行：  
   `python main_embedding.py --dataset 2020 --llm OpenAI`（或 BAAI/m3e/Instructor）。
5. 确认 `data_params_dict.py` 中 `2020` 的维度，训练时 `data_type=2020`、`text_embedding_model` 与嵌入模型一致。
