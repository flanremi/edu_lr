# -*- coding: utf-8 -*-
"""
将 LPR 数据集转为 DFCD 所需格式。

输入：
  - 训练集 CSV：QuestionId, UserId, AnswerId, IsCorrect, CorrectAnswer, AnswerValue
  - 知识点层级 CSV：SubjectId, Name, ParentId, Level
  - 题目-知识点 CSV：QuestionId, SubjectId（SubjectId 为 "[3, 71, 98]" 形式）
  - 题目内容 JSON（可选）：用于后续嵌入，格式见 questions_result.json

输出：
  - TotalData.csv：stu_id, exer_id, score(0/1)，无表头，连续从 0 的 ID
  - q.csv：题目数×知识点数 0/1 矩阵，无表头
  - config.json：stu_num, exercise_num, knowledge_num 等
  - map.pkl：stu_map, question_map, concept_map, reverse_*，供嵌入或回溯用
"""

import os
import re
import json
import pickle
import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# =============================================================================
# 【代码内变量】路径与过滤参数，按需修改
# =============================================================================
# 输入路径（相对本脚本所在目录或绝对路径）
DIR_DATA = os.path.dirname(os.path.abspath(__file__))
PATH_TRAIN_CSV = os.path.join(DIR_DATA, "train_data", "train_task_3_4.csv")
PATH_QUESTION_SUBJECT_CSV = os.path.join(DIR_DATA, "metadata", "question_metadata_task_3_4.csv")
PATH_SUBJECT_CSV = os.path.join(DIR_DATA, "metadata", "subject_metadata.csv")
PATH_QUESTIONS_JSON = os.path.join(DIR_DATA, "questions_result.json")  # 可选，用于导出题目文本

# 输出目录（将在此目录下生成 TotalData.csv, q.csv, config.json, map.pkl）
OUTPUT_DIR = os.path.join(DIR_DATA, "dfcd_format")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 过滤参数（与 DFCD 预处理类似；设为 None 表示不限制）
STU_NUM = None          # 最多保留学生数（按答题量从多到少）
EXER_NUM = None         # 最多保留题目数（按出现次数从多到少）
KNOW_NUM = None         # 最多保留知识点数（按出现次数从多到少）
LEAST_RESPONSE_NUM = 50  # 每个学生至少答题数，低于则剔除该学生
SEED = 0

# 同一 (UserId, QuestionId) 出现多次时：取 "last" 最后一次 | "first" 第一次 | "majority" 多数
DEDUPLICATE_MODE = "last"

# 题目 JSON 与 QuestionId 的对应方式（仅影响 question_texts.csv 导出）：
# "image" = 用每项的 "image" 字段解析出题目 ID（如 "39.jpg" -> 39），与训练集/元数据中的 QuestionId 匹配（推荐）
# "order" = JSON 列表顺序与「本脚本中排序后的题目列表」一致
# "index" = 用 JSON 列表下标当作 QuestionId（questions_list[QuestionId]）
# "id_field" = 用 JSON 每项中的 "question_id" / "id" / "QuestionId" 字段与 QuestionId 匹配
QUESTION_JSON_MATCH = "image"


def _parse_question_id_from_image(image_val):
    """从 image 字段（如 '39.jpg'、'123.png'）解析出题目 ID（整数）。"""
    if image_val is None or (isinstance(image_val, str) and not image_val.strip()):
        return None
    s = str(image_val).strip()
    # 去掉常见图片扩展名后取数字
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    # 若仍有非数字后缀（如路径），只取主文件名
    base = os.path.basename(s) if "/" in s or "\\" in s else s
    m = re.match(r"^(\d+)", base)
    return int(m.group(1)) if m else None


def _parse_subject_id_list(s):
    """将 SubjectId 列解析为 list[int]，支持 '[3, 71, 98]' 或 '3,71,98'。"""
    if pd.isna(s) or s == "":
        return []
    s = str(s).strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            pass
    # 退化为按逗号分割数字
    out = []
    for x in re.findall(r"\d+", s):
        out.append(int(x))
    return out


def run():
    np.random.seed(SEED)
    print("读取训练集...")
    train = pd.read_csv(PATH_TRAIN_CSV)
    # 列名兼容
    train = train.rename(columns={
        "QuestionId": "QuestionId",
        "UserId": "UserId",
        "IsCorrect": "IsCorrect",
    })
    if "IsCorrect" not in train.columns:
        raise ValueError("训练集需包含 IsCorrect 列")
    train["IsCorrect"] = train["IsCorrect"].astype(int).clip(0, 1)

    print("读取题目-知识点...")
    qs = pd.read_csv(PATH_QUESTION_SUBJECT_CSV)
    if "SubjectId" not in qs.columns or "QuestionId" not in qs.columns:
        raise ValueError("题目-知识点 CSV 需包含 QuestionId, SubjectId")
    # QuestionId -> list of SubjectId
    question_subjects = {}
    for _, row in tqdm(qs.iterrows(), total=len(qs), desc="解析 SubjectId"):
        qid = int(row["QuestionId"])
        sub_ids = _parse_subject_id_list(row["SubjectId"])
        if sub_ids:
            question_subjects[qid] = list(set(question_subjects.get(qid, []) + sub_ids))

    # 只保留在题目-知识点表中出现的题目
    train = train[train["QuestionId"].isin(question_subjects)].copy()
    if train.empty:
        raise ValueError("训练集中没有出现在题目-知识点表中的 QuestionId")

    # 同一 (UserId, QuestionId) 多条记录时的处理
    if DEDUPLICATE_MODE == "last":
        train = train.drop_duplicates(subset=["UserId", "QuestionId"], keep="last")
    elif DEDUPLICATE_MODE == "first":
        train = train.drop_duplicates(subset=["UserId", "QuestionId"], keep="first")
    elif DEDUPLICATE_MODE == "majority":
        def maj(x):
            return x.value_counts().index[0]
        train = train.groupby(["UserId", "QuestionId"], as_index=False).agg({"IsCorrect": maj})
    else:
        pass  # 保留全部记录

    # 学生答题数
    stu_counts = train.groupby("UserId").size()
    if LEAST_RESPONSE_NUM is not None:
        valid_students = stu_counts[stu_counts >= LEAST_RESPONSE_NUM].index.tolist()
        train = train[train["UserId"].isin(valid_students)]
        print(f"  至少 {LEAST_RESPONSE_NUM} 条记录的学生数: {len(valid_students)}")

    if train.empty:
        raise ValueError("过滤后无数据")

    # 确定学生、题目、知识点的候选与数量限制
    all_students = sorted(train["UserId"].unique().tolist())
    all_questions = sorted(train["QuestionId"].unique().tolist())
    all_subjects = set()
    for qid in question_subjects:
        if qid in set(all_questions):
            all_subjects.update(question_subjects[qid])
    all_subjects = sorted(all_subjects)

    if STU_NUM is not None and len(all_students) > STU_NUM:
        # 按答题量从多到少保留
        stu_counts = train.groupby("UserId").size().sort_values(ascending=False)
        all_students = stu_counts.head(STU_NUM).index.tolist()
        train = train[train["UserId"].isin(all_students)]
        all_students = sorted(all_students)
    if EXER_NUM is not None and len(all_questions) > EXER_NUM:
        exer_counts = train.groupby("QuestionId").size().sort_values(ascending=False)
        all_questions = exer_counts.head(EXER_NUM).index.tolist()
        train = train[train["QuestionId"].isin(all_questions)]
        all_questions = sorted(all_questions)
        all_subjects = set()
        for qid in question_subjects:
            if qid in set(all_questions):
                all_subjects.update(question_subjects[qid])
        all_subjects = sorted(all_subjects)
    if KNOW_NUM is not None and len(all_subjects) > KNOW_NUM:
        sub_counts = defaultdict(int)
        for qid in all_questions:
            for s in question_subjects.get(qid, []):
                sub_counts[s] += 1
        top_subjects = sorted(sub_counts.items(), key=lambda x: -x[1])[:KNOW_NUM]
        all_subjects = [s for s, _ in top_subjects]
        all_subjects = sorted(all_subjects)

    stu_map = {uid: i for i, uid in enumerate(all_students)}
    question_map = {qid: i for i, qid in enumerate(all_questions)}
    concept_map = {sid: k for k, sid in enumerate(all_subjects)}
    reverse_question_map = {i: qid for qid, i in question_map.items()}
    reverse_stu_map = {i: uid for uid, i in stu_map.items()}
    reverse_concept_map = {k: sid for sid, k in concept_map.items()}

    stu_num = len(stu_map)
    prob_num = len(question_map)
    know_num = len(concept_map)

    print("构建 TotalData.csv...")
    rows = []
    for _, r in train.iterrows():
        uid, qid, correct = int(r["UserId"]), int(r["QuestionId"]), int(r["IsCorrect"])
        if uid not in stu_map or qid not in question_map:
            continue
        rows.append([stu_map[uid], question_map[qid], correct])
    total_data = np.array(rows, dtype=np.int64)
    np.savetxt(
        os.path.join(OUTPUT_DIR, "TotalData.csv"),
        total_data,
        delimiter=",",
        fmt=["%d", "%d", "%d"],
    )
    print(f"  TotalData 行数: {total_data.shape[0]}")

    print("构建 q.csv...")
    q_matrix = np.zeros((prob_num, know_num), dtype=np.float64)
    for qid, exer_id in question_map.items():
        for sub_id in question_subjects.get(qid, []):
            if sub_id in concept_map:
                q_matrix[exer_id, concept_map[sub_id]] = 1
    np.savetxt(os.path.join(OUTPUT_DIR, "q.csv"), q_matrix, delimiter=",")
    print(f"  q 矩阵形状: {q_matrix.shape}")

    config = {
        "dataset": "LPR_task34",
        "files": {"q_matrix": "q.csv", "response": "TotalData.csv"},
        "info": {
            "student_num": stu_num,
            "exercise_num": prob_num,
            "knowledge_num": know_num,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    map_data = {
        "stu_map": stu_map,
        "question_map": question_map,
        "concept_map": concept_map,
        "reverse_question_map": reverse_question_map,
        "reverse_stu_map": reverse_stu_map,
        "reverse_concept_map": reverse_concept_map,
    }
    with open(os.path.join(OUTPUT_DIR, "map.pkl"), "wb") as f:
        pickle.dump(map_data, f)
    print("  config.json、map.pkl 已写入")

    # 可选：导出题目文本（供后续嵌入脚本使用），仅保存 id 和 content
    # id = image 去掉 .jpg 等内容；content = question + 选项（如 "Which angle...? A. A, B. B, C. C, D. D"）
    if os.path.isfile(PATH_QUESTIONS_JSON):
        print("读取题目 JSON，导出题目文本表（id, content）...")
        with open(PATH_QUESTIONS_JSON, "r", encoding="utf-8") as f:
            questions_list = json.load(f)
        if QUESTION_JSON_MATCH == "image":
            json_by_qid = {}
            for item in questions_list:
                qid = _parse_question_id_from_image(item.get("image"))
                if qid is not None:
                    json_by_qid[qid] = item
        elif QUESTION_JSON_MATCH == "order":
            json_by_qid = {}
            for i, qid in enumerate(all_questions):
                if i < len(questions_list):
                    json_by_qid[qid] = questions_list[i]
        elif QUESTION_JSON_MATCH == "index":
            json_by_qid = {i: questions_list[i] for i in range(min(len(questions_list), 200000))}
        else:
            json_by_qid = {}
            for i, item in enumerate(questions_list):
                qid = item.get("question_id", item.get("QuestionId", item.get("id", i)))
                if isinstance(qid, (int, float)):
                    json_by_qid[int(qid)] = item
            if not json_by_qid and questions_list:
                json_by_qid = {i: questions_list[i] for i in range(len(questions_list))}

        def _image_to_id(image_val):
            """image 字段去掉 .jpg 等扩展名后的内容（字符串）。"""
            if image_val is None or (isinstance(image_val, str) and not image_val.strip()):
                return ""
            s = str(image_val).strip()
            for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
                if s.lower().endswith(ext):
                    s = s[: -len(ext)]
                    break
            return os.path.basename(s) if "/" in s or "\\" in s else s

        def _normalize_line(s):
            """去掉换行、合并多余空白为单个空格。"""
            if not s or not isinstance(s, str):
                return ""
            s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            return " ".join(s.split()).strip()

        def _item_to_content(item):
            """question + 选项拼接为 '题目? A. xxx, B. xxx, C. xxx, D. xxx'，无多余换行。"""
            if not item:
                return ""
            question = _normalize_line(item.get("question", ""))
            parts = [question]
            opts = item.get("options", [])
            if opts:
                opt_strs = []
                for o in opts:
                    label = o.get("label", "")
                    text = _normalize_line(o.get("text", ""))
                    opt_strs.append(f"{label}. {text}".strip())
                parts.append(", ".join(opt_strs))
            return " ".join(parts).strip()

        text_rows = []
        for exer_id in range(prob_num):
            orig_qid = reverse_question_map[exer_id]
            item = json_by_qid.get(orig_qid)
            row_id = _image_to_id(item.get("image")) if item else ""
            content = _item_to_content(item)
            text_rows.append({"id": row_id, "content": content})
        text_df = pd.DataFrame(text_rows)
        text_df.to_csv(os.path.join(OUTPUT_DIR, "question_texts.csv"), index=False, encoding="utf-8-sig")
        print("  question_texts.csv 已写入（列: id, content）")
    else:
        print("未找到题目 JSON，跳过题目文本导出。")

    print("完成。输出目录:", os.path.abspath(OUTPUT_DIR))
    print("DFCD 使用方式：将 OUTPUT_DIR 下的 TotalData.csv、q.csv 复制到 DFCD 的 data/<数据集名>/，并在 data_params_dict.py 中增加：")
    print("  'LPR_task34': {'stu_num': %d, 'prob_num': %d, 'know_num': %d, 'batch_size': 16}" % (stu_num, prob_num, know_num))
    return OUTPUT_DIR, stu_num, prob_num, know_num


if __name__ == "__main__":
    run()
