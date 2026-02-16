# -*- coding: utf-8 -*-
"""
将 LPR 数据集转为 DFCD 所需格式。

数据格式说明（参考 NeurIPS2020 preprocess/embedding）：
  - 训练集：QuestionId, UserId, AnswerId, IsCorrect, CorrectAnswer, AnswerValue
  - 训练测试集（含 IsTarget）：同上 + IsTarget，IsTarget=0 表示可纳入训练
  - 最终验证集：test_public_task_4_more_splits.csv 格式，含 IsTarget_0..IsTarget_9
    用于 starter_kit/task_4/pytorch evaluation.py 评估

输出：TotalData.csv, q.csv, config.json, map.pkl, question_texts.csv（可选）
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
# 【请按需填写】路径与过滤参数
# =============================================================================
DIR_DATA = os.path.dirname(os.path.abspath(__file__))

# 训练集：列名 QuestionId, UserId, AnswerId, IsCorrect, CorrectAnswer, AnswerValue
PATH_TRAIN_CSV = os.path.join(DIR_DATA, "train_data", "train_task_3_4.csv")

# 训练测试集（可选）：同上 + IsTarget。若提供且 USE_IS_TARGET_FILTER=True，仅用 IsTarget==0 的行
PATH_TRAIN_TEST_CSV = None  # 如: os.path.join(DIR_DATA, "train_data", "train_test_with_target.csv")
USE_IS_TARGET_FILTER = True  # 当 PATH_TRAIN_TEST_CSV 有 IsTarget 列时，True=仅用 IsTarget==0

# 最终验证集：starter_kit evaluation 使用的 ground-truth 文件（如 test_public_task_4_more_splits.csv）
PATH_VALIDATION_CSV = os.path.join(DIR_DATA, "test_data", "test_public_task_4_more_splits.csv")

# 元数据
PATH_QUESTION_SUBJECT_CSV = os.path.join(DIR_DATA, "metadata", "question_metadata_task_3_4.csv")
PATH_SUBJECT_CSV = os.path.join(DIR_DATA, "metadata", "subject_metadata.csv")
PATH_QUESTIONS_JSON = os.path.join(DIR_DATA, "questions_result.json")

# 输出目录
OUTPUT_DIR = os.path.join(DIR_DATA, "dfcd_format")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试集转换：是否扩展 stu_map 以包含未见学生（测试集中 UserId 不在训练集的学生）
# True = 扩展，输出 TotalData_test 包含所有学生；False = 仅保留训练集学生，跳过未见学生
EXTEND_STU_MAP_FOR_TEST = True

# 过滤参数（None 表示不限制）
STU_NUM = None
EXER_NUM = None
KNOW_NUM = None
LEAST_RESPONSE_NUM = 50
SEED = 0

DEDUPLICATE_MODE = "last"
QUESTION_JSON_MATCH = "image"


def _parse_question_id_from_image(image_val):
    if image_val is None or (isinstance(image_val, str) and not image_val.strip()):
        return None
    s = str(image_val).strip()
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    base = os.path.basename(s) if "/" in s or "\\" in s else s
    m = re.match(r"^(\d+)", base)
    return int(m.group(1)) if m else None


def _parse_subject_id_list(s):
    if pd.isna(s) or s == "":
        return []
    s = str(s).strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            pass
    return [int(x) for x in re.findall(r"\d+", str(s))]


def _read_train_source():
    """读取训练数据，支持训练集与训练测试集。"""
    dfs = []
    if PATH_TRAIN_CSV and os.path.isfile(PATH_TRAIN_CSV):
        df = pd.read_csv(PATH_TRAIN_CSV)
        dfs.append(df)
    if PATH_TRAIN_TEST_CSV and os.path.isfile(PATH_TRAIN_TEST_CSV):
        df = pd.read_csv(PATH_TRAIN_TEST_CSV)
        if "IsTarget" in df.columns and USE_IS_TARGET_FILTER:
            df = df[df["IsTarget"] == 0].copy()
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("至少需提供 PATH_TRAIN_CSV 或 PATH_TRAIN_TEST_CSV")
    train = pd.concat(dfs, ignore_index=True)
    train = train.drop_duplicates(subset=["UserId", "QuestionId"], keep="last")
    return train


def run():
    np.random.seed(SEED)
    print("读取训练数据...")
    train = _read_train_source()

    for col in ("QuestionId", "UserId", "IsCorrect"):
        if col not in train.columns:
            raise ValueError(f"训练集需包含 {col} 列")
    train["IsCorrect"] = train["IsCorrect"].astype(int).clip(0, 1)

    print("读取题目-知识点...")
    qs = pd.read_csv(PATH_QUESTION_SUBJECT_CSV)
    if "SubjectId" not in qs.columns or "QuestionId" not in qs.columns:
        raise ValueError("题目-知识点 CSV 需包含 QuestionId, SubjectId")
    question_subjects = {}
    for _, row in tqdm(qs.iterrows(), total=len(qs), desc="解析 SubjectId"):
        qid = int(row["QuestionId"])
        sub_ids = _parse_subject_id_list(row["SubjectId"])
        if sub_ids:
            question_subjects[qid] = list(set(question_subjects.get(qid, []) + sub_ids))

    train = train[train["QuestionId"].isin(question_subjects)].copy()
    if train.empty:
        raise ValueError("训练集中没有出现在题目-知识点表中的 QuestionId")

    if DEDUPLICATE_MODE == "last":
        train = train.drop_duplicates(subset=["UserId", "QuestionId"], keep="last")
    elif DEDUPLICATE_MODE == "first":
        train = train.drop_duplicates(subset=["UserId", "QuestionId"], keep="first")
    elif DEDUPLICATE_MODE == "majority":
        def maj(x):
            return x.value_counts().index[0]
        train = train.groupby(["UserId", "QuestionId"], as_index=False).agg({"IsCorrect": maj})

    stu_counts = train.groupby("UserId").size()
    if LEAST_RESPONSE_NUM is not None:
        valid_students = stu_counts[stu_counts >= LEAST_RESPONSE_NUM].index.tolist()
        train = train[train["UserId"].isin(valid_students)]
        print(f"  至少 {LEAST_RESPONSE_NUM} 条记录的学生数: {len(valid_students)}")

    if train.empty:
        raise ValueError("过滤后无数据")

    all_students = sorted(train["UserId"].unique().tolist())
    all_questions = sorted(train["QuestionId"].unique().tolist())
    all_subjects = set()
    for qid in question_subjects:
        if qid in set(all_questions):
            all_subjects.update(question_subjects[qid])
    all_subjects = sorted(all_subjects)

    if STU_NUM is not None and len(all_students) > STU_NUM:
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
        "info": {"student_num": stu_num, "exercise_num": prob_num, "knowledge_num": know_num},
        "path_validation_csv": PATH_VALIDATION_CSV if PATH_VALIDATION_CSV else None,
    }

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

    if os.path.isfile(PATH_QUESTIONS_JSON):
        print("导出 question_texts.csv...")
        with open(PATH_QUESTIONS_JSON, "r", encoding="utf-8") as f:
            questions_list = json.load(f)
        if QUESTION_JSON_MATCH == "image":
            json_by_qid = {}
            for item in questions_list:
                qid = _parse_question_id_from_image(item.get("image"))
                if qid is not None:
                    json_by_qid[qid] = item
        elif QUESTION_JSON_MATCH == "order":
            json_by_qid = {all_questions[i]: questions_list[i] for i in range(min(len(all_questions), len(questions_list)))}
        else:
            json_by_qid = {}
            for i, item in enumerate(questions_list):
                qid = item.get("question_id", item.get("QuestionId", item.get("id", i)))
                if isinstance(qid, (int, float)):
                    json_by_qid[int(qid)] = item

        def _image_to_id(image_val):
            if image_val is None or (isinstance(image_val, str) and not image_val.strip()):
                return ""
            s = str(image_val).strip()
            for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
                if s.lower().endswith(ext):
                    s = s[: -len(ext)]
                    break
            return os.path.basename(s) if "/" in s or "\\" in s else s

        def _normalize_line(s):
            if not s or not isinstance(s, str):
                return ""
            return " ".join(str(s).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").split()).strip()

        def _item_to_content(item):
            if not item:
                return ""
            question = _normalize_line(item.get("question", ""))
            parts = [question]
            for o in item.get("options", []):
                parts.append(f"{o.get('label','')}. {_normalize_line(o.get('text',''))}".strip())
            return ", ".join(p for p in parts if p)

        text_rows = []
        for exer_id in range(prob_num):
            orig_qid = reverse_question_map[exer_id]
            item = json_by_qid.get(orig_qid)
            row_id = _image_to_id(item.get("image")) if item else ""
            content = _item_to_content(item) if item else ""
            text_rows.append({"id": row_id, "content": content})
        pd.DataFrame(text_rows).to_csv(os.path.join(OUTPUT_DIR, "question_texts.csv"), index=False, encoding="utf-8-sig")
    else:
        print("未找到题目 JSON，跳过 question_texts.csv")

    # 转换验证/测试集为 DFCD 格式
    stu_num_extended = stu_num
    if PATH_VALIDATION_CSV and os.path.isfile(PATH_VALIDATION_CSV):
        print("转换测试集为 DFCD 格式...")
        test_df = pd.read_csv(PATH_VALIDATION_CSV)
        for col in ("QuestionId", "UserId", "IsCorrect"):
            if col not in test_df.columns:
                print(f"  警告: 测试集缺少 {col} 列，跳过测试集转换")
                test_df = None
                break
        if test_df is not None:
            test_df["IsCorrect"] = test_df["IsCorrect"].astype(int).clip(0, 1)
            test_stu_unique = sorted(test_df["UserId"].unique().tolist())
            unseen_user_ids = [uid for uid in test_stu_unique if uid not in stu_map]
            stu_map_final = dict(stu_map)
            if EXTEND_STU_MAP_FOR_TEST and unseen_user_ids:
                for i, uid in enumerate(unseen_user_ids):
                    stu_map_final[uid] = stu_num + i
                stu_num_extended = stu_num + len(unseen_user_ids)
                print(f"  扩展 stu_map: 新增 {len(unseen_user_ids)} 名未见学生 (stu_id {stu_num}..{stu_num_extended-1})")
            test_rows = []
            skipped_qid = 0
            for _, r in test_df.iterrows():
                uid, qid, correct = int(r["UserId"]), int(r["QuestionId"]), int(r["IsCorrect"])
                if uid not in stu_map_final:
                    continue
                if qid not in question_map:
                    skipped_qid += 1
                    continue
                test_rows.append([stu_map_final[uid], question_map[qid], correct])
            if test_rows:
                test_data = np.array(test_rows, dtype=np.int64)
                test_out_path = os.path.join(OUTPUT_DIR, "TotalData_test.csv")
                np.savetxt(test_out_path, test_data, delimiter=",", fmt=["%d", "%d", "%d"])
                n_test_stu = len(set(test_data[:, 0]))
                n_test_exer = len(set(test_data[:, 1]))
                print(f"  TotalData_test.csv: {len(test_data)} 条记录, {n_test_stu} 学生, {n_test_exer} 题目")
                if skipped_qid > 0:
                    print(f"  跳过 {skipped_qid} 条（QuestionId 不在训练集 question_map 中）")
                if stu_num_extended > stu_num:
                    with open(os.path.join(OUTPUT_DIR, "map.pkl"), "rb") as f:
                        map_data = pickle.load(f)
                    map_data["stu_map_extended"] = stu_map_final
                    map_data["stu_num_extended"] = stu_num_extended
                    map_data["reverse_stu_map_extended"] = {i: uid for uid, i in stu_map_final.items()}
                    with open(os.path.join(OUTPUT_DIR, "map.pkl"), "wb") as f:
                        pickle.dump(map_data, f)
            else:
                print("  警告: 测试集转换后无有效记录")
    else:
        print("未提供 PATH_VALIDATION_CSV 或文件不存在，跳过测试集转换")

    config["info"]["student_num_extended"] = stu_num_extended
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("完成。输出目录:", os.path.abspath(OUTPUT_DIR))
    print("DFCD: 将 TotalData.csv、TotalData_test.csv、q.csv、config.json、map.pkl 复制到 DFCD data/2020/。")
    print("data_params_dict: stu_num=%d, prob_num=%d, know_num=%d" % (stu_num, prob_num, know_num))
    if stu_num_extended > stu_num:
        print("评估时请使用 stu_num=%d（或依赖 data/2020/config.json 的 student_num_extended 自动生效）" % stu_num_extended)
    if PATH_VALIDATION_CSV:
        print("验证集路径（starter_kit evaluation --ref_data）:", PATH_VALIDATION_CSV)
    return OUTPUT_DIR, stu_num, prob_num, know_num


if __name__ == "__main__":
    run()
