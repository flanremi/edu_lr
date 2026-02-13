# -*- coding: utf-8 -*-
"""
2020 数据集预处理：生成 TotalData.csv、q.csv、map.pkl。

两种用法（二选一）：
1) 已用 data/data/convert_to_dfcd.py 生成过：设置 CONVERT_OUTPUT_DIR 为 dfcd_format 路径，
   本脚本只做拷贝到 ../data/2020/ 与 result/data/，不再读原始 CSV。
2) 未做转换：将原始数据放到 2020/data/ 下，本脚本从原始数据过滤并写出。
"""
import os
import re
import ast
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pickle

DATASET = "2020"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "data")
RESULT_DATA = os.path.join(_THIS_DIR, "result", "data")
# 仓库根下的 data/2020（preprocess 在 data_preprocess/2020/，需回退两级）
DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "data", DATASET)

# 若已用 convert_to_dfcd.py 生成过，填其输出目录路径；None 时自动尝试下面默认路径
# 默认：假设项目结构为 edu_LPR/DFCD-master/ 与 edu_LPR/data/data/dfcd_format/
_DEFAULT_CONVERT_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "data", "data", "dfcd_format"))
CONVERT_OUTPUT_DIR = None  # 也可填绝对路径，如 r"E:\programming\edu_LPR\data\data\dfcd_format"


def _parse_subject_id_list(s):
    if pd.isna(s) or s == "":
        return []
    s = str(s).strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            pass
    out = []
    for x in re.findall(r"\d+", s):
        out.append(int(x))
    return out


def fix_seeds(seed=101):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _sync_from_convert_output(src_dir):
    """从 convert_to_dfcd 输出目录拷贝到 DATA_OUT 与 RESULT_DATA。"""
    os.makedirs(RESULT_DATA, exist_ok=True)
    os.makedirs(DATA_OUT, exist_ok=True)
    for name in ["TotalData.csv", "q.csv", "map.pkl"]:
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            raise FileNotFoundError("convert 输出目录中缺少: " + name)
        for dst_dir in [DATA_OUT, RESULT_DATA]:
            shutil.copy2(src, os.path.join(dst_dir, name))
    if os.path.isfile(os.path.join(src_dir, "question_texts.csv")):
        for dst_dir in [DATA_OUT, RESULT_DATA]:
            shutil.copy2(
                os.path.join(src_dir, "question_texts.csv"),
                os.path.join(dst_dir, "question_texts.csv"),
            )
    if os.path.isfile(os.path.join(src_dir, "config.json")):
        shutil.copy2(os.path.join(src_dir, "config.json"), os.path.join(DATA_OUT, "config.json"))
    print("已从 convert 输出目录同步到 data/2020/ 与 2020/result/data/。")


def run(config):
    # 1) 若指定了 CONVERT_OUTPUT_DIR，从该目录同步
    src = CONVERT_OUTPUT_DIR
    if src:
        if not os.path.isdir(src):
            src = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", src))
        if os.path.isdir(src) and os.path.isfile(os.path.join(src, "TotalData.csv")):
            _sync_from_convert_output(src)
            return
        raise FileNotFoundError("CONVERT_OUTPUT_DIR 指向的目录无效或缺少 TotalData.csv: " + str(CONVERT_OUTPUT_DIR))

    # 2) 未指定时：先尝试默认 convert 输出路径（edu_LPR/data/data/dfcd_format）
    if os.path.isfile(os.path.join(_DEFAULT_CONVERT_PATH, "TotalData.csv")):
        print("使用默认 convert 输出目录:", _DEFAULT_CONVERT_PATH)
        _sync_from_convert_output(_DEFAULT_CONVERT_PATH)
        return

    # 3) 若 data/2020/ 里已有 TotalData、q，只同步到 result/data（便于后续 embedding 兼容）
    if os.path.isfile(os.path.join(DATA_OUT, "TotalData.csv")) and os.path.isfile(os.path.join(DATA_OUT, "q.csv")):
        print("data/2020/ 已有 TotalData.csv 与 q.csv，仅同步到 2020/result/data/。")
        os.makedirs(RESULT_DATA, exist_ok=True)
        for name in ["TotalData.csv", "q.csv", "map.pkl"]:
            p = os.path.join(DATA_OUT, name)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(RESULT_DATA, name))
        if os.path.isfile(os.path.join(DATA_OUT, "question_texts.csv")):
            shutil.copy2(os.path.join(DATA_OUT, "question_texts.csv"), os.path.join(RESULT_DATA, "question_texts.csv"))
        return

    # 4) 否则从 2020/data/ 原始数据做过滤与重映射
    fix_seeds(config["seed"])
    least_respone_num = config["least_respone_num"]
    stu_num = config["stu_num"]
    exer_num = config["exer_num"]
    know_num = config["know_num"]

    path_train = os.path.join(DATA_DIR, "train_data", "train_task_3_4.csv")
    path_qs = os.path.join(DATA_DIR, "metadata", "question_metadata_task_3_4.csv")
    if not os.path.isfile(path_train):
        raise FileNotFoundError(
            "未找到 convert 输出目录（已尝试 %s）且 2020/data/ 下无训练集。\n"
            "请任选其一：① 将 convert_to_dfcd 的输出放到 %s 或 DFCD-master/data/2020/；"
            "② 在 preprocess.py 中设置 CONVERT_OUTPUT_DIR；③ 将原始数据放到 2020/data/train_data/ 与 metadata/。"
            % (_DEFAULT_CONVERT_PATH, _DEFAULT_CONVERT_PATH)
        )
    if not os.path.isfile(path_qs):
        raise FileNotFoundError("请将题目-知识点表放到 2020/data/metadata/question_metadata_task_3_4.csv")

    train = pd.read_csv(path_train)
    train["IsCorrect"] = train["IsCorrect"].astype(int).clip(0, 1)

    qs = pd.read_csv(path_qs)
    question_subjects = {}
    for _, row in tqdm(qs.iterrows(), total=len(qs), desc="解析 SubjectId"):
        qid = int(row["QuestionId"])
        sub_ids = _parse_subject_id_list(row["SubjectId"])
        if sub_ids:
            question_subjects[qid] = list(set(question_subjects.get(qid, []) + sub_ids))

    train = train[train["QuestionId"].isin(question_subjects)].copy()
    train = train.drop_duplicates(subset=["UserId", "QuestionId"], keep="last")

    stu_counts = train.groupby("UserId").size()
    valid_students = stu_counts[stu_counts >= least_respone_num].index.tolist()
    train = train[train["UserId"].isin(valid_students)]
    if train.empty:
        raise ValueError("过滤后无数据")

    all_students = sorted(train["UserId"].unique().tolist())
    all_questions = sorted(train["QuestionId"].unique().tolist())
    all_subjects = set()
    for qid in question_subjects:
        if qid in set(all_questions):
            all_subjects.update(question_subjects[qid])
    all_subjects = sorted(all_subjects)

    if len(all_students) > stu_num:
        sc = train.groupby("UserId").size().sort_values(ascending=False)
        all_students = sc.head(stu_num).index.tolist()
        train = train[train["UserId"].isin(all_students)]
        all_students = sorted(all_students)
    if len(all_questions) > exer_num:
        ec = train.groupby("QuestionId").size().sort_values(ascending=False)
        all_questions = ec.head(exer_num).index.tolist()
        train = train[train["QuestionId"].isin(all_questions)]
        all_questions = sorted(all_questions)
        all_subjects = set()
        for qid in question_subjects:
            if qid in set(all_questions):
                all_subjects.update(question_subjects[qid])
        all_subjects = sorted(all_subjects)
    if len(all_subjects) > know_num:
        sub_counts = defaultdict(int)
        for qid in all_questions:
            for s in question_subjects.get(qid, []):
                sub_counts[s] += 1
        top = sorted(sub_counts.items(), key=lambda x: -x[1])[:know_num]
        all_subjects = sorted([s for s, _ in top])

    stu_map = {uid: i for i, uid in enumerate(all_students)}
    question_map = {qid: i for i, qid in enumerate(all_questions)}
    concept_map = {sid: k for k, sid in enumerate(all_subjects)}
    reverse_question_map = {i: qid for qid, i in question_map.items()}

    cnt_stu = len(stu_map)
    cnt_question = len(question_map)
    cnt_concept = len(concept_map)

    rows = []
    for _, r in train.iterrows():
        uid, qid, correct = int(r["UserId"]), int(r["QuestionId"]), int(r["IsCorrect"])
        if uid not in stu_map or qid not in question_map:
            continue
        rows.append([stu_map[uid], question_map[qid], correct])
    TotalData = np.array(rows, dtype=np.int64)

    q_matrix = np.zeros((cnt_question, cnt_concept), dtype=np.float64)
    for qid, exer_id in question_map.items():
        for sub_id in question_subjects.get(qid, []):
            if sub_id in concept_map:
                q_matrix[exer_id, concept_map[sub_id]] = 1

    print(
        "Final student number: {}, Final question number: {}, Final concept number: {}, Final response number: {}".format(
            cnt_stu, cnt_question, cnt_concept, len(TotalData)
        )
    )

    os.makedirs(RESULT_DATA, exist_ok=True)
    os.makedirs(DATA_OUT, exist_ok=True)
    np.savetxt(os.path.join(RESULT_DATA, "TotalData.csv"), TotalData, delimiter=",", fmt=["%d", "%d", "%d"])
    np.savetxt(os.path.join(RESULT_DATA, "q.csv"), q_matrix, delimiter=",")
    np.savetxt(os.path.join(DATA_OUT, "TotalData.csv"), TotalData, delimiter=",", fmt=["%d", "%d", "%d"])
    np.savetxt(os.path.join(DATA_OUT, "q.csv"), q_matrix, delimiter=",")

    reverse_concept_map = {k: sid for sid, k in concept_map.items()}
    config_map = {
        "stu_map": stu_map,
        "question_map": question_map,
        "concept_map": concept_map,
        "reverse_question_map": reverse_question_map,
        "reverse_concept_map": reverse_concept_map,
    }
    with open(os.path.join(RESULT_DATA, "map.pkl"), "wb") as f:
        pickle.dump(config_map, f)
    with open(os.path.join(DATA_OUT, "map.pkl"), "wb") as f:
        pickle.dump(config_map, f)
