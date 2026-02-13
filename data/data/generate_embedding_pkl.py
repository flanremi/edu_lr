# -*- coding: utf-8 -*-
"""
根据 convert_to_dfcd.py 的输出，生成 DFCD 所需的 embedding_*.pkl。

前置：先运行 convert_to_dfcd.py，得到 dfcd_format/ 下的 TotalData.csv、q.csv、
     map.pkl、question_texts.csv；并确保 subject_metadata.csv 存在（用于知识点名称）。

输出：dfcd_format/embedding_<模型名>.pkl，可直接用于 DFCD 训练。
      pkl 内键：student_embeddings, exercise_embeddings, knowledge_embeddings（与 DFCD utils 一致）。
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# 【代码内变量】按需修改
# =============================================================================
DIR_DATA = os.path.dirname(os.path.abspath(__file__))
# convert_to_dfcd 的输出目录（内含 TotalData.csv, q.csv, map.pkl, question_texts.csv）
DFCD_FORMAT_DIR = os.path.join(DIR_DATA, "dfcd_format")
# 知识点名称表（SubjectId -> Name），用于拼题目/学生文本
PATH_SUBJECT_CSV = os.path.join(DIR_DATA, "metadata", "subject_metadata.csv")

# 使用的嵌入模型：openai | BAAI | m3e | instructor（需与 DFCD 的 text_embedding_model 一致）
EMBEDDING_MODEL = "openai"

# 若需走代理，取消下面两行注释并改成你的代理地址
# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"


def _load_subject_names(path_subject_csv):
    """SubjectId -> Name"""
    if not os.path.isfile(path_subject_csv):
        return {}
    df = pd.read_csv(path_subject_csv)
    if "SubjectId" not in df.columns or "Name" not in df.columns:
        return {}
    return df.set_index("SubjectId")["Name"].astype(str).to_dict()


def _build_texts(dfcd_format_dir, path_subject_csv):
    """构建 knowledge_text, exercise_text, student_text（与 DFCD 预处理逻辑一致）。"""
    with open(os.path.join(dfcd_format_dir, "map.pkl"), "rb") as f:
        config_map = pickle.load(f)
    TotalData = pd.read_csv(
        os.path.join(dfcd_format_dir, "TotalData.csv"),
        header=None,
        names=["stu", "exer", "answervalue"],
    )
    q_np = pd.read_csv(os.path.join(dfcd_format_dir, "q.csv"), header=None).to_numpy()
    prob_num, know_num = q_np.shape
    stu_num = TotalData["stu"].max() + 1

    subject_names = _load_subject_names(path_subject_csv)
    reverse_concept_map = config_map["reverse_concept_map"]  # know_id -> SubjectId

    # 1) knowledge_text: 每个知识点一行文本（用名称，无则用 SubjectId）
    knowledge_text = []
    for k in range(know_num):
        sid = reverse_concept_map.get(k, k)
        name = subject_names.get(sid, str(sid))
        knowledge_text.append(name)
    config = {"knowledge_text": knowledge_text}

    # 2) exercise_text: 按 exercise_id 顺序，来自 question_texts.csv
    qt_path = os.path.join(dfcd_format_dir, "question_texts.csv")
    if not os.path.isfile(qt_path):
        raise FileNotFoundError("请先运行 convert_to_dfcd.py 并保留 question_texts.csv")
    qt = pd.read_csv(qt_path)
    qt = qt.sort_values("exercise_id")
    exercise_text = qt["text"].fillna("").astype(str).tolist()
    if len(exercise_text) != prob_num:
        raise ValueError(f"question_texts 行数 {len(exercise_text)} 与题目数 {prob_num} 不一致")
    config["exercise_text"] = exercise_text

    # 3) student_text: 每个学生一个 list，顺序与 TotalData 中该学生的答题顺序一致（供 se_map 索引）
    prompt_right = "I was asked the question: {question}.\nAnd this question is about: {Name}.\nAnd I give the correct answer."
    prompt_wrong = "I was asked the question: {question}.\nAnd this question is about: {Name}.\nBut I give the wrong answer."
    exer_id_to_text = dict(zip(qt["exercise_id"], qt["text"].fillna("").astype(str)))

    def concept_names_for_exer(exer_id):
        row = q_np[exer_id]
        names = []
        for k in np.where(row != 0)[0]:
            sid = reverse_concept_map.get(k, k)
            names.append(subject_names.get(sid, str(sid)))
        return names if names else ["General"]

    student_text = []
    for stu_id in tqdm(range(stu_num), desc="Building student text"):
        logs = TotalData.loc[TotalData["stu"] == stu_id]
        texts = []
        for _, row in logs.iterrows():
            exer_id, correct = int(row["exer"]), int(row["answervalue"])
            question = exer_id_to_text.get(exer_id, "")
            name = concept_names_for_exer(exer_id)
            name_str = ", ".join(name)
            if correct == 1:
                texts.append(prompt_right.format(question=question, Name=name_str))
            else:
                texts.append(prompt_wrong.format(question=question, Name=name_str))
        student_text.append(texts)
    config["student_text"] = student_text
    return config


def _embed_openai(config, out_path):
    from langchain.embeddings import OpenAIEmbeddings
    model = OpenAIEmbeddings()
    config["knowledge_embeddings"] = np.array(model.embed_documents(config["knowledge_text"]), dtype=np.float64)
    config["exercise_embeddings"] = np.array(model.embed_documents(config["exercise_text"]), dtype=np.float64)
    config["student_embeddings"] = [
        np.array(model.embed_documents(tex), dtype=np.float64) for tex in tqdm(config["student_text"], desc="Student embed")
    ]
    to_save = {
        "student_embeddings": config["student_embeddings"],
        "exercise_embeddings": config["exercise_embeddings"],
        "knowledge_embeddings": config["knowledge_embeddings"],
    }
    with open(out_path, "wb") as f:
        pickle.dump(to_save, f)


def _embed_baai(config, out_path):
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    config["knowledge_embeddings"] = np.array(
        model.encode(config["knowledge_text"], batch_size=12, max_length=8192)["dense_vecs"], dtype=np.float64
    )
    config["exercise_embeddings"] = np.array(
        model.encode(config["exercise_text"], batch_size=12, max_length=8192)["dense_vecs"], dtype=np.float64
    )
    config["student_embeddings"] = [
        np.array(model.encode(tex, batch_size=12, max_length=8192)["dense_vecs"], dtype=np.float64)
        for tex in tqdm(config["student_text"], desc="Student embed")
    ]
    to_save = {
        "student_embeddings": config["student_embeddings"],
        "exercise_embeddings": config["exercise_embeddings"],
        "knowledge_embeddings": config["knowledge_embeddings"],
    }
    with open(out_path, "wb") as f:
        pickle.dump(to_save, f)


def _embed_m3e(config, out_path):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("moka-ai/m3e-base")
    config["knowledge_embeddings"] = np.array(model.encode(config["knowledge_text"]), dtype=np.float64)
    config["exercise_embeddings"] = np.array(model.encode(config["exercise_text"]), dtype=np.float64)
    config["student_embeddings"] = [
        np.array(model.encode(tex), dtype=np.float64) for tex in tqdm(config["student_text"], desc="Student embed")
    ]
    to_save = {
        "student_embeddings": config["student_embeddings"],
        "exercise_embeddings": config["exercise_embeddings"],
        "knowledge_embeddings": config["knowledge_embeddings"],
    }
    with open(out_path, "wb") as f:
        pickle.dump(to_save, f)


def _embed_instructor(config, out_path):
    from InstructorEmbedding import INSTRUCTOR
    model = INSTRUCTOR("hkunlp/instructor-base")
    k_text = [["Represent the knowledge title:", t] for t in config["knowledge_text"]]
    config["knowledge_embeddings"] = np.array(model.encode(k_text), dtype=np.float64)
    e_text = [["Represent the exercise description:", t] for t in config["exercise_text"]]
    config["exercise_embeddings"] = np.array(model.encode(e_text), dtype=np.float64)
    config["student_embeddings"] = []
    for tex in tqdm(config["student_text"], desc="Student embed"):
        t_in = [["Represent the student response log:", t] for t in tex]
        config["student_embeddings"].append(np.array(model.encode(t_in), dtype=np.float64))
    to_save = {
        "student_embeddings": config["student_embeddings"],
        "exercise_embeddings": config["exercise_embeddings"],
        "knowledge_embeddings": config["knowledge_embeddings"],
    }
    with open(out_path, "wb") as f:
        pickle.dump(to_save, f)


def run():
    model_name = EMBEDDING_MODEL.strip().lower()
    if model_name == "openai":
        model_name = "openai"
    elif model_name == "baai":
        model_name = "BAAI"
    elif model_name in ("m3e", "instructor"):
        pass
    else:
        raise ValueError("EMBEDDING_MODEL 须为 openai | BAAI | m3e | instructor")

    out_path = os.path.join(DFCD_FORMAT_DIR, f"embedding_{model_name}.pkl")
    print("构建题目/知识点/学生文本...")
    config = _build_texts(DFCD_FORMAT_DIR, PATH_SUBJECT_CSV)
    print("调用嵌入模型并写入 pkl...")
    if EMBEDDING_MODEL.lower() == "openai":
        _embed_openai(config, out_path)
    elif EMBEDDING_MODEL.lower() == "baai":
        _embed_baai(config, out_path)
    elif EMBEDDING_MODEL.lower() == "m3e":
        _embed_m3e(config, out_path)
    elif EMBEDDING_MODEL.lower() == "instructor":
        _embed_instructor(config, out_path)
    else:
        raise ValueError("EMBEDDING_MODEL 须为 openai | BAAI | m3e | instructor")
    print("完成。输出:", os.path.abspath(out_path))
    print("将 dfcd_format 拷到 DFCD 的 data/<数据集名>/ 后，设置 text_embedding_model 为:", model_name)


if __name__ == "__main__":
    run()
