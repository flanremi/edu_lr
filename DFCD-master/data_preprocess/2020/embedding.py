# -*- coding: utf-8 -*-
"""
2020 数据集嵌入：从 data/2020/ 读入，使用远端 Embedding API 生成向量，写出 embedding_*.pkl。

- 数据来源：DFCD-master/data/2020/（TotalData.csv, q.csv, map.pkl, question_texts.csv 等）
- Embedding：仅使用远端 API（与 embedding_helper 一致），POST {url}/embedding/，请求体 {"input": [...], "model": "..."}
- 配置：.env 中 EMBEDDING_SERVICE_URL、EMBEDDING_MODEL
"""
import json
import os
import pickle
import shutil
import time
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

# 从 data_preprocess/.env 加载配置
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ===================== 远端 Embedding API（与 embedding_helper 一致） =====================
EMBEDDING_SERVICE_URL = (os.getenv("EMBEDDING_SERVICE_URL", "https://ai.smartedu.work").strip() or "https://ai.smartedu.work").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gte-multilingual-base")
# 每批最多多少条文本发一次请求，避免超时或过长
REMOTE_EMBED_BATCH_SIZE = int(os.getenv("REMOTE_EMBED_BATCH_SIZE", "64"))
# 并发请求的线程数
REMOTE_EMBED_MAX_WORKERS = int(os.getenv("REMOTE_EMBED_MAX_WORKERS", "64"))
# 单批请求失败时的重试次数（SSL/网络抖动时可自动重试）
REMOTE_EMBED_RETRIES = int(os.getenv("REMOTE_EMBED_RETRIES", "3"))
# ============================================================================

# 断点续跑：任务状态与每批输入/输出存放目录（在 result/embedding 下）
TASK_DIR_NAME = "embedding_remote_task"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

DATASET = "2020"
_DATA_PREPROCESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(os.path.dirname(_DATA_PREPROCESS_ROOT), "data", DATASET)
RESULT_EMBED = os.path.join(os.path.dirname(__file__), "result", "embedding")


def _embed_one_batch(batch: list, model: str, url: str) -> list:
    """
    单批请求远端 API，返回该批的 embedding 列表。供线程池调用。
    对 SSL/连接/超时类错误自动重试 REMOTE_EMBED_RETRIES 次，仍失败则抛出并给出可读说明。
    """
    payload = {"input": batch, "model": model}
    last_err = None
    for attempt in range(REMOTE_EMBED_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=DEFAULT_HEADERS, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            if len(items) != len(batch):
                raise ValueError(f"API 返回条数 {len(items)} 与请求条数 {len(batch)} 不一致")
            return [item["embedding"] for item in items]
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            if attempt < REMOTE_EMBED_RETRIES:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            # 重试耗尽，给出可读说明
            hint = (
                "  (SSL/连接错误多为网络或服务端不稳定，可：1) 稍后重试 2) 降低 REMOTE_EMBED_MAX_WORKERS 减轻并发 3) 检查 EMBEDDING_SERVICE_URL 与网络)"
            )
            raise RuntimeError(f"远端 Embedding 请求在重试 {REMOTE_EMBED_RETRIES} 次后仍失败: {e}{hint}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"远端 Embedding HTTP 错误: {e}") from e
    assert last_err is not None
    raise RuntimeError(f"远端 Embedding 失败: {last_err}") from last_err


def _task_dir():
    return os.path.join(RESULT_EMBED, TASK_DIR_NAME)


def _meta_path():
    return os.path.join(_task_dir(), "meta.json")


def _stage_dir(stage_name: str) -> str:
    return os.path.join(_task_dir(), stage_name)


def _batch_in_path(stage_dir: str, idx: int) -> str:
    return os.path.join(stage_dir, f"batch_{idx:06d}.in.pkl")


def _batch_out_path(stage_dir: str, idx: int) -> str:
    return os.path.join(stage_dir, f"batch_{idx:06d}.out.npy")


def _load_meta() -> dict:
    p = _meta_path()
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_meta(meta: dict):
    os.makedirs(_task_dir(), exist_ok=True)
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _meta_matches(meta: dict, batch_size: int, knowledge_total: int, exercise_total: int, student_total: int) -> bool:
    if not meta or meta.get("batch_size") != batch_size:
        return False
    for stage, total in [
        ("knowledge", knowledge_total),
        ("exercise", exercise_total),
        ("student", student_total),
    ]:
        s = meta.get(stage)
        if not s or s.get("total") != total:
            return False
    return True


def _prepare_stage(stage_name: str, texts: list, batch_size: int) -> int:
    """将本阶段所有批次写入本地 .in.pkl，返回批次数。"""
    stage_d = _stage_dir(stage_name)
    os.makedirs(stage_d, exist_ok=True)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    for idx, batch in enumerate(batches):
        with open(_batch_in_path(stage_d, idx), "wb") as f:
            pickle.dump(batch, f)
    return len(batches)


def _run_one_batch_from_disk(args) -> int:
    """从磁盘读入一批、请求 API、写回 .out.npy。参数 (batch_idx, stage_dir, model, url)。返回 batch_idx。"""
    batch_idx, stage_dir, model, url = args
    with open(_batch_in_path(stage_dir, batch_idx), "rb") as f:
        batch = pickle.load(f)
    emb = _embed_one_batch(batch, model, url)
    arr = np.array(emb, dtype=np.float64)
    np.save(_batch_out_path(stage_dir, batch_idx), arr)
    return batch_idx


def _run_stage_resumable(
    stage_name: str,
    num_batches: int,
    desc: str,
    model: str,
    url: str,
    max_workers: int,
) -> np.ndarray:
    """
    本阶段：已完成批次从磁盘加载，未完成批次并发请求并写回磁盘；进度条按「每完成一批」更新。
    """
    stage_d = _stage_dir(stage_name)
    done = [i for i in range(num_batches) if os.path.isfile(_batch_out_path(stage_d, i))]
    todo = [i for i in range(num_batches) if i not in done]
    if todo:
        pbar = tqdm(total=num_batches, initial=len(done), desc=desc, unit="batch")
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_run_one_batch_from_disk, (i, stage_d, model, url)): i for i in todo
                }
                for future in as_completed(future_to_idx):
                    try:
                        future.result()
                    except Exception as e:
                        idx = future_to_idx[future]
                        err_msg = (
                            f"远端 Embedding 失败 [阶段={stage_name} 批次 {idx + 1}/{num_batches}] "
                            f"URL={url}\n  异常: {type(e).__name__}: {e}"
                        )
                        print(err_msg, flush=True)
                        pbar.close()
                        raise RuntimeError(err_msg) from e
                    pbar.update(1)
        finally:
            pbar.close()
    # 按顺序加载所有批次的 .out.npy 并拼成一大块
    parts = []
    for i in range(num_batches):
        arr = np.load(_batch_out_path(stage_d, i))
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def _load_subject_names():
    """从 data/2020/subject_metadata.csv 或 2020/data/metadata/subject_metadata.csv 读 SubjectId -> Name"""
    for path in [
        os.path.join(DATA_OUT, "subject_metadata.csv"),
        os.path.join(os.path.dirname(__file__), "data", "metadata", "subject_metadata.csv"),
    ]:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if "SubjectId" in df.columns and "Name" in df.columns:
                return df.set_index("SubjectId")["Name"].astype(str).to_dict()
    return {}


def _build_texts_from_convert_format():
    """
    从 data/2020/ 下的 convert 输出构建 knowledge_text, exercise_text, student_text。
    使用论文中的 prompt 逻辑（纯 str.format，不依赖 LangChain）。
    """
    with open(os.path.join(DATA_OUT, "map.pkl"), "rb") as f:
        config_map = pickle.load(f)
    TotalData = pd.read_csv(
        os.path.join(DATA_OUT, "TotalData.csv"),
        header=None,
        names=["stu", "exer", "answervalue"],
    )
    q_np = pd.read_csv(os.path.join(DATA_OUT, "q.csv"), header=None).to_numpy()
    prob_num, know_num = q_np.shape
    stu_num = int(TotalData["stu"].max()) + 1

    subject_names = _load_subject_names()
    reverse_concept = config_map.get("reverse_concept_map") or {
        v: k for k, v in config_map["concept_map"].items()
    }

    knowledge_text = [
        subject_names.get(reverse_concept.get(k, k), str(reverse_concept.get(k, k)))
        for k in range(len(reverse_concept))
    ]
    config = {"knowledge_text": knowledge_text}

    path_qt = os.path.join(DATA_OUT, "question_texts.csv")
    if not os.path.isfile(path_qt):
        raise FileNotFoundError("data/2020/ 下缺少 question_texts.csv，请先运行 convert_to_dfcd.py 并拷贝到 data/2020/")
    qt = pd.read_csv(path_qt).sort_values("exercise_id")
    if "text" not in qt.columns:
        raise ValueError("question_texts.csv 需包含 text 列")
    exercise_text = qt["text"].fillna("").astype(str).tolist()
    if len(exercise_text) != prob_num:
        raise ValueError("question_texts 行数 %d 与题目数 %d 不一致" % (len(exercise_text), prob_num))
    config["exercise_text"] = exercise_text

    def concept_names_for_exer(exer_id):
        row = q_np[exer_id]
        names = []
        for k in range(len(row)):
            if row[k] != 0:
                sid = reverse_concept.get(k, k)
                names.append(subject_names.get(sid, str(sid)))
        return names if names else ["General"]

    prompt_right = "I was asked the question: {question}.\nAnd this question is about: {Name}.\nAnd I give the correct answer."
    prompt_wrong = "I was asked the question: {question}.\nAnd this question is about: {Name}.\nBut I give the wrong answer."
    exer_to_text = dict(zip(qt["exercise_id"], qt["text"].fillna("").astype(str)))

    student_text = []
    for stu_id in tqdm(range(stu_num), desc="Building student text"):
        logs = TotalData.loc[TotalData["stu"] == stu_id]
        texts = []
        for _, row in logs.iterrows():
            exer_id, correct = int(row["exer"]), int(row["answervalue"])
            question = exer_to_text.get(exer_id, "")
            name_str = ", ".join(concept_names_for_exer(exer_id))
            if correct == 1:
                texts.append(prompt_right.format(question=question, Name=name_str))
            else:
                texts.append(prompt_wrong.format(question=question, Name=name_str))
        student_text.append(texts)
    config["student_text"] = student_text
    return config


def _generate_embeddings_remote(config):
    """
    嵌入前先把每批任务写入本地；每完成一批更新进度并写回 .out.npy。
    启动时检测上次任务是否完成，未完成则从断点继续。
    """
    batch_size = REMOTE_EMBED_BATCH_SIZE
    url = f"{EMBEDDING_SERVICE_URL}/embedding/"
    model = EMBEDDING_MODEL
    max_workers = REMOTE_EMBED_MAX_WORKERS

    knowledge_text = config["knowledge_text"]
    exercise_text = config["exercise_text"]
    student_lens = [len(tex) for tex in config["student_text"]]
    all_student_texts = [t for tex in config["student_text"] for t in tex]

    n_k = (len(knowledge_text) + batch_size - 1) // batch_size
    n_e = (len(exercise_text) + batch_size - 1) // batch_size
    n_s = (len(all_student_texts) + batch_size - 1) // batch_size

    meta = _load_meta()
    if _meta_matches(meta, batch_size, len(knowledge_text), len(exercise_text), len(all_student_texts)):
        print("检测到未完成的上次任务，将从中断处继续。任务目录:", _task_dir())
        print("  (若需从头重跑，可删除该目录后重新运行)")
    else:
        # 全新任务或数据变化：清空任务目录，写入所有批次的输入
        task_d = _task_dir()
        if os.path.isdir(task_d):
            shutil.rmtree(task_d)
        os.makedirs(task_d, exist_ok=True)
        print("嵌入开始前，将本阶段所有批次写入本地:", task_d)
        _prepare_stage("knowledge", knowledge_text, batch_size)
        _prepare_stage("exercise", exercise_text, batch_size)
        _prepare_stage("student", all_student_texts, batch_size)
        _save_meta({
            "batch_size": batch_size,
            "knowledge": {"total": len(knowledge_text), "num_batches": n_k},
            "exercise": {"total": len(exercise_text), "num_batches": n_e},
            "student": {"total": len(all_student_texts), "num_batches": n_s},
        })
        print("批次写入完成。开始请求远端 API，每完成一批会更新进度并保存。")

    print("使用远端 Embedding API:", EMBEDDING_SERVICE_URL, "模型:", model, "并发:", max_workers, "线程")

    config["knowledge_embeddings"] = _run_stage_resumable("knowledge", n_k, "knowledge 嵌入", model, url, max_workers)
    config["exercise_embeddings"] = _run_stage_resumable("exercise", n_e, "exercise 嵌入", model, url, max_workers)
    flat_student_emb = _run_stage_resumable("student", n_s, "student 嵌入", model, url, max_workers)

    offset = 0
    config["student_embeddings"] = []
    for n in student_lens:
        config["student_embeddings"].append(flat_student_emb[offset : offset + n])
        offset += n

    to_save = {
        "student_embeddings": config["student_embeddings"],
        "exercise_embeddings": config["exercise_embeddings"],
        "knowledge_embeddings": config["knowledge_embeddings"],
    }
    os.makedirs(DATA_OUT, exist_ok=True)
    os.makedirs(RESULT_EMBED, exist_ok=True)
    out_name = "embedding_remote.pkl"
    with open(os.path.join(DATA_OUT, out_name), "wb") as f:
        pickle.dump(to_save, f)
    with open(os.path.join(RESULT_EMBED, out_name), "wb") as f:
        pickle.dump(to_save, f)
    print("已写入:", os.path.join(DATA_OUT, out_name))
    print("DFCD 使用时请将 data_params_dict 中 2020 的 text_embedding_model 设为 remote，并设置 in_channels_llm 为远端模型维度（如 gte-multilingual-base 为 768）。")


def run(arg):
    if not os.path.isfile(os.path.join(DATA_OUT, "TotalData.csv")):
        raise FileNotFoundError(
            "data/2020/ 下缺少 TotalData.csv。请先将 convert_to_dfcd 的输出拷贝到 DFCD-master/data/2020/，"
            "或在 preprocess 中设置 CONVERT_OUTPUT_DIR 后运行 main_filter.py --dataset 2020 ..."
        )
    config = _build_texts_from_convert_format()
    _generate_embeddings_remote(config)


if __name__ == "__main__":
    run({})
