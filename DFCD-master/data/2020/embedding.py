"""
嵌入脚本 - 使用远端 API 获取文本嵌入
直接运行: python embedding.py（需在 data_preprocess 目录下执行，或根据 BASE_DIR 调整路径）
"""
import os
import pickle
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============ 配置变量（直接修改此处） ============
EMBEDDING_API_URL = "https://ai.smartedu.work/embedding/"
EMBEDDING_MODEL = "gte-multilingual-base"
BATCH_SIZE = 512
OUTPUT_FILENAME = "embedding_remote.pkl"  # 输出 pkl 文件名

# 数据路径（脚本位于 data_preprocess/NeurIPS2020/embedding.py）
SCRIPT_DIR = Path(__file__).parent  # data/2020
MAP_PATH = SCRIPT_DIR / "map.pkl"
TOTALDATA_PATH = SCRIPT_DIR / "TotalData.csv"
RESPONSE_LOGS_PATH = SCRIPT_DIR / "merged_data.csv"
QUESTIONS_PATH = SCRIPT_DIR / "question_texts.csv"
OUTPUT_PATH_PROJECT = SCRIPT_DIR / OUTPUT_FILENAME
OUTPUT_PATH_LOCAL = SCRIPT_DIR / "embedding" / OUTPUT_FILENAME
# ================================================


def generate_text(config, config_map, TotalData, response_logs, questions):
    student_prompt_template_right = "I was asked the question: {question}.\nAnd this question is about: {Name}.\n.And I give the correct answer."
    student_prompt_template_wrong = "I was asked the question: {question}.\nAnd this question is about: {Name}.\n.But I give the wrong answer."
    question_template = "The question's content is: {content} and it is about: {tag}."

    knowledge_original = []
    for concept in tqdm(config_map['concept_map'], desc="knowledge"):
        knowledge_original.append(str(concept))
    config["knowledge_text"] = knowledge_original

    # 预建 O(1) 查找字典，避免循环内重复 DataFrame 扫描
    question_content = dict(zip(questions['id'], questions['content']))
    qid_to_name = response_logs.drop_duplicates('QuestionId').set_index('QuestionId')['Name'].to_dict()

    exercise_original = []
    for question in tqdm(config_map['question_map'], desc="exercise"):
        content = question_content[question]
        tag = qid_to_name[question]
        exercise_original.append(question_template.format(content=content, tag=tag))
    config["exercise_text"] = exercise_original

    # 按学生分组一次，避免对每个学生都 loc 扫描 TotalData
    student_groups = {k: v for k, v in TotalData.groupby('stu')}
    reverse_q = config_map['reverse_question_map']

    student_original = []
    for student in tqdm(range(len(config_map['stu_map'])), desc="student"):
        tmp = []
        student_logs = student_groups.get(student, pd.DataFrame(columns=['stu', 'exer', 'answervalue']))
        for log in student_logs.values:
            q_id = reverse_q[log[1]]
            question = question_content[q_id]
            Name = qid_to_name[q_id]
            if log[2] == 1:
                tmp.append(student_prompt_template_right.format(question=question, Name=Name))
            else:
                tmp.append(student_prompt_template_wrong.format(question=question, Name=Name))
        student_original.append(tmp)
    config['student_text'] = student_original

    return config


def embed_texts_batch(texts, batch_size=BATCH_SIZE):
    """分批请求远端 API 获取嵌入，每批 batch_size 条"""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
        batch = texts[i:i + batch_size]
        payload = {"input": batch, "model": EMBEDDING_MODEL}
        try:
            resp = requests.post(EMBEDDING_API_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            for item in sorted(data.get("data", []), key=lambda x: x["index"]):
                all_embeddings.append(item["embedding"])
        except requests.RequestException as e:
            raise RuntimeError(f"API 请求失败: {e}") from e
    return all_embeddings


def generate_embeddings_remote(config):
    config["knowledge_embeddings"] = embed_texts_batch(config["knowledge_text"])

    config["exercise_embeddings"] = embed_texts_batch(config["exercise_text"])

    config["student_embeddings"] = []
    flat_texts = []
    flat_indices = []
    for student_text in config['student_text']:
        start = len(flat_texts)
        flat_texts.extend(student_text)
        flat_indices.append((start, len(flat_texts)))

    if flat_texts:
        all_embs = embed_texts_batch(flat_texts)
        for start, end in flat_indices:
            config["student_embeddings"].append(all_embs[start:end])
    else:
        config["student_embeddings"] = [[] for _ in config['student_text']]

    OUTPUT_PATH_PROJECT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH_LOCAL.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH_PROJECT, 'wb') as f:
        pickle.dump(config, f)
    with open(OUTPUT_PATH_LOCAL, 'wb') as f:
        pickle.dump(config, f)
    print(f"已保存: {OUTPUT_PATH_PROJECT}")
    print(f"已保存: {OUTPUT_PATH_LOCAL}")


def main():
    with open(MAP_PATH, 'rb') as f:
        config_map = pickle.load(f)

    TotalData = pd.read_csv(TOTALDATA_PATH, header=None, names=['stu', 'exer', 'answervalue'])
    response_logs = pd.read_csv(RESPONSE_LOGS_PATH)
    questions = pd.read_csv(QUESTIONS_PATH)

    response_logs = response_logs.loc[response_logs['QuestionId'].isin(questions['id'].unique())]
    grouped = response_logs.groupby('UserId').size()
    grouped = grouped.loc[grouped > 50]
    response_logs = response_logs.loc[response_logs['UserId'].isin(grouped.index)]

    config = {}
    config = generate_text(config, config_map, TotalData, response_logs, questions)
    generate_embeddings_remote(config)


if __name__ == "__main__":
    main()
