import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


def _is_missing(val):
    """判断值是否缺失（None、NaN、空字符串或纯空白）"""
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return len(s) == 0 or s.lower() == 'nan'


def validate_preprocess_data(response_logs, response_data, TotalData, config_map):
    """
    校验预处理后的数据是否存在缺失项，有则输出。
    返回缺失项数量。
    """
    issues = []

    # 1. 校验 response_logs 关键列 NaN（高效方式）
    for col in ['UserId', 'QuestionId', 'IsCorrect', 'Name']:
        if col in response_logs.columns:
            null_count = response_logs[col].isna().sum()
            if null_count > 0:
                sample_idx = response_logs[response_logs[col].isna()].index.tolist()[:20]
                issues.append(f"[response_logs] 列 {col} 有 {null_count} 处 NaN，示例行: {sample_idx}")

    # 2. 校验 response_data
    for (uid, question, concept), response in list(response_data.items())[:10000]:
        if _is_missing(uid):
            issues.append(f"[response_data] uid={uid} 缺失")
        if _is_missing(question):
            issues.append(f"[response_data] question={question} 缺失")
        if _is_missing(concept):
            issues.append(f"[response_data] (uid={uid},q={question}) concept 缺失")
        if _is_missing(response):
            issues.append(f"[response_data] (uid={uid},q={question}) response 缺失")

    # 3. 校验 TotalData 每条记录
    reverse_q = config_map.get('reverse_question_map', {})
    for i, row in enumerate(TotalData):
        if len(row) < 3:
            issues.append(f"[TotalData 行{i}] 列数不足: {row}")
            continue
        stu, exer, answervalue = row[0], row[1], row[2]
        if exer not in reverse_q:
            issues.append(f"[TotalData 行{i}] exer={exer} 不在 reverse_question_map")
        if _is_missing(answervalue) and answervalue not in (0, 1):
            issues.append(f"[TotalData 行{i}] answervalue={answervalue} 异常")

    if issues:
        print("\n===== 预处理数据缺失项校验 =====", flush=True)
        for msg in issues[:100]:
            print(f"  {msg}", flush=True)
        if len(issues) > 100:
            print(f"  ... 还有 {len(issues) - 100} 条未显示", flush=True)
        print(f"共 {len(issues)} 项缺失\n", flush=True)
    else:
        print("\n[校验] 预处理数据未发现缺失项\n", flush=True)

    return len(issues)


def fix_seeds(seed=101):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
    np.random.seed(seed)

def run(config):
    fix_seeds(config['seed'])
    response_logs = pd.read_csv('../data/2020/merged_data.csv')
    questions = pd.read_csv("../data/2020/question_texts.csv")

    # 只保留 content 非空且有效的题目，移除“不存在”的题目（content 缺失或为空）
    valid_question_ids = set()
    for _, row in questions.iterrows():
        qid = row.get('id', row.iloc[0] if len(row) > 0 else None)
        content = row.get('content', row.iloc[1] if len(row) > 1 else "")
        if not _is_missing(qid) and not _is_missing(content):
            try:
                valid_question_ids.add(int(qid))
            except (ValueError, TypeError):
                valid_question_ids.add(qid)
    all_qids = set()
    for x in questions['id'].dropna().unique():
        try:
            all_qids.add(int(x))
        except (ValueError, TypeError):
            all_qids.add(x)
    excluded = all_qids - valid_question_ids
    if excluded:
        try:
            preview = sorted(excluded)[:20]
        except TypeError:
            preview = list(excluded)[:20]
        print(f'[过滤] 移除 {len(excluded)} 个 content 缺失的题目: {preview}{"..." if len(excluded) > 20 else ""}')

    response_logs = response_logs.loc[response_logs['QuestionId'].isin(valid_question_ids)]
    grouped = response_logs.groupby('UserId').size()
    grouped = grouped.loc[grouped > config['least_respone_num']]
    response_logs = response_logs.loc[response_logs['UserId'].isin(grouped.index)]

    original_stu_map = {}
    cnt = 0
    for stu in response_logs["UserId"].unique():
        original_stu_map[cnt] = stu
        cnt += 1
    
    response_data = dict()
    stu_response_data = dict()
    missing_in_filter = []  # 收集 Filter 阶段的缺失项
    stu_num = np.random.choice(np.arange(len(response_logs["UserId"].unique())), size=config['stu_num'], replace=False)
    least_respone_num = 0

    for stu in tqdm(stu_num, desc='Filter student'):
        for data in response_logs.loc[response_logs['UserId'] == original_stu_map[stu]].values:
            uid, question, response, concept = data[1], data[0], data[3], data[7] if len(data) > 7 else ""
            if _is_missing(uid):
                missing_in_filter.append(f"[Filter] UserId 缺失: data={data[:4]}")
            if _is_missing(question):
                missing_in_filter.append(f"[Filter] QuestionId 缺失: uid={uid}")
            if _is_missing(response) and len(missing_in_filter) < 50:
                missing_in_filter.append(f"[Filter] IsCorrect 缺失: uid={uid} q={question}")
            if _is_missing(concept) and len(missing_in_filter) < 50:
                missing_in_filter.append(f"[Filter] Name 缺失: uid={uid} q={question}")
            tmp_data = (uid, question, concept)
            response_data[tmp_data] = response
        

    for key, value in tqdm(response_data.items(), desc='Filter the least respone number'):
        if key[0] not in stu_response_data:
            stu_response_data[key[0]] = []
        stu_response_data[key[0]].append([int(key[0]), int(key[1]), int(value), key[2]])
    
    TotalData = []
    stu_map = dict()
    cnt_stu = 0
    question_set = set()
    cnt_question = 0
    question_map = dict()
    concept_set = set()
    cnt_concept = 0
    concept_map = dict()
    reverse_question_map = {}

    for key, value in tqdm(stu_response_data.items(), desc='Remap student_id, question_id and concept_id'):
        if len(value) >= least_respone_num:
            stu_map[key] = cnt_stu
            cnt_stu += 1
            for data in value:
                question_set.add(data[1])
                concept = data[3]
                concept_set.add(concept)

    for question in question_set:
        question_map[question] = cnt_question
        reverse_question_map[cnt_question] = question
        cnt_question += 1

    for concept in concept_set:
        concept_map[concept] = cnt_concept
        cnt_concept += 1
    
    TotalData = []
    q_matrix = np.zeros((cnt_question, cnt_concept))

    for key, value in tqdm(stu_response_data.items(), desc='Construct final data'):
        if len(value) >= least_respone_num:
            for data in value:
                TotalData.append([stu_map[data[0]], question_map[data[1]], data[2]])
                concept = data[3]
                q_matrix[question_map[data[1]]][concept_map[concept]] = 1
    
    print('Final student number: {}, Final question number: {}, Final concept number: {}, Final response number: {}'.format(cnt_stu, cnt_question, cnt_concept, len(TotalData)))
    np.savetxt('../data/2020/TotalData.csv', TotalData, delimiter=',')
    np.savetxt('../data/2020/q.csv', q_matrix, delimiter=',')
    # np.savetxt('../../data/NeurIPS2020/TotalData.csv', TotalData, delimiter=',')
    # np.savetxt('../data/NeurIPS2020/q.csv', q_matrix, delimiter=',')
    config_map = {}
    config_map['stu_map'] = stu_map
    config_map['question_map'] = question_map
    config_map['concept_map'] = concept_map
    config_map['reverse_question_map'] = reverse_question_map

    # 校验合成数据是否存在缺失项
    if missing_in_filter:
        print("\n===== Filter 阶段缺失项 =====", flush=True)
        for msg in missing_in_filter[:50]:
            print(f"  {msg}", flush=True)
        if len(missing_in_filter) > 50:
            print(f"  ... 还有 {len(missing_in_filter) - 50} 条", flush=True)
        print()
    n_issues = validate_preprocess_data(response_logs, response_data, TotalData, config_map)
    if n_issues > 0 or missing_in_filter:
        print(f"[校验] 共发现 {n_issues + len(missing_in_filter)} 处缺失项，请检查上方输出。")

    with open('../data/2020/map.pkl', 'wb') as f:
        pickle.dump(config_map, f)
    # with open('../data/NeurIPS2020/map.pkl', 'wb') as f:
    #     pickle.dump(config_map, f)

