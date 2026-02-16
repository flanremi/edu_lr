"""
数据集整合脚本
生成 merged_data 数据集，结构为：
QuestionId, UserId, AnswerId, IsCorrect, CorrectAnswer, AnswerValue, SubjectId, Name

用法:
    cd data/NeurIPS2020
    python merge_data.py
    # 或指定输出文件: python merge_data.py --output merged_data_task_3_4.csv
"""
import os
import ast
import pandas as pd
from pathlib import Path


def parse_subject_ids(subject_str):
    """
    解析 question_metadata 中的 SubjectId 字符串，如 "[3, 71, 98, 209]"
    返回 subject id 列表；若解析失败返回空列表
    """
    if pd.isna(subject_str):
        return []
    if isinstance(subject_str, str):
        try:
            return ast.literal_eval(subject_str)
        except (ValueError, SyntaxError):
            return []
    return []


def load_subject_name_map(subject_metadata_path):
    """加载 SubjectId -> Name 映射"""
    df = pd.read_csv(subject_metadata_path)
    return dict(zip(df['SubjectId'].astype(int), df['Name']))


def build_question_subject_map(question_metadata_path, use_last=True):
    """
    构建 QuestionId -> SubjectId 映射
    question_metadata 中 SubjectId 为数组如 [3, 71, 98, 209]，表示知识点路径
    use_last=True: 取最后一个（最细粒度知识点）
    use_last=False: 取第一个
    """
    df = pd.read_csv(question_metadata_path)
    q2s = {}
    for _, row in df.iterrows():
        qid = row['QuestionId']
        subject_ids = parse_subject_ids(row['SubjectId'])
        if subject_ids:
            sid = subject_ids[-1] if use_last else subject_ids[0]
            q2s[qid] = int(sid)
    return q2s


def merge_response_data(response_df, question_subject_map, subject_name_map):
    """
    为答题数据添加 SubjectId 和 Name 列
    """
    response_df = response_df.copy()
    response_df['SubjectId'] = response_df['QuestionId'].map(question_subject_map)
    response_df['Name'] = response_df['SubjectId'].map(subject_name_map)

    # 过滤掉无法匹配到 Subject 的记录（可选，也可保留并填 NaN）
    before = len(response_df)
    merged = response_df.dropna(subset=['SubjectId', 'Name'])
    after = len(merged)
    if before > after:
        print(f"  警告: {before - after} 条记录因无法匹配 Subject 被排除")

    return merged


def main(
    data_dir=None,
    output_file='merged_data.csv',
    train_file='train_task_3_4.csv',
    test_file='test_public_task_4_more_splits.csv',
    subject_file='subject_metadata.csv',
    question_file='question_metadata_task_3_4.csv',
    response_cols=['QuestionId', 'UserId', 'AnswerId', 'IsCorrect', 'CorrectAnswer', 'AnswerValue'],
):
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir / "data" / "2020"

    data_dir = Path(data_dir)
    print(f"数据目录: {data_dir}")

    # 1. 加载 subject_metadata
    subject_path = data_dir / subject_file
    subject_name_map = load_subject_name_map(subject_path)
    print(f"已加载 {len(subject_name_map)} 个 Subject")

    # 2. 加载 question_metadata，构建 QuestionId -> SubjectId
    question_path = data_dir / question_file
    question_subject_map = build_question_subject_map(question_path, use_last=True)
    print(f"已加载 {len(question_subject_map)} 个 Question 的 Subject 映射")

    # 3. 加载 train 和 test（仅保留答题相关列）
    train_path = data_dir / train_file
    test_path = data_dir / test_file

    train_df = pd.read_csv(train_path, usecols=response_cols)
    print(f"train: {len(train_df)} 条")

    # test 文件可能包含 IsTarget_* 等列，只取前 6 列
    test_df = pd.read_csv(test_path, usecols=response_cols)
    print(f"test:  {len(test_df)} 条")

    # 4. 合并 train + test
    response_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"合并后: {len(response_df)} 条")

    # 5. 添加 SubjectId 和 Name
    merged = merge_response_data(response_df, question_subject_map, subject_name_map)

    # 6. 调整列顺序
    out_cols = ['QuestionId', 'UserId', 'AnswerId', 'IsCorrect', 'CorrectAnswer', 'AnswerValue', 'SubjectId', 'Name']
    merged = merged[out_cols]
    merged['SubjectId'] = merged['SubjectId'].astype(int)

    # 7. 保存
    out_path = data_dir / output_file
    merged.to_csv(out_path, index=False)
    print(f"已保存至: {out_path}")
    print(f"最终记录数: {len(merged)}")
    print(f"列: {list(merged.columns)}")

    return merged


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录，默认与脚本同目录')
    parser.add_argument('--output', type=str, default='merged_data.csv')
    parser.add_argument('--train', type=str, default='train_task_3_4.csv')
    parser.add_argument('--test', type=str, default='test_public_task_4_more_splits.csv')
    parser.add_argument('--subject', type=str, default='subject_metadata.csv')
    parser.add_argument('--question', type=str, default='question_metadata_task_3_4.csv')
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_file=args.output,
        train_file=args.train,
        test_file=args.test,
        subject_file=args.subject,
        question_file=args.question,
    )
