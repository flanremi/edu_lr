# -*- coding: utf-8 -*-
"""
知识点（Subject）存储与检索：读取 subject_metadata.csv，以 dict/树形结构存储层级关系。
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd


# 默认 metadata 路径（相对本包所在目录回退到 data/data/metadata）
_OUR_DIR = Path(__file__).resolve().parent
_DEFAULT_CSV = _OUR_DIR.parent / "data" / "data" / "metadata" / "subject_metadata.csv"


class SubjectStore:
    """
    读取 subject_metadata.csv，以 dict 方式存储知识点关系。
    提供按 id 检索、按层级检索、获取子节点等方法。
    """

    def __init__(self, csv_path: Optional[os.PathLike] = None):
        csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV
        if not csv_path.is_file():
            raise FileNotFoundError(f"知识点元数据文件不存在: {csv_path}")
        self._csv_path = csv_path
        self._df = pd.read_csv(csv_path)
        # 列名规范
        self._df = self._df.rename(columns={
            "SubjectId": "SubjectId",
            "Name": "Name",
            "ParentId": "ParentId",
            "Level": "Level",
        })
        self._df["ParentId"] = self._df["ParentId"].replace("NULL", None)
        self._df["ParentId"] = pd.to_numeric(self._df["ParentId"], errors="coerce")
        self._by_id: dict[int, dict] = {}
        self._by_level: dict[int, list[dict]] = {}
        self._children: dict[int, list[dict]] = {}
        self._build()

    def _build(self) -> None:
        """构建 id -> 节点、level -> 列表、父 -> 子列表。"""
        for _, row in self._df.iterrows():
            sid = int(row["SubjectId"])
            level = int(row["Level"]) if pd.notna(row["Level"]) else 0
            parent = None
            if pd.notna(row["ParentId"]):
                try:
                    parent = int(row["ParentId"])
                except (ValueError, TypeError):
                    parent = None
            node = {
                "id": sid,
                "name": str(row["Name"]).strip(),
                "parent_id": parent,
                "level": level,
            }
            self._by_id[sid] = node
            self._by_level.setdefault(level, []).append(node)
            if parent is not None:
                self._children.setdefault(parent, []).append(node)

    def get_by_id(self, subject_id: int) -> Optional[dict]:
        """根据 id 检索节点，返回 {id, name, parent_id, level}，不存在返回 None。"""
        return self._by_id.get(int(subject_id))

    def get_children(self, subject_id: int) -> list[dict]:
        """返回指定 id 的直接子节点列表。"""
        return self._children.get(int(subject_id), [])

    def get_by_level(self, level: int) -> list[dict]:
        """返回指定层级（Level 字段）的所有节点。"""
        return self._by_level.get(int(level), [])

    def get_level1(self) -> list[tuple[int, str]]:
        """返回层级为 1 的 (id, 名称) 列表，即 dict 的第二层。"""
        nodes = self.get_by_level(1)
        return [(n["id"], n["name"]) for n in nodes]

    def get_descendants_at_level(self, parent_ids: list[int], level: int) -> list[dict]:
        """返回 parent_ids 下所有层级为 level 的后代节点。"""
        descendant_ids = set()
        for pid in parent_ids:
            descendant_ids.add(pid)
            descendant_ids |= self.get_all_descendant_ids(pid)
        return [n for n in self.get_by_level(level) if n["id"] in descendant_ids]

    def get_all_descendant_ids(self, parent_id: int) -> set[int]:
        """返回某节点下所有后代 id（含直接、间接）。"""
        out = set()
        for child in self.get_children(parent_id):
            cid = child["id"]
            out.add(cid)
            out |= self.get_all_descendant_ids(cid)
        return out

    def max_level(self) -> int:
        """返回元数据中的最大层级。"""
        return max(self._by_level.keys(), default=0)
