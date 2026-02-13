# -*- coding: utf-8 -*-
"""
根据用户需求字符串，在知识点树中逐层筛选（层级1 -> 2 -> 4/最深），最后由 LLM 设计学习路径并输出理由。
"""

import json
import re
import sys
from pathlib import Path

try:
    from .llm_client import chat
    from .subject_store import SubjectStore
except ImportError:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from our.llm_client import chat
    from our.subject_store import SubjectStore

# 需求字符串：直接修改变量即可，无需 input()
INPUT_STRING = "我想巩固初中代数与几何中的方程与图形部分"


def _parse_json_from_content(content: str) -> dict | list:
    """从 LLM 返回中解析 JSON（允许被 ```json 包裹）。"""
    if not content or not content.strip():
        raise ValueError("LLM 返回为空")
    text = content.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"无法解析 JSON: {text[:300]}")


def _validate_level1_selection(selected: list[dict], store: SubjectStore) -> list[dict]:
    """
    校验层级1的选择：名称正确但 id 错误则按名称修正 id；名称与 id 都对保留；都错则抛弃。
    """
    level1_id_by_name = {name: sid for sid, name in store.get_level1()}
    result = []
    for item in selected:
        name = (item.get("name") or item.get("Name") or "").strip()
        raw_id = item.get("id") or item.get("SubjectId")
        if name in level1_id_by_name:
            correct_id = level1_id_by_name[name]
            result.append({"id": correct_id, "name": name})
        # 名称不在层级1中则抛弃该项
    return result


def run_learning_path_flow(description: str | None = None) -> dict:
    """
    完整流程：1）选层级1 -> 校验 2）选层级2 3）选层级4（或最深）4）设计学习路径与理由。
    description 为空则使用模块变量 INPUT_STRING。
    """
    desc = description or INPUT_STRING
    store = SubjectStore()
    max_level = store.max_level()

    # ---------- Step 1: 层级1 ----------
    level1_items = store.get_level1()
    level1_text = "\n".join(f"{sid}: {name}" for sid, name in level1_items)
    prompt1 = f"""用户需求：{desc}

以下为知识点层级1（顶层）的 id 与名称列表：
{level1_text}

请从上述列表中选出与用户需求相符的项，只返回 JSON 数组，每项为 {{"id": 数字, "name": "名称"}}。不要解释。"""

    content1 = chat([{"role": "user", "content": prompt1}])
    try:
        raw1 = _parse_json_from_content(content1)
        selected1 = raw1 if isinstance(raw1, list) else raw1.get("selected", raw1.get("items", []))
    except Exception as e:
        selected1 = []
        print(f"[Step1] 解析 LLM 返回失败: {e}")
    selected1 = _validate_level1_selection(selected1, store)
    print(f"[Step1] 层级1 筛选并校验后: {selected1}")

    if not selected1:
        return {"level1": [], "level2": [], "level_max": [], "path": [], "reason": "层级1 无有效选项"}

    ids1 = [x["id"] for x in selected1]

    # ---------- Step 2: 层级2（上述 id 的子节点）----------
    children2 = []
    for pid in ids1:
        children2.extend(store.get_children(pid))
    if not children2:
        return {"level1": selected1, "level2": [], "level_max": [], "path": [], "reason": "层级2 无子节点"}

    level2_text = "\n".join(f"{n['id']}: {n['name']}" for n in children2)
    prompt2 = f"""用户需求：{desc}

以下为与需求相关的层级2知识点（id: 名称）：
{level2_text}

请从上述列表中选出与用户需求相符的项，只返回 JSON 数组，每项为 {{"id": 数字, "name": "名称"}}。不要解释。"""

    content2 = chat([{"role": "user", "content": prompt2}])
    try:
        raw2 = _parse_json_from_content(content2)
        selected2_raw = raw2 if isinstance(raw2, list) else raw2.get("selected", raw2.get("items", []))
        # 校验 id 必须在 children2 中
        id2_set = {n["id"] for n in children2}
        selected2 = [x for x in selected2_raw if (x.get("id") or x.get("SubjectId")) in id2_set]
        selected2 = [{"id": x.get("id") or x.get("SubjectId"), "name": x.get("name") or x.get("Name") or ""} for x in selected2]
    except Exception as e:
        selected2 = []
        print(f"[Step2] 解析失败: {e}")
    print(f"[Step2] 层级2 筛选后: {selected2}")

    ids2 = [x["id"] for x in selected2]
    if not ids2:
        return {"level1": selected1, "level2": [], "level_max": [], "path": [], "reason": "层级2 无有效选项"}

    # ---------- Step 3: 层级4（或最深层级）----------
    level_target = min(4, max_level)
    descendants_max = store.get_descendants_at_level(ids2, level_target)
    if not descendants_max:
        level_target = max_level
        descendants_max = store.get_descendants_at_level(ids2, level_target)

    if not descendants_max:
        return {"level1": selected1, "level2": selected2, "level_max": [], "path": [], "reason": "最深层级无节点"}

    level_max_text = "\n".join(f"{n['id']}: {n['name']}" for n in descendants_max)
    prompt3 = f"""用户需求：{desc}

以下为与需求相关的层级{level_target}（深层）知识点（id: 名称）：
{level_max_text}

请从上述列表中选出与用户需求最相关的项，只返回 JSON 数组，每项为 {{"id": 数字, "name": "名称"}}。不要解释。"""

    content3 = chat([{"role": "user", "content": prompt3}])
    try:
        raw3 = _parse_json_from_content(content3)
        selected3_raw = raw3 if isinstance(raw3, list) else raw3.get("selected", raw3.get("items", []))
        id_max_set = {n["id"] for n in descendants_max}
        selected3 = [x for x in selected3_raw if (x.get("id") or x.get("SubjectId")) in id_max_set]
        selected3 = [{"id": x.get("id") or x.get("SubjectId"), "name": x.get("name") or x.get("Name") or ""} for x in selected3]
    except Exception as e:
        selected3 = []
        print(f"[Step3] 解析失败: {e}")
    print(f"[Step3] 层级{level_target} 筛选后: {len(selected3)} 项")

    if not selected3:
        return {"level1": selected1, "level2": selected2, "level_max": [], "path": [], "reason": "最深层级无有效选项"}

    # ---------- Step 4: 设计学习路径与理由 ----------
    path_text = "\n".join(f"{n['id']}: {n['name']}" for n in selected3)
    prompt4 = f"""用户需求：{desc}

以下为已筛选出的深层知识点（id: 名称）：
{path_text}

请为这些知识点设计一个合理的学习路径（推荐的学习顺序），并输出：
1) 学习路径：按顺序列出 id 或 (id, 名称)，表示建议的学习先后。
2) 理由：简要说明为什么这样排序（例如前置依赖、由浅入深等）。

只返回一个 JSON 对象，格式如：
{{"path": [{{"id": 数字, "name": "名称"}}, ...], "reason": "理由文字"}}
或 {{"path": [id1, id2, ...], "reason": "理由文字"}}"""

    content4 = chat([{"role": "user", "content": prompt4}])
    try:
        out = _parse_json_from_content(content4)
        path = out.get("path", [])
        reason = out.get("reason", "")
    except Exception as e:
        path = []
        reason = f"解析失败: {e}"
        print(f"[Step4] {reason}")

    return {
        "level1": selected1,
        "level2": selected2,
        "level_max": selected3,
        "path": path,
        "reason": reason,
    }


if __name__ == "__main__":
    result = run_learning_path_flow()
    print("--- 最终结果 ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
