# -*- coding: utf-8 -*-
"""
LLM 访问脚本：OpenAI 风格 chat completions，API 等参数通过 .env 配置。
用法参考 data/data/extract_questions_deepseek.py。
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# 优先加载 our/.env，其次上级 .env
_OUR_DIR = Path(__file__).resolve().parent
for _env in (_OUR_DIR / ".env", _OUR_DIR.parent / ".env"):
    if _env.is_file():
        load_dotenv(_env)
        break

# ===================== 从环境变量读取 =====================
CHAT_COMPLETIONS_URL = os.getenv("LLM_CHAT_URL", "*")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
MODEL_NAME = os.getenv("LLM_MODEL", "Qwen2.5-72B-Instruct-GPTQ-Int4")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
TOP_P = float(os.getenv("LLM_TOP_P", "1"))
FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0"))
PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0"))


def build_headers() -> dict:
    headers = {"accept": "*/*", "Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    return headers


def chat(messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> str:
    """
    调用 OpenAI 风格 chat completions，返回 assistant 的 content 文本。
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    """
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": messages,
        "temperature": temperature if temperature is not None else TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        "top_p": TOP_P,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
    }
    resp = requests.post(
        CHAT_COMPLETIONS_URL,
        headers=build_headers(),
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
