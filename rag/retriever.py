"""
检索入口。

把向量库返回的原始结构整理成「文本 + 来源 + 距离」，
所有调用方（tools/rag_tool.py 等）都走这里，避免各自重复解析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from rag.chroma_db import query

DEFAULT_TOP_K = 3

UNKNOWN_SOURCE = "未知来源"


@dataclass
class RetrievedChunk:
    """一条检索结果。"""

    text: str
    source: str
    distance: float


def _pick(sequence: Any, index: int, default: Any) -> Any:
    if isinstance(sequence, (list, tuple)) and index < len(sequence):
        return sequence[index]
    return default


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
    """检索与问题最相关的前 top_k 个切片。"""
    result = query(question, n_results=top_k)

    documents = _pick(result.get("documents"), 0, []) or []
    metadatas = _pick(result.get("metadatas"), 0, []) or []
    distances = _pick(result.get("distances"), 0, []) or []

    chunks: List[RetrievedChunk] = []
    for index, text in enumerate(documents):
        metadata = _pick(metadatas, index, {}) or {}
        distance = _pick(distances, index, float("nan"))
        chunks.append(
            RetrievedChunk(
                text=text,
                source=str(metadata.get("source", UNKNOWN_SOURCE)),
                distance=float(distance),
            )
        )

    return chunks
