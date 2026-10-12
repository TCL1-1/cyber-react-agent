"""
嵌入模型。

统一使用 Chroma 自带的本地 ONNX 模型（all-MiniLM-L6-v2）。

为什么不再用 sentence-transformers:
    原实现「入库用 sentence-transformers 算向量、查询交给 Chroma 的默认嵌入函数」，
    两条路径只要有一点差异，检索结果就不可信；而且 sentence-transformers 会拉起
    torch 与 huggingface_hub，多一层与检索无关的失败面。
    统一到 Chroma 自己的嵌入函数后，入库和查询必然使用同一个模型、同一份权重。
    首次使用会把 ONNX 模型缓存到 ~/.cache/chroma/onnx_models，此后离线可用。
"""

from __future__ import annotations

from typing import Any

from chromadb.utils import embedding_functions

# Chroma 默认嵌入函数使用的模型名，仅作说明
MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_function: Any = None


def get_embedding_function():
    """返回（并缓存）进程内共用的嵌入函数。"""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_function
