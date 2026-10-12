"""
rag_search 工具：检索本地安全知识库。

返回内容带来源标注；检索不到时给出可执行的说明，
而不是把空字符串丢回给模型（那会让它只能回一句「知识库未检索到相关内容」）。
"""

from __future__ import annotations

from rag.chroma_db import count
from rag.retriever import DEFAULT_TOP_K, retrieve


def rag_search(question):
    """查询本地安全知识库（CVE / OWASP / Kali / 渗透测试 / 安全论文）。"""
    chunks = retrieve(question, top_k=DEFAULT_TOP_K)

    if not chunks:
        total = count()
        if total == 0:
            return (
                "知识库当前为空（0 条记录），尚未导入任何文档。"
                "请先执行 `python -m rag.ingest` 导入 knowledge_base 目录；"
                "在此之前请如实告诉用户知识库还没有建立，不要假装检索过。"
            )
        return (
            f"没有检索到与「{question}」相关的内容"
            f"（知识库现有 {total} 条记录）。"
            "可以换一个更具体的问法，或如实说明知识库中未找到。"
        )

    blocks = [
        f"[{index}] 来源: {chunk.source}\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    header = f"知识库检索到 {len(chunks)} 条相关内容（按相关度排序）："
    return header + "\n\n" + "\n\n".join(blocks)
