"""
知识库导入。

用法（在项目根目录执行）:

    .venv\\Scripts\\python.exe -m rag.ingest

特性:
- 可重复执行：每个来源导入前先删除旧切片，文档改短也不会留下陈旧内容
- 切片 id 由「来源 + 序号」决定，固定可复现
- 导入路径锚定项目根目录，不受当前工作目录影响
- 导入时才连接向量库、计算嵌入，import 本模块不会产生副作用
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Iterable, List

from rag.chroma_db import delete_by_source, get_collection, insert, resolve_db_path
from rag.embedding import MODEL_NAME
from rag.loader import load_documents
from rag.paths import DEFAULT_KNOWLEDGE_BASE
from rag.splitter import DEFAULT_CHUNK_SIZE, split_text

logger = logging.getLogger(__name__)


def build_records(
    documents: Iterable[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[Dict[str, Any]]:
    """把文档列表展开成「切片 + 元数据 + id」。"""
    records: List[Dict[str, Any]] = []
    for document in documents:
        source = document["source"]
        chunks = split_text(document["content"], chunk_size=chunk_size)
        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{source}::{index}",
                    "text": chunk,
                    "metadata": {"source": source, "chunk": index},
                }
            )
    return records


def ingest(
    base_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, Any]:
    """导入知识库，返回统计信息。"""
    root = str(base_path or DEFAULT_KNOWLEDGE_BASE)

    documents = load_documents(root)
    if not documents:
        logger.warning("在 %s 下没有找到可导入的 .md / .txt 文档", root)

    records = build_records(documents, chunk_size=chunk_size)

    # 按来源分组，先清理旧切片再写入，保证可重复执行
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(record["metadata"]["source"], []).append(record)

    removed = 0
    for source, group in by_source.items():
        removed += delete_by_source(source)
        insert(
            ids=[record["id"] for record in group],
            texts=[record["text"] for record in group],
            metadata=[record["metadata"] for record in group],
        )

    total = get_collection().count()

    return {
        "knowledge_base": root,
        "db_path": str(resolve_db_path()),
        "documents": len(documents),
        "chunks": len(records),
        "replaced": removed,
        "collection_size": total,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入本地安全知识库到 Chroma")
    parser.add_argument(
        "--base-path",
        default=None,
        help=f"知识库目录（默认 {DEFAULT_KNOWLEDGE_BASE}）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"切片长度（字符，默认 {DEFAULT_CHUNK_SIZE}）",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = ingest(args.base_path, args.chunk_size)

    print(f"嵌入模型      : {MODEL_NAME}")
    print(f"知识库目录    : {summary['knowledge_base']}")
    print(f"向量库路径    : {summary['db_path']}")
    print(f"文档数        : {summary['documents']}")
    print(f"本次写入切片  : {summary['chunks']}（替换旧切片 {summary['replaced']} 条）")
    print(f"集合现有条数  : {summary['collection_size']}")
    print("知识库导入完成")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
