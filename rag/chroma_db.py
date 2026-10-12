"""
Chroma 向量库连接。

两个关键约定:

1. 路径锚定项目根目录（见 rag/paths.py），不依赖当前工作目录。
   「知识库检索无结果」的根因就是：写入落在 storage/chroma_db，
   读取却落在 ./chroma_db，两个库一个 5 条、一个 0 条。

2. 客户端、集合、嵌入函数都是惰性创建，模块导入不产生副作用。
   入库和查询共用同一个嵌入函数，杜绝两条嵌入路径不一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb

from rag.embedding import get_embedding_function
from rag.paths import DEFAULT_CHROMA_PATH, resolve_db_path

COLLECTION_NAME = "security_kb"

_client: Optional[Any] = None
_collection: Optional[Any] = None


def get_client(path: Optional[str] = None):
    """获取（并缓存）Chroma 持久化客户端。"""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(resolve_db_path(path)))
    return _client


def get_collection(name: str = COLLECTION_NAME, path: Optional[str] = None):
    """获取（并缓存）知识库集合。"""
    global _collection
    if _collection is None:
        _collection = get_client(path).get_or_create_collection(
            name=name,
            embedding_function=get_embedding_function(),
        )
    return _collection


def reset_cache() -> None:
    """丢弃缓存（切换数据库路径后需要调用，主要供测试和脚本使用）。"""
    global _client, _collection
    _client = None
    _collection = None


def insert(
    ids: Sequence[str],
    texts: Sequence[str],
    metadata: Sequence[Dict[str, Any]],
) -> None:
    """
    写入文档。

    这里刻意不传 embeddings：由集合自己的嵌入函数计算，
    保证与查询阶段使用完全相同的模型与权重。
    """
    get_collection().add(
        ids=list(ids),
        documents=list(texts),
        metadatas=list(metadata),
    )


def query(text: str, n_results: int = 3) -> Dict[str, Any]:
    """按文本检索，返回 Chroma 原始结果结构。"""
    return get_collection().query(
        query_texts=[text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def count() -> int:
    """集合内文档数量。"""
    return get_collection().count()


def delete_by_source(source: str) -> int:
    """
    删除某个来源的全部切片，返回删除条数。

    供 rag.ingest 做幂等导入：重新导入前先清掉该文件的旧切片，
    避免文档变短后留下永远检索得到的陈旧内容。
    """
    collection = get_collection()
    existing = collection.get(where={"source": source}, include=[])
    ids = list(existing.get("ids") or [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def describe() -> str:
    """一行描述当前使用的库，便于排障。"""
    path: Path = resolve_db_path()
    return f"{path}（collection={COLLECTION_NAME}, count={count()}）"
