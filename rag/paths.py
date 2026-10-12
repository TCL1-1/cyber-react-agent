"""
项目路径解析。

所有路径都锚定到项目根目录，不再依赖「当前工作目录」——
知识库检索不到内容的根因，就是写入和读取用了两个不同的相对路径。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHROMA_PATH = PROJECT_ROOT / "storage" / "chroma_db"
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"

CHROMA_PATH_ENV = "CHROMA_DB_PATH"


def project_path(*parts: str) -> Path:
    """拼接项目内的绝对路径。"""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_db_path(path: str | os.PathLike[str] | None = None) -> Path:
    """
    解析向量库路径，优先级：显式参数 > 环境变量 CHROMA_DB_PATH > 项目内默认值。

    相对路径一律按项目根目录解析（而不是当前工作目录）。
    """
    raw = path or os.getenv(CHROMA_PATH_ENV) or DEFAULT_CHROMA_PATH
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    return resolved
