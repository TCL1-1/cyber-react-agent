"""
文本切分。

按 markdown 结构切分（标题、空行分段），再把小段合并到目标长度，
相邻块之间保留少量重叠；只有单个段落本身就超长时才硬切。
原实现是无条件按 500 字符硬切，会把一句话拦腰截断，
而且完全无视标题带来的语义边界。
"""

from __future__ import annotations

from typing import List

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 80

_HEADING_PREFIXES = ("#", "##", "###", "####", "#####", "######")


def _split_blocks(text: str) -> List[str]:
    """先按空行分段；标题单独成段，作为语义边界。"""
    blocks: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            blocks.append("\n".join(buffer).strip())
            buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith(_HEADING_PREFIXES):
            flush()
            blocks.append(stripped)
            continue
        buffer.append(line.rstrip())
    flush()

    return [block for block in blocks if block]


def _hard_split(text: str, chunk_size: int) -> List[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """把一段文本切成若干切片。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须 > 0")

    overlap = max(0, min(overlap, chunk_size // 2))
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    current = ""

    for block in _split_blocks(text):
        # 单段就超长：先冲洗缓冲区，再硬切，最后一片留作下一轮的起点
        if len(block) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            pieces = _hard_split(block, chunk_size)
            chunks.extend(pieces[:-1])
            current = pieces[-1]
            continue

        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{block}" if tail else block

    if current:
        chunks.append(current)

    return chunks
