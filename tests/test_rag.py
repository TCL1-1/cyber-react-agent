"""
RAG 相关离线测试：切分逻辑、路径解析、检索结果整理、空结果提示。

不访问网络，也不读写真实向量库。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import paths, splitter
from rag.retriever import RetrievedChunk
from tools import rag_tool


class SplitterTest(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(splitter.split_text(""), [])
        self.assertEqual(splitter.split_text("   \n  "), [])

    def test_short_text_is_single_chunk(self):
        self.assertEqual(splitter.split_text("## 简介\n\nNmap 是扫描工具。"), ["## 简介\n\nNmap 是扫描工具。"])

    def test_rejects_bad_chunk_size(self):
        with self.assertRaises(ValueError):
            splitter.split_text("abc", chunk_size=0)

    def test_long_text_respects_chunk_size(self):
        paragraphs = [f"## 小节{i}\n\n" + "内容" * 40 for i in range(8)]
        chunks = splitter.split_text("\n\n".join(paragraphs), chunk_size=200, overlap=0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200, f"切片超长: {len(chunk)}")

    def test_no_content_is_lost(self):
        text = "## A\n\n" + "甲" * 50 + "\n\n## B\n\n" + "乙" * 50
        chunks = splitter.split_text(text, chunk_size=60, overlap=0)
        joined = "".join(chunks)
        for marker in ("## A", "## B", "甲" * 50, "乙" * 50):
            self.assertIn(marker, joined)

    def test_heading_stays_with_its_body(self):
        text = "## 防御方法\n\n" + "使用参数化查询。" * 20
        chunks = splitter.split_text(text, chunk_size=120, overlap=0)
        self.assertTrue(chunks[0].startswith("## 防御方法"))

    def test_overlap_is_capped(self):
        text = "句子。" * 300
        chunks = splitter.split_text(text, chunk_size=100, overlap=999)
        self.assertGreater(len(chunks), 1)


class PathsTest(unittest.TestCase):
    def test_default_path_is_absolute_and_under_project(self):
        original = os.environ.pop(paths.CHROMA_PATH_ENV, None)
        try:
            resolved = paths.resolve_db_path()
        finally:
            if original is not None:
                os.environ[paths.CHROMA_PATH_ENV] = original

        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved, paths.PROJECT_ROOT / "storage" / "chroma_db")

    def test_relative_path_is_anchored_to_project_root(self):
        resolved = paths.resolve_db_path("./some/db")
        self.assertEqual(resolved, (paths.PROJECT_ROOT / "some" / "db").resolve())

    def test_absolute_path_is_kept(self):
        target = Path("D:/tmp/whatever").resolve() if os.name == "nt" else Path("/tmp/whatever")
        self.assertEqual(paths.resolve_db_path(str(target)), target)

    def test_env_var_is_honoured(self):
        original = os.environ.get(paths.CHROMA_PATH_ENV)
        os.environ[paths.CHROMA_PATH_ENV] = "./from_env"
        try:
            self.assertEqual(
                paths.resolve_db_path(),
                (paths.PROJECT_ROOT / "from_env").resolve(),
            )
        finally:
            if original is None:
                os.environ.pop(paths.CHROMA_PATH_ENV, None)
            else:
                os.environ[paths.CHROMA_PATH_ENV] = original


class RagToolTest(unittest.TestCase):
    """rag_search 的返回值决定了模型会不会说「没检索到」。"""

    def setUp(self):
        self._retrieve = rag_tool.retrieve
        self._count = rag_tool.count

    def tearDown(self):
        rag_tool.retrieve = self._retrieve
        rag_tool.count = self._count

    def test_includes_sources(self):
        rag_tool.retrieve = lambda question, top_k=3: [
            RetrievedChunk(text="参数化查询可防御注入", source="kb/owasp.md", distance=0.9)
        ]
        result = rag_tool.rag_search("SQL注入")
        self.assertIn("知识库检索到 1 条", result)
        self.assertIn("kb/owasp.md", result)
        self.assertIn("参数化查询可防御注入", result)

    def test_empty_collection_explains_how_to_ingest(self):
        rag_tool.retrieve = lambda question, top_k=3: []
        rag_tool.count = lambda: 0
        result = rag_tool.rag_search("SQL注入")
        self.assertIn("知识库当前为空", result)
        self.assertIn("rag.ingest", result)

    def test_miss_on_non_empty_collection(self):
        rag_tool.retrieve = lambda question, top_k=3: []
        rag_tool.count = lambda: 5
        result = rag_tool.rag_search("SQL注入")
        self.assertIn("没有检索到", result)
        self.assertIn("5 条记录", result)


class RetrieverTest(unittest.TestCase):
    def test_parses_chroma_result(self):
        from rag import retriever

        original = retriever.query
        retriever.query = lambda text, n_results=3: {
            "documents": [["doc-a", "doc-b"]],
            "metadatas": [[{"source": "a.md"}, {"source": "b.md"}]],
            "distances": [[0.5, 1.5]],
        }
        try:
            chunks = retriever.retrieve("q")
        finally:
            retriever.query = original

        self.assertEqual([c.source for c in chunks], ["a.md", "b.md"])
        self.assertEqual([c.distance for c in chunks], [0.5, 1.5])

    def test_tolerates_missing_metadata(self):
        from rag import retriever

        original = retriever.query
        retriever.query = lambda text, n_results=3: {"documents": [["doc"]], "metadatas": [[]], "distances": [[]]}
        try:
            chunks = retriever.retrieve("q")
        finally:
            retriever.query = original

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "未知来源")


if __name__ == "__main__":
    unittest.main(verbosity=2)
