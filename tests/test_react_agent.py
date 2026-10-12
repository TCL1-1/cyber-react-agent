"""
离线单元测试：全部使用假客户端，不访问网络、不读写向量库。

运行:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试会故意触发告警与工具异常日志，这里静音，避免掩盖真正的失败
logging.disable(logging.ERROR)

import httpx
import openai

import tools
from agent.memory import ConversationMemory
from agent import react_agent
from agent.react_agent import (
    ModelCallError,
    parse_arguments,
    parse_response,
    run_agent,
)
from tools import ToolError, call_tool


DUMMY_CALLS = []


def _dummy_tool(query):
    DUMMY_CALLS.append(query)
    return f"dummy:{query}"


def _dummy_tool_alt(question):
    DUMMY_CALLS.append(question)
    return f"alt:{question}"


def _fake_stream(pieces):
    for piece in pieces:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))],
            usage=None,
        )


class FakeCompletions:
    """按脚本依次响应。脚本项可以是字符串、字符串列表（流式分片）或异常实例。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("假客户端脚本已耗尽：模型被调用了额外次数")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if kwargs.get("stream"):
            return _fake_stream(item if isinstance(item, list) else [item])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))],
            usage=None,
        )


class FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=FakeCompletions(script))

    @property
    def calls(self):
        return self.chat.completions.calls


class BaseAgentTest(unittest.TestCase):
    def setUp(self):
        DUMMY_CALLS.clear()
        self._orig_client = react_agent._client
        self._orig_delay = react_agent.RETRY_BASE_DELAY
        react_agent.RETRY_BASE_DELAY = 0.0
        tools.TOOLS["dummy"] = _dummy_tool
        tools.TOOLS["dummy_alt"] = _dummy_tool_alt

    def tearDown(self):
        react_agent._client = self._orig_client
        react_agent.RETRY_BASE_DELAY = self._orig_delay
        tools.TOOLS.pop("dummy", None)
        tools.TOOLS.pop("dummy_alt", None)

    def use_script(self, script):
        client = FakeClient(script)
        react_agent._client = client
        return client

    def quiet(self, prompt, memory=None, **kwargs):
        """静默模式跑一轮（不打印思考过程）。"""
        kwargs.setdefault("stream", False)
        kwargs.setdefault("verbose", False)
        if memory is None:
            memory = ConversationMemory()
        with contextlib.redirect_stdout(io.StringIO()):
            return run_agent(prompt, memory, **kwargs)

    @staticmethod
    def observations(messages):
        """取出所有 Observation 回填的内容（去掉前缀）。"""
        return [
            message["content"][len(react_agent.OBSERVATION_PREFIX) :]
            for message in messages
            if message["role"] == react_agent.OBSERVATION_ROLE
            and message["content"].startswith(react_agent.OBSERVATION_PREFIX)
        ]


# --------------------------------------------------------------------------- #
# 输出解析
# --------------------------------------------------------------------------- #

class ParseResponseTest(unittest.TestCase):
    def test_extracts_all_fields(self):
        parsed = parse_response(
            'Thought: 需要查询\nAction: dummy\nInput: {"query": "x"}'
        )
        self.assertEqual(parsed.thought, "需要查询")
        self.assertEqual(parsed.action, "dummy")
        self.assertEqual(parsed.raw_input, '{"query": "x"}')
        self.assertIsNone(parsed.answer)

    def test_answer_is_multiline(self):
        parsed = parse_response("Thought: 完成\nAnswer: 第一行\n第二行")
        self.assertTrue(parsed.has_answer)
        self.assertEqual(parsed.answer, "第一行\n第二行")

    def test_answer_inside_thought_is_not_terminal(self):
        """句中出现 Answer: 不能当成终止信号 —— 这是旧版的核心缺陷。"""
        parsed = parse_response(
            'Thought: 我稍后会给出 Answer: 42，先查知识库\n'
            'Action: dummy\nInput: {"query": "x"}'
        )
        self.assertIsNone(parsed.answer)
        self.assertEqual(parsed.action, "dummy")

    def test_bold_and_chinese_field_names(self):
        parsed = parse_response(
            '**思考**: 分析\n**行动**: dummy\n**输入**: {"query": "中文"}\n**最终答案**：完成'
        )
        self.assertEqual(parsed.action, "dummy")
        self.assertEqual(parsed.raw_input, '{"query": "中文"}')
        self.assertEqual(parsed.answer, "完成")

    def test_markdown_list_and_heading_markers(self):
        parsed = parse_response(
            '### Thought: 分析\n- Action: dummy\n1. Input: {"query": "a"}\nAnswer: 好'
        )
        self.assertEqual(parsed.action, "dummy")
        self.assertEqual(parsed.raw_input, '{"query": "a"}')
        self.assertEqual(parsed.answer, "好")

    def test_action_input_alias_maps_to_input(self):
        parsed = parse_response('Action: dummy\nAction Input: {"query": "y"}')
        self.assertEqual(parsed.raw_input, '{"query": "y"}')

    def test_action_takes_first_token_only(self):
        parsed = parse_response("Action: dummy 查询知识库\nInput: {}")
        self.assertEqual(parsed.action, "dummy")

    def test_last_answer_wins(self):
        parsed = parse_response(
            "Thought: a\nAnswer: 第一版\nThought: b\nAnswer: 第二版"
        )
        self.assertEqual(parsed.answer, "第二版")

    def test_plain_prose_has_no_fields(self):
        parsed = parse_response("我直接回答，没有字段")
        self.assertEqual(parsed.fields_found, [])
        self.assertIsNone(parsed.action)
        self.assertIsNone(parsed.answer)

    def test_deprecated_helpers_still_work(self):
        msg = 'Thought: t\nAction: dummy\nInput: {"query": "z"}'
        self.assertEqual(react_agent.parse_action(msg), "dummy")
        self.assertEqual(react_agent.parse_input(msg), '{"query": "z"}')


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #

class ParseArgumentsTest(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_arguments('{"query": "x"}'), {"query": "x"})

    def test_fenced_json(self):
        raw = '```json\n{"query": "x"}\n```'
        self.assertEqual(parse_arguments(raw), {"query": "x"})

    def test_single_quotes(self):
        self.assertEqual(parse_arguments("{'query': 'x'}"), {"query": "x"})

    def test_trailing_comma(self):
        self.assertEqual(parse_arguments('{"query": "x",}'), {"query": "x"})

    def test_json_embedded_in_prose(self):
        raw = '查询参数如下 {"query": "x"} 请执行'
        self.assertEqual(parse_arguments(raw), {"query": "x"})

    def test_multiline_json(self):
        raw = '{\n  "query": "x",\n  "n": 3\n}'
        self.assertEqual(parse_arguments(raw), {"query": "x", "n": 3})

    def test_empty_input(self):
        self.assertEqual(parse_arguments(""), {})

    def test_non_json_falls_back_to_single_value(self):
        self.assertEqual(parse_arguments("knowledge_base/cve"), {"input": "knowledge_base/cve"})

    def test_bare_scalar(self):
        self.assertEqual(parse_arguments("3"), {"input": 3})


# --------------------------------------------------------------------------- #
# 工具调用
# --------------------------------------------------------------------------- #

class CallToolTest(unittest.TestCase):
    def setUp(self):
        DUMMY_CALLS.clear()
        tools.TOOLS["dummy"] = _dummy_tool
        tools.TOOLS["dummy_alt"] = _dummy_tool_alt

    def tearDown(self):
        tools.TOOLS.pop("dummy", None)
        tools.TOOLS.pop("dummy_alt", None)

    def test_binds_by_parameter_name(self):
        self.assertEqual(call_tool("dummy", {"query": "x"}), "dummy:x")

    def test_single_parameter_falls_back_by_position(self):
        """模型写错参数名时，单参数工具应自动兜底绑定。"""
        self.assertEqual(call_tool("dummy_alt", {"query": "x"}), "alt:x")

    def test_unknown_tool_lists_available(self):
        with self.assertRaises(ToolError) as ctx:
            call_tool("nope", {})
        self.assertIn("不存在工具", str(ctx.exception))
        self.assertIn("dummy", str(ctx.exception))

    def test_missing_required_parameter(self):
        with self.assertRaises(ToolError) as ctx:
            call_tool("dummy", {})
        self.assertIn("缺少必需参数", str(ctx.exception))

    def test_unknown_parameter_reports_expectation(self):
        tools.TOOLS["two_args"] = lambda query, limit: f"{query}/{limit}"
        try:
            with self.assertRaises(ToolError) as ctx:
                call_tool("two_args", {"bad": 1})
            self.assertIn("无法识别参数", str(ctx.exception))
            self.assertIn("limit", str(ctx.exception))
        finally:
            tools.TOOLS.pop("two_args", None)

    def test_non_dict_arguments(self):
        with self.assertRaises(ToolError):
            call_tool("dummy", "not-a-dict")


# --------------------------------------------------------------------------- #
# 循环行为
# --------------------------------------------------------------------------- #

class RunAgentTest(BaseAgentTest):
    def test_tool_then_answer(self):
        client = self.use_script(
            [
                'Thought: 需要查询\nAction: dummy\nInput: {"query": "abc"}',
                "Thought: 够了\nAnswer: 最终结论",
            ]
        )
        result = self.quiet("问题")

        self.assertEqual(result, "最终结论")
        self.assertEqual(DUMMY_CALLS, ["abc"])
        self.assertEqual(len(client.calls), 2)

        # 第二次调用必须带上 Observation（且不能用 tool 角色，见下）
        second = client.calls[1]["messages"]
        roles = [m["role"] for m in second]
        self.assertNotIn(
            "tool",
            roles,
            "DeepSeek 会以 422 拒绝没有 tool_call_id 的 tool 消息",
        )
        self.assertEqual(self.observations(second), ["dummy:abc"])

        # stop 序列已下发，防止模型自己编造 Observation
        self.assertEqual(client.calls[0]["stop"], ["Observation:", "观察:"])
        self.assertEqual(client.calls[0]["model"], react_agent.MODEL)
        self.assertEqual(client.calls[0]["messages"][0]["role"], "system")

    def test_max_steps_triggers_finalize(self):
        client = self.use_script(
            [
                'Action: dummy\nInput: {"query": "a"}',
                'Action: dummy\nInput: {"query": "b"}',
                "Answer: 预算用尽",
            ]
        )
        result = self.quiet("问题", max_steps=2)

        self.assertEqual(result, "预算用尽")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(DUMMY_CALLS, ["a", "b"])

        last_messages = client.calls[-1]["messages"]
        self.assertTrue(
            any("预算已经用尽" in m["content"] for m in last_messages),
            "收尾调用应带有收尾指令",
        )

    def test_max_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.quiet("问题", max_steps=0)

    def test_retry_then_success(self):
        request = httpx.Request("POST", "https://api.deepseek.com")
        client = self.use_script(
            [openai.APIConnectionError(request=request), "Answer: 重试成功"]
        )
        result = self.quiet("问题")

        self.assertEqual(result, "重试成功")
        self.assertEqual(len(client.calls), 2)

    def test_retry_exhausted_returns_graceful_message(self):
        request = httpx.Request("POST", "https://api.deepseek.com")
        client = self.use_script(
            [openai.APIConnectionError(request=request) for _ in range(3)]
        )
        result = self.quiet("问题")

        self.assertTrue(result.startswith("[模型调用失败]"), result)
        self.assertEqual(len(client.calls), 3)

    def test_unknown_tool_error_is_fed_back(self):
        client = self.use_script(['Action: nope\nInput: {}', "Answer: 已修正"])
        result = self.quiet("问题")

        self.assertEqual(result, "已修正")
        observations = self.observations(client.calls[1]["messages"])
        self.assertTrue(observations)
        self.assertIn("不存在工具", observations[0])
        self.assertIn("可用工具", observations[0])

    def test_missing_parameter_is_fed_back(self):
        client = self.use_script(["Action: read_file\nInput: {}", "Answer: 补上参数"])
        result = self.quiet("问题")

        self.assertEqual(result, "补上参数")
        observations = self.observations(client.calls[1]["messages"])
        self.assertIn("缺少必需参数", observations[0])
        self.assertIn("path", observations[0])

    def test_tool_exception_becomes_observation(self):
        def boom(query):
            raise RuntimeError("工具内部爆炸")

        tools.TOOLS["boom"] = boom
        try:
            client = self.use_script(["Action: boom\nInput: {\"query\": \"x\"}", "Answer: 兜住了"])
            result = self.quiet("问题")
        finally:
            tools.TOOLS.pop("boom", None)

        self.assertEqual(result, "兜住了")
        observations = self.observations(client.calls[1]["messages"])
        self.assertIn("工具执行失败", observations[0])
        self.assertIn("工具内部爆炸", observations[0])

    def test_plain_prose_is_returned_as_answer(self):
        client = self.use_script(["我直接回答，没有用任何字段"])
        result = self.quiet("问题")

        self.assertEqual(result, "我直接回答，没有用任何字段")
        self.assertEqual(len(client.calls), 1)

    def test_format_error_nudges_then_gives_up(self):
        client = self.use_script(
            ["Thought: 我想想", "Thought: 再想想", "Thought: 还是想"]
        )
        result = self.quiet("问题")

        self.assertEqual(result, "Thought: 还是想")
        self.assertEqual(len(client.calls), 3)
        hints = [
            m["content"]
            for m in client.calls[-1]["messages"]
            if "不符合 ReAct 格式" in m["content"]
        ]
        self.assertTrue(hints)

    def test_empty_response_then_recovers(self):
        client = self.use_script(["", "Answer: 恢复了"])
        result = self.quiet("问题")

        self.assertEqual(result, "恢复了")
        self.assertEqual(len(client.calls), 2)

    def test_repeated_action_triggers_finalize(self):
        repeated = 'Action: dummy\nInput: {"query": "same"}'
        client = self.use_script([repeated] * 5 + ["Answer: 停止重复"])
        result = self.quiet("问题")

        self.assertEqual(result, "停止重复")
        self.assertEqual(len(client.calls), 6)
        # 重复的调用不再真正执行工具
        self.assertEqual(DUMMY_CALLS, ["same", "same"])

    def test_observation_is_truncated(self):
        tools.TOOLS["huge"] = lambda query: "x" * 10000
        try:
            client = self.use_script(
                ['Action: huge\nInput: {"query": "q"}', "Answer: 好了"]
            )
            self.quiet("问题")
        finally:
            tools.TOOLS.pop("huge", None)

        observations = self.observations(client.calls[1]["messages"])
        self.assertLessEqual(len(observations[0]), react_agent.MAX_OBSERVATION_CHARS + 100)
        self.assertIn("已截断", observations[0])

    def test_quiet_mode_prints_nothing(self):
        self.use_script(['Action: dummy\nInput: {"query": "1"}', "Answer: 完成"])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_agent("问题", ConversationMemory(), stream=False, verbose=False)
        self.assertEqual(buffer.getvalue(), "")


# --------------------------------------------------------------------------- #
# 流式输出
# --------------------------------------------------------------------------- #

class StreamingTest(BaseAgentTest):
    def test_tokens_are_forwarded_in_order(self):
        client = self.use_script([["Thought: 想", "\nAnswer: ", "流式完成"]])
        pieces = []

        result = run_agent(
            "问题",
            ConversationMemory(),
            stream=True,
            verbose=False,
            on_token=pieces.append,
        )

        self.assertEqual("".join(pieces), "Thought: 想\nAnswer: 流式完成")
        self.assertEqual(result, "流式完成")
        self.assertTrue(client.calls[0]["stream"])

    def test_verbose_streaming_prints_trace_and_answer(self):
        self.use_script([["Thought: 想", "\nAnswer: 打印出来的答案"]])
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            result = run_agent("问题", ConversationMemory(), stream=True, verbose=True)

        output = buffer.getvalue()
        self.assertIn("--- Agent 思考 ---", output)
        self.assertIn("打印出来的答案", output)
        self.assertEqual(result, "打印出来的答案")

    def test_non_stream_mode_does_not_set_stream_flag(self):
        client = self.use_script(["Answer: 非流式"])
        self.quiet("问题")

        self.assertNotIn("stream", client.calls[0])


# --------------------------------------------------------------------------- #
# 重试判定
# --------------------------------------------------------------------------- #

class MemoryTest(unittest.TestCase):
    def test_trim_never_leaves_leading_tool_message(self):
        memory = ConversationMemory(max_messages=3)
        for role in ("user", "assistant", "tool", "assistant", "tool", "assistant"):
            memory.add_message(role, role)

        roles = [m["role"] for m in memory.get_messages()]
        self.assertNotEqual(roles[0], "tool", f"窗口不应以 tool 开头: {roles}")
        self.assertLessEqual(len(roles), 3)

    def test_trim_always_keeps_at_least_one_message(self):
        memory = ConversationMemory(max_messages=1)
        memory.add_message("user", "1")
        memory.add_message("assistant", "2")
        self.assertTrue(memory.get_messages())

    def test_get_messages_returns_copy(self):
        memory = ConversationMemory()
        memory.add_message("user", "hi")

        snapshot = memory.get_messages()
        snapshot.append({"role": "user", "content": "外部改动"})

        self.assertEqual(len(memory.get_messages()), 1)

    def test_max_messages_must_be_positive(self):
        with self.assertRaises(ValueError):
            ConversationMemory(max_messages=0)


class RetryClassificationTest(unittest.TestCase):
    def test_network_error_is_retryable(self):
        request = httpx.Request("POST", "https://api.deepseek.com")
        self.assertTrue(
            react_agent._is_retryable(openai.APIConnectionError(request=request))
        )
        self.assertTrue(
            react_agent._is_retryable(openai.APITimeoutError(request=request))
        )

    def test_arbitrary_exception_is_not_retryable(self):
        self.assertFalse(react_agent._is_retryable(ValueError("boom")))

    def test_model_call_error_is_agent_error(self):
        self.assertTrue(issubclass(ModelCallError, react_agent.AgentError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
