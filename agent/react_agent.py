"""
ReAct Agent 主循环。

本模块职责:
1. 调用大模型（支持流式输出）
2. 解析 Thought / Action / Input / Answer 字段
3. 调度工具执行，并把 Observation 写回记忆
4. 步数上限、终止判定、参数解析、异常处理与自动重试

与旧版的接口差异:
- run_agent 返回「提取出的最终答案」，不再是原始整段文本
- 工具参数不再以原始字符串下发，而是解析为 dict 后按参数名绑定
- 客户端改为惰性创建，模块导入不再产生副作用
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from agent.memory import ConversationMemory
from agent.prompt import SYSTEM_PROMPT
from tools import ToolError, call_tool

logger = logging.getLogger(__name__)

TokenCallback = Callable[[str], None]

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# ---- 循环控制 ----
DEFAULT_MAX_STEPS = 8  # 单轮对话最多调用模型的次数（步数上限）
MAX_NUDGES = 2  # 允许模型格式出错后纠正的次数
MAX_REPEAT_ACTIONS = 2  # 同一工具 + 同一参数允许重复的次数
MAX_OBSERVATION_CHARS = 4000  # 单条 Observation 写入上下文的长度上限

# ---- 模型调用 ----
DEFAULT_TIMEOUT = 60.0  # 单次请求超时（秒）
MAX_RETRIES = 3  # 可重试错误的最大尝试次数
RETRY_BASE_DELAY = 1.0  # 退避基数（秒）
RETRY_MAX_DELAY = 8.0  # 退避上限（秒）

# 让模型在「该由系统给出 Observation」的位置停下，避免它自己编造工具结果
STOP_SEQUENCES = ["Observation:", "观察:"]

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# 可重试的异常类型（APIStatusError 另行按状态码判断）
_RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class AgentError(Exception):
    """Agent 运行期错误基类。"""


class ModelCallError(AgentError):
    """模型调用最终失败（已耗尽重试）。"""


# --------------------------------------------------------------------------- #
# 模型客户端（惰性创建，便于测试注入）
# --------------------------------------------------------------------------- #

_client: Optional[OpenAI] = None


def _normalize_no_proxy() -> List[str]:
    """
    把 NO_PROXY 中 httpx 解析不了的方括号 IPv6 写法改成裸写形式。

    背景: 部分代理客户端会把 IPv6 回环写成 `[::1]`。httpx 不把这种写法当作
    IPv6，于是按域名拼出 `all://*[::1]` 这样的挂载键，在构造 Client 时直接抛
    InvalidURL —— 结果是 Agent 连启动都做不到。裸写的 `::1` 语义等价且能被
    正确识别，所以这里做一次改写（只动这一种写法）。

    如果本机代理配置已经正常，这个函数不会有任何副作用。
    """
    normalized: List[str] = []

    for name in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(name)
        if not raw:
            continue

        entries: List[str] = []
        changed = False
        for chunk in raw.split(","):
            item = chunk.strip()
            if not item:
                continue
            if len(item) > 2 and item.startswith("[") and item.endswith("]"):
                entries.append(item[1:-1])
                normalized.append(item)
                changed = True
            else:
                entries.append(item)

        if changed:
            os.environ[name] = ",".join(entries)

    return normalized


def get_client() -> OpenAI:
    """返回全局 OpenAI 客户端，首次调用时才创建。"""
    global _client
    if _client is None:
        normalized = _normalize_no_proxy()
        if normalized:
            logger.warning(
                "NO_PROXY 中的 %s 会让 httpx 无法构造客户端，已改写为裸 IPv6 形式",
                normalized,
            )

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ModelCallError(
                "未配置 DEEPSEEK_API_KEY，请在项目根目录的 .env 中设置后重试。"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=DEFAULT_TIMEOUT,
            max_retries=0,  # 重试由本模块自己控制，便于日志与退避
        )
    return _client


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试（网络抖动 / 限流 / 服务端 5xx）。"""
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) in _RETRYABLE_STATUS
    return False


# --------------------------------------------------------------------------- #
# 输出解析
# --------------------------------------------------------------------------- #

_FIELD_ALIASES: Dict[str, str] = {
    "thought": "thought",
    "思考": "thought",
    "推理": "thought",
    "action input": "input",
    "action": "action",
    "行动": "action",
    "input": "input",
    "输入": "input",
    "参数": "input",
    "observation": "observation",
    "观察": "observation",
    "final answer": "answer",
    "answer": "answer",
    "最终答案": "answer",
    "答案": "answer",
    "结论": "answer",
}


def _alias_pattern(alias: str) -> str:
    """把 'action input' 编译成 'action[ \\t]+input'，容忍中间多个空格。"""
    return r"[ \t]+".join(re.escape(word) for word in alias.split())


_FIELD_PATTERN = "|".join(
    sorted((_alias_pattern(alias) for alias in _FIELD_ALIASES), key=len, reverse=True)
)

# 字段名必须出现在行首（允许 markdown 的列表符号 / 标题符号 / 加粗），
# 这样 "Thought: 我稍后会给出 Answer: xxx" 不会被误判为终止。
_FIELD_RE = re.compile(
    r"^[ \t]*"
    r"(?:\d+[.、)][ \t]*)?"
    r"(?:[-*+][ \t]+)?"
    r"(?:#{1,6}[ \t]*)?"
    r"(?:\*\*|__)?[ \t]*"
    r"(" + _FIELD_PATTERN + r")"
    r"[ \t]*(?:\*\*|__)?[ \t]*[:：][ \t]*(.*)$",
    re.IGNORECASE,
)

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?|```")


@dataclass
class ParsedResponse:
    """一次模型回复的结构化结果。"""

    thought: str = ""
    action: Optional[str] = None
    raw_input: str = ""
    answer: Optional[str] = None
    fields_found: List[str] = field(default_factory=list)
    preamble: str = ""

    @property
    def has_answer(self) -> bool:
        return bool((self.answer or "").strip())


def clean_json(text: str) -> str:
    """去掉模型可能包裹的 markdown 代码块围栏。"""
    if not text:
        return ""
    return _FENCE_RE.sub("", text).strip()


def _normalize_field(name: str) -> str:
    key = name.strip().lower()
    return _FIELD_ALIASES.get(key, key)


def parse_response(msg: str) -> ParsedResponse:
    """
    把模型回复解析成 Thought / Action / Input / Answer。

    解析策略:
    - 只有「行首」出现的字段名才算字段，句中出现的不算
    - Action 取第一个非空块，Input 取第一个块（与 Action 配对）
    - Answer 取最后一个非空块（模型可能有多次自我修正）
    """
    result = ParsedResponse()
    if not msg:
        return result

    blocks: List[Tuple[str, List[str]]] = []
    preamble: List[str] = []
    current: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current, current_lines
        if current is not None:
            blocks.append((current, current_lines))
        current, current_lines = None, []

    for line in msg.splitlines():
        match = _FIELD_RE.match(line)
        if match:
            flush()
            current = _normalize_field(match.group(1))
            head = match.group(2).strip()
            current_lines = [head] if head else []
            continue
        if current is None:
            preamble.append(line)
        else:
            current_lines.append(line)
    flush()

    result.preamble = "\n".join(preamble).strip()
    for name, _ in blocks:
        if name not in result.fields_found:
            result.fields_found.append(name)

    thoughts = [lines for name, lines in blocks if name == "thought"]
    if thoughts:
        result.thought = "\n".join(
            text for text in ("\n".join(lines).strip() for lines in thoughts) if text
        )

    for name, lines in blocks:
        if name != "action":
            continue
        candidates = [line.strip() for line in lines if line.strip()]
        if not candidates:
            continue
        # 只取第一个 token，避免 "rag_search 查询知识库" 这类粘连内容
        token = re.split(r"[\s(（,，。;；]", candidates[0], maxsplit=1)[0]
        token = token.strip().strip("`'\"*").strip()
        if token:
            result.action = token
            break

    for name, lines in blocks:
        if name == "input":
            result.raw_input = "\n".join(lines).strip()
            break

    answers = [
        "\n".join(lines).strip() for name, lines in blocks if name == "answer"
    ]
    answers = [text for text in answers if text]
    if answers:
        result.answer = answers[-1]

    return result


def parse_action(msg: str) -> Optional[str]:
    """[已弃用] 请改用 parse_response(msg).action。"""
    return parse_response(msg).action


def parse_input(msg: str) -> str:
    """[已弃用] 请改用 parse_response(msg).raw_input。"""
    return parse_response(msg).raw_input


# --------------------------------------------------------------------------- #
# 参数解析（健壮版）
# --------------------------------------------------------------------------- #

_TRAILING_COMMA_RE = re.compile(r",[ \t]*(\r?\n[ \t]*)?([}\]])")


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """从文本中截取第一个括号配平的片段（忽略字符串内部的括号）。"""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _argument_candidates(text: str) -> List[str]:
    """给出若干候选写法，按可信度从高到低尝试。"""
    candidates: List[str] = []

    def add(value: Optional[str]) -> None:
        if value:
            stripped = value.strip()
            if stripped and stripped not in candidates:
                candidates.append(stripped)

    add(text)
    add(_TRAILING_COMMA_RE.sub(r"\2", text))
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        fragment = _extract_balanced(text, open_ch, close_ch)
        add(fragment)
        if fragment:
            add(_TRAILING_COMMA_RE.sub(r"\2", fragment))
    return candidates


def parse_arguments(raw: str) -> Dict[str, Any]:
    """
    把 Input 字段解析成参数字典。

    依次尝试: 标准 JSON -> 去尾逗号 -> 括号截取 -> Python 字面量 -> 单值兜底。
    任何情况下都返回 dict，不抛异常；参数是否合法交给 call_tool 校验。
    """
    text = clean_json(raw or "")
    if not text:
        return {}

    for candidate in _argument_candidates(text):
        parsed: Any
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, (list, str, int, float, bool)):
            # 模型有时只给一个裸值，交给 call_tool 的单参数兜底绑定
            return {"input": parsed}
        continue

    # 完全不是结构化数据（例如 read_file 的 Input 只写了裸路径），
    # 同样交给 call_tool 的单参数兜底绑定
    return {"input": text}


# --------------------------------------------------------------------------- #
# 模型调用（流式 + 重试）
# --------------------------------------------------------------------------- #

def _call_model(
    messages: List[Dict[str, str]],
    *,
    stream: bool,
    on_token: Optional[TokenCallback],
) -> Tuple[str, Any]:
    """执行一次模型调用，返回 (文本, usage)。异常交给上层处理。"""
    client = get_client()
    kwargs: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "stop": list(STOP_SEQUENCES),
    }

    if not stream:
        response = client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return "", getattr(response, "usage", None)
        content = getattr(choices[0].message, "content", None) or ""
        return content, getattr(response, "usage", None)

    parts: List[str] = []
    usage: Any = None
    for chunk in client.chat.completions.create(stream=True, **kwargs):
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        piece = getattr(delta, "content", None) if delta is not None else None
        if piece:
            parts.append(piece)
            if on_token is not None:
                on_token(piece)
    return "".join(parts), usage


def complete_with_retry(
    messages: List[Dict[str, str]],
    *,
    stream: bool,
    on_token: Optional[TokenCallback] = None,
    max_retries: int = MAX_RETRIES,
    notice: Optional[TokenCallback] = None,
) -> Tuple[str, Any]:
    """
    带指数退避的模型调用。

    - 网络错误 / 超时 / 限流 / 5xx: 重试
    - 4xx（鉴权、参数错误等）: 立即失败，不做无谓重试
    - 最终失败抛 ModelCallError

    notice: 可选的「重试提示」回调，由调用方决定是否展示给用户。
    """
    attempts = max(1, max_retries)
    last_error: Optional[BaseException] = None

    for attempt in range(1, attempts + 1):
        try:
            return _call_model(messages, stream=stream, on_token=on_token)
        except OpenAIError as exc:
            if not _is_retryable(exc):
                logger.error("模型调用失败（不可重试）: %s", exc)
                raise ModelCallError(f"模型调用失败: {exc}") from exc

            last_error = exc
            if attempt >= attempts:
                break
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            logger.warning(
                "模型调用异常（第 %s/%s 次）: %s，%.1fs 后重试",
                attempt,
                attempts,
                exc,
                delay,
            )
            if notice is not None:
                notice(
                    f"[网络异常 {type(exc).__name__}，{delay:.1f}s 后重试 "
                    f"{attempt}/{attempts - 1}]"
                )
            time.sleep(delay)
        except Exception as exc:  # 兜底：不让任何异常穿透到主循环之外
            logger.exception("模型调用出现未预期异常")
            raise ModelCallError(f"模型调用出现未预期异常: {exc}") from exc

    raise ModelCallError(
        f"模型调用连续失败 {attempts} 次: {last_error}"
    ) from last_error


# --------------------------------------------------------------------------- #
# 工具执行
# --------------------------------------------------------------------------- #

def _stringify(value: Any, *, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """把工具返回值转成写入上下文的字符串，并做长度保护。"""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    if len(text) > limit:
        text = text[:limit] + f"\n…（内容过长，已截断至 {limit} 字符）"
    return text


def execute_tool(name: str, raw_input: str) -> str:
    """解析参数并执行工具，返回可直接作为 Observation 的文本。"""
    arguments = parse_arguments(raw_input)
    try:
        result = call_tool(name, arguments)
    except ToolError as exc:
        logger.warning("工具调用被拒绝: %s", exc)
        return f"工具调用被拒绝: {exc}"
    except Exception as exc:
        logger.exception("工具 %s 执行异常", name)
        return f"工具执行失败: {type(exc).__name__}: {exc}"
    return _stringify(result)


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #

_FORMAT_HINT = (
    "[系统] 你上一条回复不符合 ReAct 格式。请严格按下面的格式重新输出，"
    "字段必须独占一行并以冒号结尾：\n"
    "Thought: <分析>\n"
    'Action: <工具名>\n'
    'Input: {"<参数名>": "<参数值>"}\n'
    "如果已经有足够信息，请直接输出 Answer: <最终答案>。"
)

_EMPTY_HINT = (
    "[系统] 你上一条回复是空的。请重新输出，内容必须包含 Answer: 或 Action: 字段。"
)

_REPEAT_HINT = (
    "[系统] 你已经多次用相同参数调用工具 {action}。请改用其它工具或其它参数，"
    "或者基于已有的 Observation 直接输出 Answer: <最终答案>。"
)

_FINALIZE_INSTRUCTION = (
    "[系统] 本轮的工具调用预算已经用尽。请立刻停止调用工具，"
    "只依据上面已经获得的 Observation，用 `Answer: ` 开头给出一行最终结论。"
    "如果信息仍然不足，就明确说明还缺少什么，不要编造。"
)

# Observation 回填用的消息角色。
#
# 不要改成 "tool": DeepSeek 当前接口要求 tool 角色消息必须带 tool_call_id，
# 而文本 ReAct 协议里并不存在真正的 function calling 调用记录。实测会直接失败：
#   422 Failed to deserialize ... messages[3]: missing field `tool_call_id`
# 所以按文本 ReAct 的通行做法，用一条带 "Observation: " 前缀的 user 消息回填。
OBSERVATION_ROLE = "user"
OBSERVATION_PREFIX = "Observation: "


def build_messages(memory: ConversationMemory) -> List[Dict[str, str]]:
    """拼装发给模型的消息列表。"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, *memory.get_messages()]


def _make_emitter(
    *,
    stream: bool,
    verbose: bool,
    on_token: Optional[TokenCallback],
) -> Tuple[TokenCallback, bool]:
    """返回 (token 回调, 是否打印思考表头)。"""
    if on_token is not None:
        return on_token, False
    if not stream or not verbose:
        return (lambda _piece: None), False
    return (lambda piece: print(piece, end="", flush=True)), True


def run_agent(
    user_input: str,
    memory: ConversationMemory,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    stream: bool = True,
    verbose: bool = True,
    on_token: Optional[TokenCallback] = None,
) -> str:
    """
    Agent 入口函数。

    参数:
        max_steps: 单轮最多调用模型的次数（步数上限）
        stream:    是否流式输出
        verbose:   是否把思考过程打印到 stdout
        on_token:  自定义流式回调；提供后由调用方接管输出

    返回:
        提取出的最终答案文本；失败时返回带说明的兜底文本。
    """
    if max_steps < 1:
        raise ValueError("max_steps 必须 >= 1")

    emit, show_header = _make_emitter(stream=stream, verbose=verbose, on_token=on_token)

    def notify(text: str) -> None:
        if verbose:
            print(f"\n{text}", flush=True)

    notice = notify if verbose else None

    def call_model(messages: List[Dict[str, str]]) -> str:
        if stream:
            if show_header:
                print("\n--- Agent 思考 ---")
            text, usage = complete_with_retry(
                messages, stream=True, on_token=emit, notice=notice
            )
            if show_header:
                print()
        else:
            text, usage = complete_with_retry(
                messages, stream=False, on_token=None, notice=notice
            )
            if verbose:
                print("\n--- Agent 思考 ---")
                print(text)
        if usage is not None:
            logger.info("token 用量: %s", usage)
        return text

    def finalize(reason: str) -> str:
        """预算用尽时补一次「只总结、不调用工具」的收尾调用。"""
        memory.add_message("user", _FINALIZE_INSTRUCTION)
        try:
            msg = call_model(build_messages(memory))
        except ModelCallError as exc:
            logger.error("收尾回答失败（%s）: %s", reason, exc)
            return f"[{reason}] 收尾回答也失败了: {exc}"
        if not msg.strip():
            return f"[{reason}] 模型没有给出最终回答。"
        memory.add_message("assistant", msg)
        parsed = parse_response(msg)
        return (parsed.answer or "").strip() or msg.strip()

    memory.add_message("user", user_input)

    turns = 0
    nudges = 0
    action_counts: Dict[Tuple[str, str], int] = {}

    while True:
        # ---- 步数上限 ----
        if turns >= max_steps:
            logger.warning("达到步数上限 max_steps=%s", max_steps)
            return finalize(f"已达到步数上限 {max_steps}")

        turns += 1

        try:
            msg = call_model(build_messages(memory))
        except ModelCallError as exc:
            logger.error("模型调用最终失败: %s", exc)
            return f"[模型调用失败] {exc}"

        # ---- 空响应 ----
        if not msg.strip():
            nudges += 1
            if nudges > MAX_NUDGES:
                return "[模型连续返回空响应，已终止]"
            memory.add_message("assistant", "(空响应)")
            memory.add_message("user", _EMPTY_HINT)
            continue

        memory.add_message("assistant", msg)
        parsed = parse_response(msg)

        # ---- 终止判定: 只认行首的 Answer 字段 ----
        answer = (parsed.answer or "").strip()
        if answer:
            return answer

        # ---- 没有 Action: 模型可能没按格式输出 ----
        if parsed.action is None:
            if not parsed.fields_found:
                # 完全没写字段，就把正文当作最终答案，避免空转
                logger.info("模型未使用 ReAct 字段，直接作为最终答案返回")
                return msg.strip()
            nudges += 1
            if nudges > MAX_NUDGES:
                logger.warning("模型连续 %s 次未给出 Action/Answer，终止", nudges)
                return msg.strip()
            memory.add_message("user", _FORMAT_HINT)
            continue

        # ---- 重复调用检测 ----
        key = (parsed.action, " ".join(parsed.raw_input.split()))
        action_counts[key] = action_counts.get(key, 0) + 1
        if action_counts[key] > MAX_REPEAT_ACTIONS:
            nudges += 1
            if nudges > MAX_NUDGES:
                logger.warning("检测到重复工具调用，终止: %s", parsed.action)
                return finalize(f"检测到重复工具调用 {parsed.action}")
            memory.add_message("user", _REPEAT_HINT.format(action=parsed.action))
            continue

        # ---- 执行工具并写回 Observation ----
        observation = execute_tool(parsed.action, parsed.raw_input)
        memory.add_message(OBSERVATION_ROLE, f"{OBSERVATION_PREFIX}{observation}")
