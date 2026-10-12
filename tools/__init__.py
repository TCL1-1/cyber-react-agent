"""
工具注册表与调用入口。

TOOLS 仍然只是「名字 -> 可调用对象」的字典，方便继续新增工具；
参数按函数签名绑定，所以工具函数可以保持最朴素的写法。
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List

from tools.file_tool import read_file
from tools.rag_tool import rag_search
from tools.search_tool import web_search

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """工具不存在，或参数不合法。会被转换成 Observation 回传给模型。"""


TOOLS: Dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "read_file": read_file,
    "rag_search": rag_search,
}


def _bindable_parameters(func: Callable[..., Any]) -> List[inspect.Parameter]:
    return [
        param
        for param in inspect.signature(func).parameters.values()
        if param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


def _describe_parameters(parameters: List[inspect.Parameter]) -> str:
    if not parameters:
        return "（无）"
    parts = []
    for param in parameters:
        if param.default is inspect.Parameter.empty:
            parts.append(f"{param.name}（必填）")
        else:
            parts.append(f"{param.name}（可选，默认 {param.default!r}）")
    return "、".join(parts)


def _unknown_tool_message(name: str) -> str:
    return f"不存在工具 {name!r}，可用工具: {', '.join(sorted(TOOLS))}"


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """
    执行工具。

    - 未注册的工具、参数缺失、参数无法识别 -> 抛 ToolError（由上层转成 Observation）
    - 模型把参数名写错时，若该工具只有一个参数，则自动按位置兜底绑定
    """
    if name not in TOOLS:
        raise ToolError(_unknown_tool_message(name))

    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolError(
            f"工具参数必须是一个 JSON 对象，实际收到 {type(arguments).__name__}"
        )

    func = TOOLS[name]
    signature = inspect.signature(func)
    parameters = _bindable_parameters(func)
    accepts_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )

    kwargs: Dict[str, Any] = {}
    unknown: List[str] = []
    for key, value in arguments.items():
        if key in signature.parameters or accepts_var_keyword:
            kwargs[key] = value
        else:
            unknown.append(key)

    # 兼容: 模型常给单个参数起别的名字，例如 read_file 传 {"file_path": "..."}
    if not kwargs and len(parameters) == 1 and len(arguments) == 1:
        kwargs[parameters[0].name] = next(iter(arguments.values()))
        unknown = []

    if not kwargs and unknown:
        raise ToolError(
            f"{name} 无法识别参数 {unknown}。期望参数: {_describe_parameters(parameters)}"
        )

    if unknown:
        logger.warning("工具 %s 收到未知参数并已忽略: %s", name, unknown)

    missing = [
        param.name
        for param in parameters
        if param.default is inspect.Parameter.empty and param.name not in kwargs
    ]
    if missing:
        raise ToolError(
            f"{name} 缺少必需参数 {missing}。期望参数: {_describe_parameters(parameters)}"
        )

    return func(**kwargs)
