"""
Agent Memory Module

负责管理 Agent 上下文记忆。
"""

from typing import Dict, List, Optional


class ConversationMemory:
    """按条数限制的短期对话记忆。"""

    def __init__(self, max_messages: int = 20):
        if max_messages < 1:
            raise ValueError("max_messages 必须 >= 1")

        # 最大保存消息数量
        self.max_messages = max_messages

        # 保存历史消息
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        添加一条消息。

        role: system / user / assistant / tool
        """
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        """
        控制上下文长度。

        旧实现直接 pop(0)，会让窗口以 tool 消息开头，把 assistant/tool 配对切一半。
        这里在丢弃溢出条数之后继续向后移动切点，保证窗口第一条不是 tool 消息。
        """
        overflow = len(self.messages) - self.max_messages
        if overflow <= 0:
            return

        cut = min(overflow, len(self.messages))
        last_index = len(self.messages) - 1
        while cut < last_index and self.messages[cut]["role"] == "tool":
            cut += 1

        del self.messages[:cut]

    def get_messages(self) -> List[Dict[str, str]]:
        """获取全部历史消息（返回副本，避免外部误改内部状态）。"""
        return list(self.messages)

    def clear(self) -> None:
        """清空记忆。"""
        self.messages = []

    def get_last_message(self) -> Optional[Dict[str, str]]:
        """获取最近一次消息。"""
        if self.messages:
            return self.messages[-1]
        return None
