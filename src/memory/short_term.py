# ============================================================
# 短期记忆（对话窗口）
# ------------------------------------------------------------
# 大模型是"无状态"的：每次调用都得把历史对话重新塞进提示词。
# 但上下文长度有限，不能无限堆——所以要维护一个"滑动窗口"：
# 只保留最近 N 轮对话，更早的交给长期记忆去按需召回。
# ============================================================
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Turn:
    """一轮对话：用户问 + 助手答。"""
    question: str
    answer: str


class ShortTermMemory:
    """固定容量的滑动窗口，超出容量自动丢弃最旧的一轮。"""

    def __init__(self, window: int = 6):
        self.window = window
        # deque(maxlen=...) 天然实现"先进先出、自动淘汰"
        self._turns: deque[Turn] = deque(maxlen=window)

    def add(self, question: str, answer: str) -> None:
        self._turns.append(Turn(question, answer))

    def as_text(self) -> str:
        """把窗口内的对话拼成一段可注入提示词的文本。"""
        if not self._turns:
            return "（无历史对话）"
        lines = []
        for t in self._turns:
            lines.append(f"用户: {t.question}")
            lines.append(f"助手: {t.answer}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
