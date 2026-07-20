"""
session.py — Conversation history + context management for the terminal agent.
Tracks message history, auto-summarizes when context grows too large.
"""

from dataclasses import dataclass


@dataclass
class Message:
    role: str  # "user", "assistant", "tool"
    content: str


class Session:
    """Manages conversation state across turns. Summarizes old context
    when the message count exceeds the configured limit."""

    def __init__(self, config: dict):
        self.max_messages = config.get("max_turns_before_summary", 15) * 2
        self.messages: list[Message] = []
        self.summary: str = ""
        self._task: str = ""

    def set_task(self, task: str):
        self._task = task

    def add_user(self, text: str):
        self.messages.append(Message("user", text))
        self._maybe_summarize()

    def add_assistant(self, text: str):
        self.messages.append(Message("assistant", text))

    def add_tool(self, text: str):
        self.messages.append(Message("tool", text))

    def build_context_for_llm(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[PREVIOUS CONTEXT]\n{self.summary}\n[/PREVIOUS CONTEXT]")
        for m in self.messages:
            if m.role == "user":
                parts.append(f"USER: {m.content}")
            elif m.role == "assistant":
                parts.append(f"ASSISTANT: {m.content}")
            elif m.role == "tool":
                parts.append(f"TOOL OUTPUT: {m.content}")
        return "\n\n".join(parts)

    def last_assistant(self) -> str:
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m.content
        return ""

    def _maybe_summarize(self):
        if len(self.messages) < self.max_messages:
            return

        split = len(self.messages) // 2
        old = self.messages[:split]
        self.messages = self.messages[split:]

        summary_parts = [f"Task: {self._task}"] if self._task else []
        actions = []
        for m in old:
            if m.role == "tool":
                first_line = m.content.split("\n")[0][:120]
                actions.append(f"  Tool result: {first_line}")
            elif m.role == "assistant" and len(m.content) < 200:
                actions.append(f"  Assistant: {m.content[:200]}")

        if actions:
            summary_parts.append("Key actions taken:")
            summary_parts.extend(actions[-20:])

        self.summary = "\n".join(summary_parts)

    def clear(self):
        self.messages.clear()
        self.summary = ""
