"""
会话短期记忆管理器。

LangChain 1.2.x 记忆模式：
- 短期记忆 = State（消息列表）+ Checkpointer + Thread ID
- 直接存储消息列表，无需 ConversationBufferWindowMemory 等旧 API

特点：
- 30 分钟 TTL 过期，过期自动清理
- 按 session_id（thread_id）隔离
- 支持上下文压缩（超出最大轮次时自动摘要）
- 1M token 上下文管理：达到阈值时压缩旧消息，保留最近 N 条消息
- 主动过期检查与懒清理
"""

import time
from dataclasses import dataclass, field

from config import settings
from logger import log


@dataclass
class SessionMemory:
    """单个会话的记忆数据"""
    messages: list = field(default_factory=list)   # 消息对象列表
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    turn_count: int = 0
    summary: str = ""  # 压缩后的摘要


class MemoryManager:
    """会话短期记忆管理器 - 30 分钟 TTL，按 session_id 隔离"""

    def __init__(self):
        self._sessions: dict[str, SessionMemory] = {}

    def _is_expired(self, session: SessionMemory) -> bool:
        """检查会话是否已过期（超过 TTL 未活动）"""
        elapsed = time.time() - session.last_active_at
        return elapsed > settings.memory_ttl_minutes * 60

    def _cleanup_expired(self):
        """清理所有过期会话"""
        expired_ids = [
            sid for sid, session in self._sessions.items()
            if self._is_expired(session)
        ]
        for sid in expired_ids:
            del self._sessions[sid]
            log.debug(f"已清理过期会话: {sid[:8]}...")

    # ──────────────────────────────────────────────────────────────────
    # Token 估算
    # ──────────────────────────────────────────────────────────────────
    def estimate_tokens(self, session_id: str) -> int:
        """估算当前会话上下文的 token 数量（含摘要 + 历史消息）

        用于判断是否即将超过 1M token 上限。
        采用保守估算：中英文混合文本约 2 字符 / token。
        """
        session = self._sessions.get(session_id)
        if not session:
            return 0

        total_chars = 0

        # 摘要部分
        if session.summary:
            total_chars += len(session.summary)

        # 历史消息部分
        for msg in session.messages:
            if hasattr(msg, 'content'):
                total_chars += len(msg.content)
            elif isinstance(msg, dict) and 'content' in msg:
                total_chars += len(msg['content'])

        # 粗略估算：中英文混合 ~2 chars/token
        return total_chars // 2

    def needs_token_compression(self, session_id: str) -> bool:
        """判断是否需要进行上下文压缩（基于 token 数量阈值）"""
        session = self._sessions.get(session_id)
        if not session:
            return False

        estimated = self.estimate_tokens(session_id)
        threshold = int(settings.context_max_tokens * settings.context_compress_ratio)
        return estimated >= threshold

    # ──────────────────────────────────────────────────────────────────
    # 上下文压缩（保留最近 N 条消息）
    # ──────────────────────────────────────────────────────────────────
    def compress_and_keep_recent(self, session_id: str, summary_llm, summary_prompt) -> str:
        """压缩历史对话，保留最近 N 条消息，其余部分摘要化。

        Args:
            session_id: 会话 ID。
            summary_llm: 用于生成摘要的 LLM 实例。
            summary_prompt: ChatPromptTemplate 实例，用于构建摘要消息。

        Returns:
            更新后的摘要字符串。
        """
        session = self._sessions.get(session_id)
        if not session:
            return ""

        keep_count = settings.context_keep_last_messages  # 默认 3

        all_messages = list(session.messages)

        if len(all_messages) <= keep_count * 2:
            return session.summary or ""

        # 分离：保留最近 N 轮，压缩其余部分
        recent_messages = all_messages[-keep_count * 2:]
        old_messages = all_messages[:-keep_count * 2]

        # 将旧消息转为纯文本用于摘要
        old_text = ""
        for msg in old_messages:
            msg_type = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, 'type', '')
            msg_content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, 'content', "")
            role = "用户" if msg_type in ("human", "user") else "客服"
            old_text += f"{role}: {msg_content}\n"

        # 使用 LLM 生成摘要（通过 Message 消息模式）
        try:
            msgs = summary_prompt.format_messages(history=old_text)
            response = summary_llm.invoke(msgs)
            summary = response.content.strip()
        except Exception as e:
            log.error(f"上下文压缩摘要生成失败: {e}")
            summary = old_text[:500]  # 截断保底

        # 更新会话摘要（追加方式）
        if session.summary:
            session.summary = f"{session.summary}\n【后续对话摘要】{summary}"
        else:
            session.summary = summary

        # 保留最近消息，清空旧消息
        session.messages.clear()
        session.messages.extend(recent_messages)

        log.info(
            f"上下文压缩完成: session_id={session_id[:8]}..., "
            f"保留最近 {keep_count} 条消息, "
            f"旧消息已摘要 ({len(old_messages)} 条 → {len(summary)} 字符)"
        )

        return session.summary

    # ──────────────────────────────────────────────────────────────────
    # 基础操作
    # ──────────────────────────────────────────────────────────────────
    def get_or_create_session(self, session_id: str) -> SessionMemory:
        """获取或创建会话"""
        self._cleanup_expired()

        now = time.time()

        if session_id not in self._sessions:
            session = SessionMemory()
            self._sessions[session_id] = session
            log.info(f"创建新会话: session_id={session_id[:8]}...")
        else:
            session = self._sessions[session_id]
            session.last_active_at = now

        return session

    def add_messages(self, session_id: str, *msgs):
        """向会话追加消息"""
        session = self.get_or_create_session(session_id)
        for msg in msgs:
            session.messages.append(msg)
        session.last_active_at = time.time()

    def get_messages(self, session_id: str) -> list:
        """获取会话的消息列表（返回副本，避免外部修改）"""
        session = self._sessions.get(session_id)
        if not session:
            return []
        session.last_active_at = time.time()
        return list(session.messages)

    def get_session_summary(self, session_id: str) -> str:
        """获取会话的摘要信息"""
        session = self._sessions.get(session_id)
        if session:
            return session.summary
        return ""

    def update_summary(self, session_id: str, summary: str):
        """更新会话摘要"""
        session = self._sessions.get(session_id)
        if session:
            session.summary = summary
            log.debug(f"更新会话摘要: session_id={session_id[:8]}...")

    def increment_turn(self, session_id: str):
        """增加会话轮次计数"""
        session = self._sessions.get(session_id)
        if session:
            session.turn_count += 1
            session.last_active_at = time.time()

    def get_turn_count(self, session_id: str) -> int:
        """获取当前会话轮次"""
        session = self._sessions.get(session_id)
        return session.turn_count if session else 0

    def needs_compression(self, session_id: str) -> bool:
        """判断是否需要压缩历史对话（超出最大轮次阈值）"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        return session.turn_count >= settings.memory_max_turns * 0.8

    def clear_memory(self, session_id: str):
        """清除指定会话的记忆"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            log.info(f"已清除会话记忆: session_id={session_id[:8]}...")

    def clear_all(self):
        """清除所有会话记忆"""
        self._sessions.clear()
        log.info("已清除所有会话记忆")

    def get_active_session_count(self) -> int:
        """获取当前活跃（未过期）会话数"""
        self._cleanup_expired()
        return len(self._sessions)


# 全局单例
memory_manager = MemoryManager()