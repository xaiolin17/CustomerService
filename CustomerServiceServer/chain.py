"""
LangChain 处理链：集成主题过滤、RAG 检索、工具调用、LLM 推理。

流程：
1. 主题过滤 → 判断是否电商相关，防越狱
2. 上下文管理 → 检查 token 量，必要时压缩（保留最近 N 条）
3. 工具调用（指数级回退）→ 判断是否需要查询商品/订单
4. LLM 推理 → 结合上下文、记忆、工具结果生成回答

中间件集成：
- 每个主要步骤通过中间件管道（MiddlewarePipeline）执行
- 中间件在"模型调用前/后"、"工具调用前/后"、"各步骤前/后"设置钩子
- 日志记录中间件追踪每一步的执行参数、耗时和结果
- 参考 LangChain 1.2 第08章：中间件（Middleware）
"""

from langchain_openai import ChatOpenAI
from langchain_community.llms.fake import FakeListLLM
from langchain_core.messages import HumanMessage, AIMessage

from config import settings
from prompts import chat_prompt, topic_filter_prompt, summary_prompt
from memory import memory_manager
from milvus_client import MilvusClient
from tools import execute_tool_call, set_current_user, tool_map
from logger import log
from middleware import pipeline  # 中间件管道：LoggingMiddleware + TimingMiddleware

# 全局 Milvus 客户端
milvus_client = MilvusClient()


# ---------------------------------------------------------------------------
# LLM 工厂
# ---------------------------------------------------------------------------
def _create_llm(model: str, temperature: float = 0.1):
    """创建 ChatOpenAI 实例（阿里云通义千问兼容模式）"""
    return ChatOpenAI(
        model=model,
        api_key=settings.ali_api_key,
        base_url=settings.ali_openai_compatible_endpoint,
        temperature=temperature,
        max_tokens=2048,
        streaming=False,
    )


def _create_filter_llm():
    """创建用于主题过滤的 LLM（使用 Flash 模型节省成本）"""
    return ChatOpenAI(
        model=settings.llm_fallback_model,
        api_key=settings.ali_api_key,
        base_url=settings.ali_openai_compatible_endpoint,
        temperature=0.1,  # 低温度确保判断稳定
        max_tokens=50,    # 只需要输出 yes/no/jailbreak_attempt
    )


def get_llm():
    """获取主模型 LLM 实例，失败时回退到备用模型"""
    if settings.use_mock_llm:
        return FakeListLLM(responses=["这是一条模拟回复，请替换为真实 LLM。"])

    try:
        return _create_llm(settings.llm_primary_model)
    except Exception as e:
        log.warning(f"主模型加载失败，回退到备用模型: {e}")
        try:
            return _create_llm(settings.llm_fallback_model)
        except Exception as e2:
            log.error(f"备用模型加载也失败: {e2}")
            return FakeListLLM(responses=["LLM 服务暂时不可用，请稍后再试。"])


# ---------------------------------------------------------------------------
# 上下文构建
# ---------------------------------------------------------------------------
def build_context(documents: list) -> str:
    """构建检索上下文文本"""
    if not documents:
        return "暂无相关信息。"

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = f" (来源: {doc['file_name']})" if doc.get('file_name') else ""
        context_parts.append(f"[{i}]{doc['text']}{source}")

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# 主题过滤（防越狱、防跑题）
# ---------------------------------------------------------------------------
def check_topic(user_input: str) -> tuple[bool, str]:
    """检查用户输入是否与电商相关，防止越狱和跑题。

    Returns:
        (is_allowed: bool, reason: str)
        - is_allowed=True: 允许继续处理
        - is_allowed=False: 拒绝，reason 为拒绝原因
    """
    try:
        filter_llm = _create_filter_llm()
        # 使用 ChatPromptTemplate.format_messages() 构建消息列表
        messages = topic_filter_prompt.format_messages(input=user_input)
        result = filter_llm.invoke(messages).content.strip().lower()

        log.info(f"主题过滤结果: {result} | input=\"{user_input[:40]}...\"")

        if result == "jailbreak_attempt":
            return False, "jailbreak"
        elif result == "no":
            return False, "off_topic"
        else:
            return True, ""
    except Exception as e:
        log.error(f"主题过滤失败，默认放行: {e}")
        return True, ""


# ---------------------------------------------------------------------------
# 工具调用决策（使用 LangChain 1.2 model.bind_tools() 模式）
# ---------------------------------------------------------------------------
def decide_tool_call(user_input: str, session_id: str) -> dict:
    """判断是否需要调用工具，并返回工具调用结果。

    LangChain 1.2.x 标准流程：
    1. 注入当前用户上下文（set_current_user）
    2. 绑定工具到模型：model.bind_tools([...])
    3. 模型返回 response.tool_calls 表明是否需要调用工具
    4. 通过 execute_tool_call() 执行工具（带指数级回退）
    5. 返回工具执行结果

    模型调用和工具调用均通过中间件包装，实现日志记录和性能监控。

    Returns:
        {"tool_called": bool, "tool_results": str}
    """
    try:
        # 注入当前用户上下文（工具内部通过 _current_user_id 过滤数据）
        set_current_user(session_id)

        # 创建 LLM 并绑定工具
        llm = _create_llm(settings.llm_primary_model)
        model_with_tools = llm.bind_tools(list(tool_map.values()))

        # 调用模型，让模型决定是否调用工具
        # 通过中间件包装模型调用，记录日志和性能
        response = pipeline.execute(
            "model_invoke",
            model_with_tools.invoke,
            input=user_input,
        )

        # 检查模型是否想调用工具
        if not response.tool_calls:
            log.info("模型决定不调用工具")
            return {"tool_called": False, "tool_results": ""}

        # 执行所有工具调用（带指数级回退，通过中间件包装）
        log.info(f"模型决定调用工具: {[tc['name'] for tc in response.tool_calls]}")
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_msg = pipeline.execute(
                f"tool_execute({tool_call['name']})",
                execute_tool_call,
                tool_call=tool_call,
            )
            tool_messages.append(tool_msg)

        # 将工具结果拼接为字符串（兼容下游流程）
        results = []
        for msg in tool_messages:
            results.append(f"[工具 {msg.name}]: {msg.content}")
        combined = "\n\n".join(results)

        log.info(f"工具调用完成: {combined[:100]}...")
        return {"tool_called": True, "tool_results": combined}

    except Exception as e:
        log.error(f"工具调用决策失败: {e}")
        return {"tool_called": False, "tool_results": ""}


# ---------------------------------------------------------------------------
# 历史对话压缩（上下文管理）
# ---------------------------------------------------------------------------
def compress_history(session_id: str, history_str: str) -> str:
    """当历史对话过长时，进行压缩摘要"""
    try:
        llm = _create_llm(settings.llm_fallback_model, temperature=0.1)
        messages = summary_prompt.format_messages(history=history_str)
        response = llm.invoke(messages)
        summary = response.content.strip()
        memory_manager.update_summary(session_id, summary)
        log.info(f"历史对话已压缩: session_id={session_id[:8]}...")
        return summary
    except Exception as e:
        log.error(f"历史对话压缩失败: {e}")
        return history_str[:500]  # 截断保底


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process_chat(session_id: str, user_input: str) -> dict:
    """处理聊天消息的完整流程

    每个主要步骤通过中间件管道执行，自动记录日志、耗时和性能指标。
    """
    log.info(f"处理聊天请求: session_id={session_id[:8]}..., input=\"{user_input[:50]}...\"")

    # ── 步骤 1: 主题过滤（通过中间件包装） ─────────────────────────
    is_allowed, reason = pipeline.execute(
        "topic_filter",
        check_topic,
        user_input=user_input,
    )
    if not is_allowed:
        if reason == "jailbreak":
            reply = "抱歉，我只能回答电商购物相关问题。请问有什么商品或订单需要帮助的吗？"
        else:
            reply = "我是电商客服助手，只处理与购物相关的问题。请咨询商品、订单、物流等方面的问题。"
        log.info(f"主题过滤拒绝: reason={reason}, input=\"{user_input[:30]}...\"")
        return {"content": reply, "sources": None}

    # ── 步骤 2: RAG 检索（通过中间件包装） ─────────────────────────
    documents = pipeline.execute(
        "rag_search",
        milvus_client.search,
        query=user_input,
    )
    context = build_context(documents)

    # ── 步骤 3: 上下文管理（512KB token 检查，通过中间件包装） ─────
    if memory_manager.needs_token_compression(session_id):
        log.info(f"触发 512KB 上下文压缩: session_id={session_id[:8]}...")
        summary_llm = _create_llm(settings.llm_fallback_model, temperature=0.1)
        # 执行压缩（保留最近 3 条消息）
        pipeline.execute(
            "context_compress",
            memory_manager.compress_and_keep_recent,
            session_id=session_id,
            summary_llm=summary_llm,
            summary_prompt=summary_prompt,
        )

    # ── 步骤 4: 工具调用（带指数级回退 + 用户元数据注入，通过中间件包装） ──
    tool_result = pipeline.execute(
        "tool_decision",
        decide_tool_call,
        user_input=user_input,
        session_id=session_id,
    )
    tool_results_str = tool_result["tool_results"]

    # ── 步骤 5: 获取会话记忆 ──────────────────────────────────────
    history_summary = memory_manager.get_session_summary(session_id) or ""

    # 获取最近消息列表（符合 LangChain 1.2 State 模式）
    recent_messages = memory_manager.get_messages(session_id)
    recent_text = ""
    for msg in recent_messages:
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            prefix = "用户" if msg.type == "human" else "客服"
            recent_text += f"{prefix}: {msg.content}\n"

    if history_summary and recent_text:
        history_str = f"【历史摘要】{history_summary}\n【最近对话】{recent_text}"
    elif history_summary:
        history_str = f"【历史摘要】{history_summary}"
    else:
        history_str = recent_text or "暂无历史对话"

    # ── 步骤 6: LLM 生成回答（通过中间件包装） ────────────────────
    llm = get_llm()

    # 构建消息列表（使用 ChatPromptTemplate，符合 LangChain 1.2 Message 规范）
    prompt_input = {
        "context": context,
        "history": history_str,
        "tool_results": tool_results_str if tool_results_str else "无",
        "input": user_input,
    }
    messages = chat_prompt.format_messages(**prompt_input)

    # 估算最终 prompt 的 token 数量并记录
    estimated_prompt_tokens = len(str(messages)) // 2
    log.info(f"最终 prompt 估算 token: {estimated_prompt_tokens}")

    try:
        # LLM 调用（通过中间件包装，记录日志和性能）
        # 注意：invoke 的参数名是 input，不是 messages
        response = pipeline.execute(
            "llm_generate",
            llm.invoke,
            input=messages,
        )
        reply_content = response.content if isinstance(response.content, str) else str(response.content)

        log.info(f"生成回复完成: session_id={session_id[:8]}..., 回复长度={len(reply_content)}")

        # 保存消息到会话记忆（符合 LangChain 1.2 State 模式）
        memory_manager.add_messages(
            session_id,
            HumanMessage(content=user_input),
            AIMessage(content=reply_content),
        )
        memory_manager.increment_turn(session_id)

        # 构建 sources 信息
        sources = list(set(
            doc.get("file_name", "") for doc in documents if doc.get("file_name")
        ))

        return {
            "content": reply_content,
            "sources": sources if sources else None,
        }

    except Exception as e:
        log.error(f"LLM 调用失败: {e}")

        # 尝试使用备用模型重试
        try:
            fallback_llm = _create_llm(settings.llm_fallback_model)
            response = fallback_llm.invoke(messages)
            reply_content = response.content if isinstance(response.content, str) else str(response.content)

            log.info(f"备用模型回复成功: session_id={session_id[:8]}...")

            memory_manager.add_messages(
                session_id,
                HumanMessage(content=user_input),
                AIMessage(content=reply_content),
            )
            memory_manager.increment_turn(session_id)

            sources = list(set(
                doc.get("file_name", "") for doc in documents if doc.get("file_name")
            ))

            return {
                "content": reply_content,
                "sources": sources if sources else None,
            }
        except Exception as e2:
            log.error(f"备用模型也失败: {e2}")
            return {
                "content": "抱歉，我暂时无法回答您的问题，请稍后再试。",
                "sources": None,
            }