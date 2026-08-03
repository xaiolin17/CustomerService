"""
LangChain 1.2 中间件（Middleware）实现。

中间件是 Agent 执行过程中的钩子函数，在关键节点进行拦截、控制和增强。
参考：第08章：中间件（Middleware）

核心概念：
- 中间件 = Agent 执行流程中的钩子，可在"模型调用前/后"、"工具调用前/后"插入逻辑
- 日志、鉴权、重试、风控等横切需求通过中间件实现，不污染主流程

支持的钩子点：
- before_step / after_step：每个流程步骤的前后
- on_error：步骤发生错误时
"""

import time
import traceback
from typing import Any, Callable

from logger import log


# ---------------------------------------------------------------------------
# 中间件基类
# ---------------------------------------------------------------------------
class BaseMiddleware:
    """中间件基类 — 所有自定义中间件应继承此类。

    在 LangChain 1.2 中，中间件是 Agent 执行循环中的钩子函数，
    开发者通过重写以下方法实现自定义行为。
    """

    def before_step(self, step_name: str, **kwargs) -> dict:
        """步骤执行前调用。

        Args:
            step_name: 步骤名称（如 topic_filter, rag_retrieval, tool_call, llm_generate）
            **kwargs: 传入步骤的参数

        Returns:
            可修改后的参数字典
        """
        return kwargs

    def after_step(self, step_name: str, result: Any, duration: float, **kwargs) -> Any:
        """步骤执行后调用。

        Args:
            step_name: 步骤名称
            result: 步骤返回结果
            duration: 执行耗时（秒）
            **kwargs: 原始传入参数

        Returns:
            可修改后的结果
        """
        return result

    def on_error(self, step_name: str, error: Exception, duration: float, **kwargs):
        """步骤发生错误时调用。"""
        pass


# ---------------------------------------------------------------------------
# 日志记录中间件
# ---------------------------------------------------------------------------
class LoggingMiddleware(BaseMiddleware):
    """日志记录中间件 — 记录每个步骤的执行日志，包含参数、耗时、结果。

    对应 LangChain 1.2 中间件分类中的"日志与分析 — 追踪行为、调试、性能监控"。
    """

    def before_step(self, step_name: str, **kwargs) -> dict:
        log.info(f"[Middleware] >>> {step_name}")

        # 记录关键参数（截断长文本，避免日志爆炸）
        safe_params = {}
        for k, v in kwargs.items():
            if isinstance(v, str) and len(v) > 200:
                safe_params[k] = repr(v[:200] + "...")
            elif isinstance(v, list) and len(v) > 5:
                safe_params[k] = f"[{len(v)} items]"
            else:
                safe_params[k] = v
        if safe_params:
            log.debug(f"[Middleware]   {step_name} 参数: {safe_params}")

        return kwargs

    def after_step(self, step_name: str, result: Any, duration: float, **kwargs) -> Any:
        duration_str = f"{duration:.3f}s" if duration < 60 else f"{duration / 60:.1f}min"
        log.info(f"[Middleware] <<< {step_name} (耗时: {duration_str})")

        # 记录结果摘要
        if isinstance(result, str) and len(result) > 200:
            log.debug(f"[Middleware]   {step_name} 结果摘要: {result[:200]}...")
        elif isinstance(result, dict):
            # 仅记录关键字段
            summary = {k: v for k, v in result.items() if k in ("tool_called", "content")}
            if summary:
                log.debug(f"[Middleware]   {step_name} 结果: {summary}")
            else:
                log.debug(f"[Middleware]   {step_name} 结果: {result}")

        return result

    def on_error(self, step_name: str, error: Exception, duration: float, **kwargs):
        duration_str = f"{duration:.3f}s" if duration < 60 else f"{duration / 60:.1f}min"
        log.error(f"[Middleware] XXX {step_name} 失败 (耗时: {duration_str}) | {error}")
        log.debug(f"[Middleware]   {step_name} 异常堆栈:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 性能计时中间件
# ---------------------------------------------------------------------------
class TimingMiddleware(BaseMiddleware):
    """性能计时中间件 — 记录慢步骤的警告。"""

    SLOW_THRESHOLD = 3.0  # 超过 3 秒记录警告

    def after_step(self, step_name: str, result: Any, duration: float, **kwargs) -> Any:
        if duration > self.SLOW_THRESHOLD:
            log.warning(f"[Performance] {step_name} 耗时过长: {duration:.3f}s")
        return result


# ---------------------------------------------------------------------------
# 中间件管道
# ---------------------------------------------------------------------------
class MiddlewarePipeline:
    """中间件管道 — 按顺序执行中间件链。

    用法与 LangChain 1.2 的 create_agent(middleware=[...]) 类似：
    将多个中间件组合成一个管道，统一管理各步骤的钩子调用。
    """

    def __init__(self, middlewares: list[BaseMiddleware] | None = None):
        self._middlewares = middlewares or []

    def add(self, middleware: BaseMiddleware):
        """添加中间件到管道。"""
        self._middlewares.append(middleware)

    def execute(self, step_name: str, fn: Callable, **kwargs) -> Any:
        """执行带中间件包装的步骤。

        Args:
            step_name: 步骤名称，用于日志标识
            fn: 要执行的函数
            **kwargs: 传递给函数的参数

        Returns:
            函数执行结果（可能被中间件修改）
        """
        # before hooks（正向顺序执行，每个中间件可以修改参数）
        current_kwargs = kwargs
        for mw in self._middlewares:
            current_kwargs = mw.before_step(step_name, **current_kwargs)

        # 执行目标函数
        start = time.time()
        try:
            result = fn(**current_kwargs)
            duration = time.time() - start

            # after hooks（反向顺序执行，每个中间件可以修改结果）
            for mw in reversed(self._middlewares):
                result = mw.after_step(step_name, result, duration, **kwargs)

            return result
        except Exception as e:
            duration = time.time() - start
            for mw in reversed(self._middlewares):
                mw.on_error(step_name, e, duration, **kwargs)
            raise


# ---------------------------------------------------------------------------
# 全局默认管道
# ---------------------------------------------------------------------------
# 创建默认的中间件管道实例，包含日志和性能监控
pipeline = MiddlewarePipeline([
    LoggingMiddleware(),
    TimingMiddleware(),
])