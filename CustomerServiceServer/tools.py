"""
电商客服工具集：商品查询、订单查询等业务工具。

使用 LangChain 1.2.x 推荐的 @tool 装饰器 + model.bind_tools() 模式。

设计原则：
- 工具使用 @tool 装饰器定义，函数签名 + Google 风格 docstring 自动生成 tool schema
- 工具绑定到模型后，模型通过 response.tool_calls 返回调用意图
- 工具执行返回 ToolMessage，回传给模型生成最终回答
- 用户元数据（user_id）通过模块级上下文注入，不暴露给 LLM
- 工具调用失败时采用指数级回退重试机制
"""

import json
import time
import random

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from config import settings
from logger import log


# ---------------------------------------------------------------------------
# 模拟数据库（含 user_id 字段，用于按用户隔离数据）
# ---------------------------------------------------------------------------
MOCK_PRODUCTS = {
    "iPhone 15": {
        "price": "¥6,999",
        "stock": 128,
        "description": "Apple iPhone 15 128GB 蓝色，A16 芯片，4800万像素主摄",
        "specs": "6.1英寸OLED屏幕，A16芯片，128GB存储",
    },
    "MacBook Air M3": {
        "price": "¥8,999",
        "stock": 56,
        "description": "Apple MacBook Air M3 芯片 8GB 256GB 深空灰",
        "specs": "13.6英寸Liquid Retina屏幕，M3芯片，8GB统一内存，256GB存储",
    },
    "AirPods Pro 2": {
        "price": "¥1,899",
        "stock": 230,
        "description": "Apple AirPods Pro 2 降噪耳机 USB-C接口",
        "specs": "主动降噪，自适应音频，IP54防水，USB-C充电盒",
    },
}

MOCK_ORDERS = {
    "ORD20240801001": {
        "user_id": "user_001",
        "status": "已发货",
        "product": "iPhone 15",
        "quantity": 1,
        "total": "¥6,999",
        "logistics": "顺丰快递 SF1234567890",
        "estimated_delivery": "2024-08-05",
        "address": "北京市朝阳区xxx路xxx号",
    },
    "ORD20240801002": {
        "user_id": "user_001",
        "status": "待发货",
        "product": "MacBook Air M3",
        "quantity": 1,
        "total": "¥8,999",
        "logistics": "待分配",
        "estimated_delivery": "2024-08-07",
        "address": "上海市浦东新区xxx路xxx号",
    },
    "ORD20240801003": {
        "user_id": "user_002",
        "status": "已完成",
        "product": "AirPods Pro 2",
        "quantity": 2,
        "total": "¥3,798",
        "logistics": "中通快递 ZT9876543210",
        "estimated_delivery": "2024-07-30",
        "address": "广州市天河区xxx路xxx号",
    },
}


# ---------------------------------------------------------------------------
# 用户上下文（调用链在执行工具前注入当前用户标识）
# ---------------------------------------------------------------------------
_current_user_id: str = ""


def set_current_user(user_id: str):
    """设置当前工具调用的用户上下文（由 chain.py 在调用前注入）"""
    global _current_user_id
    _current_user_id = user_id


# ---------------------------------------------------------------------------
# 工具定义（使用 @tool 装饰器 + Google 风格 docstring）
# ---------------------------------------------------------------------------
@tool
def query_product(product_name: str) -> str:
    """
    查询商品信息，包括价格、库存、描述和规格。

    当用户询问商品详情、价格、是否有货、功能参数时调用此工具。

    Args:
        product_name: 商品名称，支持模糊匹配，如"iPhone 15"、"MacBook"、"AirPods"
    Returns:
        商品信息的 JSON 字符串，包含名称、价格、库存、描述和规格
    """
    log.info(f"工具调用: query_product, product_name=\"{product_name}\", user_id=\"{_current_user_id}\"")

    # 模糊匹配
    for key, info in MOCK_PRODUCTS.items():
        if product_name.lower() in key.lower():
            result = {
                "name": key,
                "price": info["price"],
                "stock": info["stock"],
                "description": info["description"],
                "specs": info["specs"],
            }
            log.info(f"工具返回: 找到商品 [{key}]")
            return json.dumps(result, ensure_ascii=False)

    log.warning(f"工具返回: 未找到商品 [{product_name}]")
    return json.dumps({
        "error": f"未找到商品「{product_name}」",
        "suggestion": "请提供更准确的商品名称，或浏览我们的商品分类",
    }, ensure_ascii=False)


@tool
def query_order(order_id: str) -> str:
    """
    查询订单状态和物流信息。

    当用户询问订单发货状态、物流进度、预计送达时间、收货地址时调用此工具。
    注意：仅返回当前登录用户的订单信息。

    Args:
        order_id: 订单编号，格式如"ORD20240801001"
    Returns:
        订单信息的 JSON 字符串，包含状态、商品、数量、总价、物流、预计送达时间、地址
    """
    log.info(f"工具调用: query_order, order_id=\"{order_id}\", user_id=\"{_current_user_id}\"")

    user_id = _current_user_id

    # 精确匹配
    order = MOCK_ORDERS.get(order_id.strip().upper())
    if order:
        # 校验订单归属：只返回当前用户的订单
        if user_id and order.get("user_id") and order["user_id"] != user_id:
            log.warning(f"越权访问: user_id={user_id} 尝试访问非本人订单 {order_id}")
            return json.dumps({
                "error": f"未找到订单「{order_id}」",
                "suggestion": "请核对订单编号，仅可查询本人的订单信息",
            }, ensure_ascii=False)

        log.info(f"工具返回: 找到订单 [{order_id}]")
        return json.dumps({
            "order_id": order_id,
            "status": order["status"],
            "product": order["product"],
            "quantity": order["quantity"],
            "total": order["total"],
            "logistics": order["logistics"],
            "estimated_delivery": order["estimated_delivery"],
            "address": order["address"],
        }, ensure_ascii=False)

    # 尝试模糊匹配部分订单号
    for key in MOCK_ORDERS:
        if order_id in key:
            order = MOCK_ORDERS[key]
            if user_id and order.get("user_id") and order["user_id"] != user_id:
                continue  # 跳过非当前用户的订单

            log.info(f"工具返回: 模糊匹配到订单 [{key}]")
            return json.dumps({
                "order_id": key,
                "status": order["status"],
                "product": order["product"],
                "quantity": order["quantity"],
                "total": order["total"],
                "logistics": order["logistics"],
                "estimated_delivery": order["estimated_delivery"],
                "address": order["address"],
            }, ensure_ascii=False)

    log.warning(f"工具返回: 未找到订单 [{order_id}]")
    return json.dumps({
        "error": f"未找到订单「{order_id}」",
        "suggestion": "请核对订单编号，仅可查询本人的订单信息",
    }, ensure_ascii=False)


# 工具列表（供 model.bind_tools() 使用）
all_tools = [query_product, query_order]


# 工具名称 -> 工具对象映射（便于快速查找）
tool_map = {tool.name: tool for tool in all_tools}


# ---------------------------------------------------------------------------
# 工具执行辅助函数（带指数级回退，供 chain.py 使用）
# ---------------------------------------------------------------------------
def execute_tool_call(tool_call: dict) -> ToolMessage:
    """执行单个工具调用，带指数级回退重试。

    Args:
        tool_call: 模型返回的 tool_call 字典，包含 name, args, id 等字段。

    Returns:
        ToolMessage 实例，可直接追加到消息列表。
    """
    name = tool_call["name"]
    tool_fn = tool_map.get(name)
    if tool_fn is None:
        return ToolMessage(
            content=json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False),
            tool_call_id=tool_call["id"],
            name=name,
        )

    last_error = None
    for attempt in range(settings.tool_call_max_retries):
        try:
            # tool.invoke(tool_call) 返回 ToolMessage
            result = tool_fn.invoke(tool_call)
            log.info(f"工具调用完成: {name} → {result.content[:100]}...")
            return result
        except Exception as e:
            last_error = e
            if attempt < settings.tool_call_max_retries - 1:
                delay = settings.tool_call_base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                log.warning(f"工具调用失败 (attempt {attempt + 1}/{settings.tool_call_max_retries}), "
                           f"{delay:.1f}s 后重试: {e}")
                time.sleep(delay)

    log.error(f"工具调用最终失败 (name={name}): {last_error}")
    return ToolMessage(
        content=json.dumps({"error": "服务暂时不可用，请稍后重试"}, ensure_ascii=False),
        tool_call_id=tool_call["id"],
        name=name,
    )


