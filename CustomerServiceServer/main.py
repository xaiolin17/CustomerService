import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from config import settings
from logger import log
from chain import process_chat, milvus_client
from memory import memory_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    log.info(f"智能客服服务启动: {settings.app_name}")
    log.info(f"配置: Milvus({settings.milvus_host}:{settings.milvus_port}), "
             f"top_k={settings.top_k}, rerank_top_n={settings.rerank_top_n}, "
             f"memory_ttl={settings.memory_ttl_minutes}min, max_turns={settings.memory_max_turns}")
    log.info(f"LLM: 主模型={settings.llm_primary_model}, 备用模型={settings.llm_fallback_model}")
    log.info(f"日志目录: {settings.log_dir}")
    yield
    # 清理资源
    milvus_client.close()
    memory_manager.clear_all()
    log.info("智能客服服务已关闭")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "app": settings.app_name}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 聊天端点"""
    await websocket.accept()

    # 从查询参数获取 session_id，如果没有则生成新的
    session_id = websocket.query_params.get("session_id", str(uuid.uuid4()))
    log.info(f"WebSocket 连接建立: session_id={session_id[:8]}...")

    try:
        # 发送欢迎消息
        await websocket.send_json({
            "type": "welcome",
            "content": "您好！请问有什么可以帮助您的？",
            "session_id": session_id
        })

        while True:
            # 接收消息
            raw_data = await websocket.receive_text()

            try:
                data = json.loads(raw_data)
                msg_type = data.get("type", "")

                if msg_type == "chat":
                    user_content = data.get("content", "").strip()
                    if not user_content:
                        continue

                    # 处理聊天
                    result = process_chat(session_id, user_content)

                    # 发送回复
                    await websocket.send_json({
                        "type": "reply",
                        "content": result["content"],
                        "sources": result.get("sources")
                    })

                elif msg_type == "clear":
                    # 清除记忆
                    memory_manager.clear_memory(session_id)
                    await websocket.send_json({
                        "type": "info",
                        "content": "对话记忆已清除"
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"未知消息类型: {msg_type}"
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "content": "消息格式错误，请发送 JSON 格式消息"
                })

    except WebSocketDisconnect:
        log.info(f"WebSocket 连接断开: session_id={session_id[:8]}...")
    except Exception as e:
        log.error(f"WebSocket 异常: {e}")
    finally:
        # 不清理记忆，保持短期记忆以便下次连接
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )