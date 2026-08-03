from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 服务配置
    app_name: str = "智能客服"
    host: str = "0.0.0.0"
    port: int = 8000

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_base"
    milvus_dimension: int = 1024  # qwen3.7-text-embedding 输出维度

    # 阿里云通义千问配置（OpenAI 兼容模式）
    ali_api_key: str = "sk-88bca0a94cfe4f48a735f2abddc8d1e6"
    ali_openai_compatible_endpoint: str = "https://ws-8o9kidft5npwykil.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    ali_dashscope_endpoint: str = "https://ws-8o9kidft5npwykil.cn-beijing.maas.aliyuncs.com/api/v1"

    # 主模型与备用模型
    llm_primary_model: str = "qwen3.7-max"
    llm_fallback_model: str = "qwen3.7-flash"
    use_mock_llm: bool = False

    # 向量模型
    embedding_model_name: str = "qwen3.7-text-embedding"

    # 检索参数
    top_k: int = 5
    rerank_top_n: int = 3
    similarity_threshold: float = 0.3

    # 短期记忆（30分钟过期）
    memory_ttl_minutes: int = 30
    memory_max_turns: int = 20

    # 上下文管理（512KB token 上限）
    context_max_tokens: int = 512_000
    context_keep_last_messages: int = 3          # 压缩时保留最近 N 条消息
    context_compress_ratio: float = 0.8          # 达到 80% 阈值时触发压缩

    # 工具调用指数级回退
    tool_call_max_retries: int = 3
    tool_call_base_delay: float = 1.0            # 初始延迟（秒）

    # 日志路径
    log_dir: str = "logs/server"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()