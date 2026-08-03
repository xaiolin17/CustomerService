from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # 阿里云 API 配置
    ali_api_key: str = "sk-88bca0a94cfe4f48a735f2abddc8d1e6"
    ali_openai_compatible_endpoint: str = "https://ws-8o9kidft5npwykil.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_base"
    milvus_dimension: int = 1024  # qwen3.7-text-embedding 输出维度


    # 文档处理配置
    docs_dir: Path = Path(__file__).parent / "docs"
    chunk_size: int = 200
    chunk_overlap: int = 50

    # Embedding 模型
    embedding_model_name: str = "qwen3.7-text-embedding"

    # 日志（与 CustomerServiceServer 日志分离）
    log_dir: str = "logs/data_handle"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()