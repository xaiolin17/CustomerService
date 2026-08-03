"""
Milvus 连接管理 - 使用 qwen3.7-text-embedding 通过 API 获取向量。
"""

from pymilvus import connections, Collection, utility
from langchain_openai import OpenAIEmbeddings

from config import settings
from logger import log


class MilvusClient:
    """Milvus 连接管理 - 单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.embedding_model = None
        self._connect()

    def _connect(self):
        """连接 Milvus 并初始化 embedding 客户端"""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            log.info(f"Milvus 连接成功: {settings.milvus_host}:{settings.milvus_port}")

            # 初始化 qwen3.7-text-embedding 客户端（通过 OpenAI 兼容 API）
            self.embedding_model = OpenAIEmbeddings(
                model=settings.embedding_model_name,
                api_key=settings.ali_api_key,
                base_url=settings.ali_openai_compatible_endpoint,
            )
            # 测试调用一次验证可用性
            test_vec = self.embedding_model.embed_query("测试")
            log.info(f"Embedding 模型初始化成功: {settings.embedding_model_name}, "
                     f"维度={len(test_vec)}")
        except Exception as e:
            log.error(f"Milvus 连接失败: {e}")
            log.warning("服务将以降级模式运行（无 RAG 检索能力）")
            self.embedding_model = None

    def search(self, query: str) -> list:
        """从 Milvus 检索相关文档"""
        if not utility.has_collection(settings.milvus_collection):
            log.warning(f"集合 {settings.milvus_collection} 不存在")
            return []

        if self.embedding_model is None:
            log.warning("Embedding 模型未初始化，无法检索")
            return []

        try:
            # 生成查询向量
            query_vector = self.embedding_model.embed_query(query)

            # 检索
            collection = Collection(settings.milvus_collection)
            collection.load()

            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=settings.top_k,
                output_fields=["file_name", "text"]
            )

            # 处理结果
            documents = []
            for hits in results:
                for hit in hits:
                    if hit.score >= settings.similarity_threshold:
                        documents.append({
                            "text": hit.entity.get("text"),
                            "file_name": hit.entity.get("file_name", ""),
                            "score": hit.score
                        })

            log.info(f"Milvus 检索完成: query=\"{query[:30]}...\", 结果数={len(documents)}")

            # 重排序（按 score 降序）
            documents.sort(key=lambda x: x["score"], reverse=True)
            documents = documents[:settings.rerank_top_n]

            return documents
        except Exception as e:
            log.error(f"Milvus 检索失败: {e}")
            return []

    def close(self):
        """关闭连接"""
        try:
            connections.disconnect("default")
            log.info("Milvus 连接已关闭")
        except Exception:
            pass