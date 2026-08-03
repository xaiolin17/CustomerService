"""
数据预处理模块：将文档离线向量化并存入 Milvus。
使用 qwen3.7-text-embedding（通过阿里云 DashScope API）生成向量。

用法:
    python ingest.py                          # 使用默认配置
    python ingest.py --docs-dir /path/to/docs # 指定文档目录
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
from loguru import logger
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# LangChain 1.2.x 社区文档加载器
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from config import settings

# ---------------------------------------------------------------------------
# 日志配置（独立于 CustomerServiceServer，输出到 logs/data_handle/）
# ---------------------------------------------------------------------------
logger.remove()
# 控制台输出
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level=settings.log_level,
    colorize=True,
)
# 文件输出（按天滚动，保留30天，独立目录）
log_path = Path(settings.log_dir)
log_path.mkdir(parents=True, exist_ok=True)
logger.add(
    str(log_path / "data_handle_{time:YYYY-MM-DD}.log"),
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level=settings.log_level,
    compression="gz",
)


# ---------------------------------------------------------------------------
# 文档加载（使用 LangChain 1.2.x 社区加载器）
# ---------------------------------------------------------------------------

# 文件扩展名 → (LangChain Loader 类, 加载参数字典)
LOADER_MAP: dict[str, tuple] = {
    ".txt":   (TextLoader,                   {"encoding": "utf-8"}),
    ".md":    (UnstructuredMarkdownLoader,   {"mode": "single", "strategy": "fast"}),
    ".pdf":   (PyPDFLoader,                  {"extraction_mode": "plain"}),
    ".docx":  (UnstructuredWordDocumentLoader, {"mode": "single"}),
    ".csv":   (CSVLoader,                    {}),
    ".json":  (JSONLoader,                   {"jq_schema": ".", "text_content": False}),
    ".html":  (UnstructuredHTMLLoader,       {"mode": "single", "strategy": "fast"}),
}

SUPPORTED_EXTENSIONS = set(LOADER_MAP.keys())


def _load_single_file(file_path: Path) -> str:
    """使用 LangChain 加载器读取单个文件，返回合并后的文本内容。

    Args:
        file_path: 文件路径。

    Returns:
        文件文本内容。如果加载器返回多个 Document，用双换行合并。
    """
    ext = file_path.suffix.lower()
    loader_cls, loader_kwargs = LOADER_MAP.get(ext)

    if loader_cls is None:
        raise ValueError(f"不支持的文件格式: {ext}")

    # 构造 loader 实例
    loader = loader_cls(file_path=str(file_path), **loader_kwargs)
    docs = loader.load()

    # 合并所有 Document 的 page_content
    texts = [doc.page_content for doc in docs if doc.page_content.strip()]
    return "\n\n".join(texts)


def load_documents(docs_dir: Path) -> List[Tuple[str, str]]:
    """加载目录下所有支持的文档。

    Args:
        docs_dir: 文档目录路径。

    Returns:
        List[Tuple[str, str]]: (文件名, 文本内容) 列表。

    Raises:
        FileNotFoundError: 目录不存在时抛出。
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    documents: List[Tuple[str, str]] = []

    files = [
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        logger.warning("目录 {} 下没有找到支持的文档（支持格式: {}）", docs_dir, supported_str)
        return documents

    logger.info("发现 {} 个待处理文件", len(files))

    for file_path in files:
        try:
            text = _load_single_file(file_path)
            if text.strip():
                documents.append((file_path.name, text))
                logger.debug("已加载文件: {} （{} 字符）", file_path.name, len(text))
            else:
                logger.warning("文件内容为空，跳过: {}", file_path.name)
        except Exception as e:
            logger.error("读取文件失败 {}: {}", file_path.name, e)

    return documents


# ---------------------------------------------------------------------------
# 文本切分
# ---------------------------------------------------------------------------
def split_texts(
    documents: List[Tuple[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Tuple[str, str, int]]:
    """将文档切分为固定大小的文本块。

    Args:
        documents: (文件名, 文本内容) 列表。
        chunk_size: 每个块的最大字符数。
        chunk_overlap: 块之间的重叠字符数。

    Returns:
        List[Tuple[str, str, int]]: (源文件名, 块文本, 块序号) 列表。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "……", "！", "？", "；", "，", " ", ""],
    )

    chunks: List[Tuple[str, str, int]] = []
    for file_name, text in documents:
        texts = splitter.split_text(text)
        for idx, chunk in enumerate(texts):
            chunks.append((file_name, chunk, idx))

    logger.info("文本切分完成: {} 个文档 → {} 个文本块", len(documents), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Milvus 操作
# ---------------------------------------------------------------------------
def create_collection_if_not_exists(collection_name: str, dimension: int) -> Collection:
    """检查集合是否存在，不存在则创建。

    Args:
        collection_name: 集合名称。
        dimension: 向量维度。

    Returns:
        Collection 实例。
    """
    if utility.has_collection(collection_name):
        logger.info("集合 '{}' 已存在，直接使用", collection_name)
        return Collection(collection_name)

    logger.info("创建集合 '{}'（维度: {}）", collection_name, dimension)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    ]

    schema = CollectionSchema(fields=fields, description="知识库文档向量")
    collection = Collection(name=collection_name, schema=schema)

    # 创建 IVF_FLAT 索引以加速检索
    index_params = {
        "metric_type": "IP",  # 内积，适合归一化后的向量
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info("集合 '{}' 创建完成，索引类型: IVF_FLAT", collection_name)

    return collection


def insert_vectors(
    collection: Collection,
    chunks: List[Tuple[str, str, int]],
    embedding_client: OpenAIEmbeddings,
) -> int:
    """通过 qwen3.7-text-embedding API 生成向量并插入 Milvus。

    Args:
        collection: Milvus Collection 实例。
        chunks: (源文件名, 块文本, 块序号) 列表。
        embedding_client: OpenAIEmbeddings 客户端。

    Returns:
        插入的向量数量。
    """
    if not chunks:
        logger.warning("没有数据可插入")
        return 0

    texts = [chunk[1] for chunk in chunks]
    logger.info("开始生成 embedding，共 {} 条文本……", len(texts))

    _batch_size = 20  # 同时满足嵌入 API 和 Milvus 限制
    inserted_count = 0

    if not chunks:
        logger.warning("警告：chunks 为空，无需处理。")
    else:
        total = len(chunks)

        for i in range(0, total, _batch_size):
            # 当前批次切片
            batch_chunks = chunks[i:i + _batch_size]
            batch_file_names = [c[0] for c in batch_chunks]
            batch_chunk_indices = [c[2] for c in batch_chunks]
            batch_texts = [c[1] for c in batch_chunks]

            try:
                # 1. 生成当前批次的 embedding
                batch_embeddings = embedding_client.embed_documents(batch_texts)

                # 2. 立即插入 Milvus
                entities = [
                    batch_file_names,
                    batch_chunk_indices,
                    batch_texts,
                    batch_embeddings
                ]
                collection.insert(entities)
                inserted_count += len(batch_chunks)
                print(
                    f"已处理批次 {i // _batch_size + 1}/{(total - 1) // _batch_size + 1}，成功插入 {len(batch_chunks)} 条")

            except Exception as e:
                print(f"批次 {i // _batch_size + 1} 处理失败：{e}")
                # 可选：记录失败批次信息以便重试

        # 所有批次完成后统一 flush（若需实时可见，可在每批后 flush）
        collection.flush()
        print(f"全部完成，共成功插入 {inserted_count}/{total} 条数据。")

    logger.info("向量入库完成: {} 条", inserted_count)
    return inserted_count


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def ingest(docs_dir: Path | None = None) -> None:
    """主流程：加载文档 → 切分 → 向量化 → 入库。

    Args:
        docs_dir: 文档目录路径，为 None 时使用配置默认值。
    """
    target_dir = docs_dir or settings.docs_dir

    # ── 第一步：加载文档 ──────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("开始数据预处理流程")
    logger.info("文档目录: {}", target_dir)
    logger.info("Milvus 目标: {}:{} / {}", settings.milvus_host, settings.milvus_port, settings.milvus_collection)
    logger.info("Embedding 模型: {}", settings.embedding_model_name)
    logger.info("=" * 56)

    try:
        documents = load_documents(target_dir)
    except FileNotFoundError as e:
        logger.error("{}", e)
        logger.error("请确保文档目录存在，或使用 --docs-dir 指定正确的路径")
        sys.exit(1)

    if not documents:
        logger.warning("没有加载到任何文档，程序退出")
        return

    # ── 第二步：文本切分 ──────────────────────────────────────────────
    chunks = split_texts(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # ── 第三步：连接 Milvus ───────────────────────────────────────────
    logger.info("连接 Milvus {}:{}……", settings.milvus_host, settings.milvus_port)
    try:
        connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    except Exception as e:
        logger.error("Milvus 连接失败: {}", e)
        logger.error(
            "请确认 Milvus 服务已启动: host={} port={}",
            settings.milvus_host,
            settings.milvus_port,
        )
        sys.exit(1)

    # ── 第四步：创建集合 ──────────────────────────────────────────────
    collection = create_collection_if_not_exists(
        settings.milvus_collection,
        settings.milvus_dimension
    )

    # ── 第五步：初始化 embedding 客户端 ──────────────────────────────
    logger.info("初始化 embedding 客户端: {}……", settings.embedding_model_name)
    try:
        # ['azure_openai', 'bedrock', 'cohere', 'google_genai',
        # 'google_vertexai', 'huggingface', 'mistralai', 'ollama', 'openai']
        # from langchain_community.embeddings import DashScopeEmbeddings
        # embedding_client = DashScopeEmbeddings(
        #     model=settings.embedding_model_name,
        #     dashscope_api_key=settings.ali_api_key,
        #     # base_url=settings.ali_openai_compatible_endpoint,
        # )
        # qianwen某些模型不适用
        # embedding_client = init_embeddings(
        #     model=settings.embedding_model_name,
        #     provider="bedrock",
        #     api_key=settings.ali_api_key,
        #     base_url=settings.ali_openai_compatible_endpoint,
        # )
        embedding_client = OpenAIEmbeddings(
            api_key=settings.ali_api_key,
            base_url=settings.ali_openai_compatible_endpoint,
            model=settings.embedding_model_name,
            # 关键参数：必须设置为 False，否则可能报错
            check_embedding_ctx_length=False,
        )
    except Exception as e:
        logger.error("Embedding 客户端初始化失败: {}", e)
        connections.disconnect(settings.milvus_host)
        sys.exit(1)

    # ── 第六步：向量化并入库 ──────────────────────────────────────────
    try:
        inserted = insert_vectors(collection, chunks, embedding_client)
    except Exception as e:
        logger.error("向量入库失败: {}", e)
        sys.exit(1)
    finally:
        connections.disconnect(settings.milvus_host)

    # ── 汇总 ──────────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("预处理完成")
    logger.info("  文件数: {}", len(documents))
    logger.info("  切分块数: {}", len(chunks))
    logger.info("  入库向量数: {}", inserted)
    logger.info("=" * 56)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="数据预处理：将文档离线向量化并存入 Milvus",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="文档目录路径（覆盖配置中的默认值）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    docs_dir = Path(args.docs_dir).resolve() if args.docs_dir else None
    ingest(docs_dir)