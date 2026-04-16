"""Agentic RAG entrypoint with hybrid retrieval and reranking."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import httpx
from langchain_community.embeddings import HuggingFaceEmbeddings
from smolagents import OpenAIServerModel, ToolCallingAgent
import torch

from utils.orchestration import (
    AGENT_EXECUTION_INSTRUCTIONS,
    ConversationMemory,
    ReflectiveConversationAgent,
)
from utils.processing import discover_knowledge_sources
from utils.retrieval import (
    AdvancedRetriever,
    IndexBuildConfig,
    SemanticRetriever,
    load_or_build_retrieval_indexes,
)
from utils.runtime import configure_runtime_environment
from utils.tools import (
    LibrarySqlConfig,
    LibrarySqlTool,
    ProxyAwareWebSearchTool,
    WebSearchProxyConfig,
)


PROJECT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
configure_runtime_environment(PROJECT_DIR)


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the agentic RAG application."""

    llm_api_key: str
    llm_base_url: str
    llm_model_id: str
    llm_use_env_proxy: bool
    llm_timeout_sec: int
    web_search_use_env_proxy: bool
    web_search_http_proxy: str
    web_search_https_proxy: str
    web_search_all_proxy: str
    web_search_no_proxy: str
    embedding_model_path: str
    embedding_device: str
    embedding_batch_size: int
    data_path: str
    index_dir: str
    tmp_path: str
    subset: int
    chunk_size: int
    chunk_overlap: int
    title_token_boost: int
    ocr_enabled: bool
    faiss_build_batch_size: int
    force_rebuild_index: bool
    alpha: float
    vector_top_k: int
    bm25_top_k: int
    hybrid_top_k: int
    final_top_k: int
    rrf_k: int
    reranker_model_name: str
    rerank_enabled: bool
    rerank_cache_size: int
    rerank_batch_size: int
    library_db_url: str
    library_books_table: str
    library_inventory_table: str
    library_book_id_column: str
    library_inventory_book_id_column: str
    library_title_column: str
    library_isbn_column: str
    library_available_count_column: str
    library_total_count_column: str
    library_shelf_location_column: str
    library_status_column: str
    log_dir: str
    log_level: str


def _resolve_path(raw_path: str) -> Path:
    """Resolves a potentially relative path from the project root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    # 配置文件里允许写相对路径，这里统一转成相对于项目根目录的绝对路径。
    return (PROJECT_DIR / path).resolve()


def _env_bool(name: str, default: bool) -> bool:
    """Reads a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_or_fallback(name: str, fallback_name: str) -> str:
    """读取环境变量；若值为空字符串，则继续回退到另一个环境变量。"""
    primary = os.getenv(name)
    if primary is not None and primary.strip():
        return primary.strip()
    fallback = os.getenv(fallback_name, "")
    return fallback.strip()


def load_config() -> AppConfig:
    """Loads runtime configuration from the local `.env` file and environment."""
    # 启动时优先读取项目内的 .env，避免依赖外部 shell 环境。
    load_dotenv(PROJECT_DIR / ".env")
    return AppConfig(
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
        llm_model_id=os.getenv("LLM_MODEL_ID", "qwen-plus").strip(),
        # 默认不继承系统代理，避免错误代理导致 DashScope/OpenAI 连接被拒。
        llm_use_env_proxy=_env_bool("LLM_USE_ENV_PROXY", False),
        llm_timeout_sec=int(os.getenv("LLM_TIMEOUT_SEC", "120")),
        # WebSearchTool 与 DashScope 分离配置，默认允许它单独走代理。
        web_search_use_env_proxy=_env_bool("WEB_SEARCH_USE_ENV_PROXY", True),
        web_search_http_proxy=_env_or_fallback("WEB_SEARCH_HTTP_PROXY", "http_proxy"),
        web_search_https_proxy=_env_or_fallback("WEB_SEARCH_HTTPS_PROXY", "https_proxy"),
        web_search_all_proxy=_env_or_fallback("WEB_SEARCH_ALL_PROXY", "all_proxy"),
        web_search_no_proxy=os.getenv("WEB_SEARCH_NO_PROXY", "127.0.0.1,localhost").strip(),
        embedding_model_path=str(_resolve_path(os.getenv("EMBEDDING_MODEL_PATH", "./data/gte-small-zh"))),
        # embedding 默认优先走 GPU；如果用户显式指定，就按配置执行。
        embedding_device=os.getenv(
            "EMBEDDING_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        ).strip(),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "128")),
        data_path=str(_resolve_path(os.getenv("DATA_PATH", "./data/rag"))),
        index_dir=str(_resolve_path(os.getenv("INDEX_DIR", "./data/faiss_rag"))),
        tmp_path=str(_resolve_path(os.getenv("TMP_PATH", "./temp"))),
        subset=int(os.getenv("SUBSET", "-1")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "20")),
        title_token_boost=int(os.getenv("TITLE_TOKEN_BOOST", "3")),
        # 处理扫描版 PDF 时启用 OCR。
        ocr_enabled=_env_bool("OCR_ENABLED", False),
        # FAISS 增量构建批大小：越小越省内存，但构建时间会更长。
        faiss_build_batch_size=int(os.getenv("FAISS_BUILD_BATCH_SIZE", "512")),
        # 设为 true 时会无视已有缓存，强制从原始语料完整重建一次双索引。
        force_rebuild_index=_env_bool("FORCE_REBUILD_INDEX", False),
        alpha=float(os.getenv("HYBRID_ALPHA", "1.25")),
        vector_top_k=int(os.getenv("VECTOR_TOP_K", "20")),
        bm25_top_k=int(os.getenv("BM25_TOP_K", "20")),
        hybrid_top_k=int(os.getenv("HYBRID_TOP_K", "20")),
        final_top_k=int(os.getenv("FINAL_TOP_K", "5")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        reranker_model_name=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3").strip(),
        rerank_enabled=_env_bool("ENABLE_RERANK", True),
        rerank_cache_size=int(os.getenv("RERANK_CACHE_SIZE", "4096")),
        rerank_batch_size=int(os.getenv("RERANK_BATCH_SIZE", "8")),
        library_db_url=os.getenv("LIBRARY_DB_URL", "").strip(),
        library_books_table=os.getenv("LIBRARY_BOOKS_TABLE", "books").strip(),
        library_inventory_table=os.getenv("LIBRARY_INVENTORY_TABLE", "inventory").strip(),
        library_book_id_column=os.getenv("LIBRARY_BOOK_ID_COLUMN", "id").strip(),
        library_inventory_book_id_column=os.getenv("LIBRARY_INVENTORY_BOOK_ID_COLUMN", "book_id").strip(),
        library_title_column=os.getenv("LIBRARY_TITLE_COLUMN", "title").strip(),
        library_isbn_column=os.getenv("LIBRARY_ISBN_COLUMN", "isbn").strip(),
        library_available_count_column=os.getenv("LIBRARY_AVAILABLE_COUNT_COLUMN", "available_count").strip(),
        library_total_count_column=os.getenv("LIBRARY_TOTAL_COUNT_COLUMN", "total_count").strip(),
        library_shelf_location_column=os.getenv("LIBRARY_SHELF_LOCATION_COLUMN", "shelf_location").strip(),
        library_status_column=os.getenv("LIBRARY_STATUS_COLUMN", "status").strip(),
        log_dir=str(_resolve_path(os.getenv("LOG_DIR", "./logs/rag"))),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def configure_logging(level: str, log_dir: str) -> None:
    """Configures application logging."""
    # 检索链路包含多阶段召回和精排，统一同时写终端和文件，便于离线排查。
    resolved_log_dir = Path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_log_dir / "agentic_rag.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level, logging.INFO))
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def validate_config(config: AppConfig) -> None:
    """Validates required runtime settings."""
    missing = []
    if not config.llm_api_key:
        missing.append("LLM_API_KEY")
    if not config.llm_base_url:
        missing.append("LLM_BASE_URL")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def discover_data_files(data_path: str) -> list[str]:
    """兼容旧调用方，返回知识库目录中的可索引文件。"""
    source_discovery = discover_knowledge_sources(data_path)
    return list(source_discovery.retrieval_files)


def build_retriever(config: AppConfig) -> AdvancedRetriever:
    """Builds the hybrid retriever stack."""
    source_discovery = discover_knowledge_sources(config.data_path)
    embedding_device = config.embedding_device
    if embedding_device.startswith("cuda") and not torch.cuda.is_available():
        # 用户要求走 CUDA 但当前进程看不到显卡时，明确告警并回退，避免误以为还在用 GPU。
        LOGGER.warning(
            "CUDA was requested for embeddings, but torch.cuda.is_available() is False. Falling back to CPU."
        )
        embedding_device = "cpu"

    # 这里显式指定 embedding 设备，避免 sentence-transformers 默认回退到 CPU。
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.embedding_model_path,
        model_kwargs={
            "device": embedding_device,
        },
        encode_kwargs={
            # 批大小做成配置项，方便根据显存大小微调索引构建速度。
            "batch_size": config.embedding_batch_size,
            "normalize_embeddings": True,
        },
    )
    LOGGER.info(
        "Embedding backend model=%s device=%s batch_size=%s",
        config.embedding_model_path,
        embedding_device,
        config.embedding_batch_size,
    )
    build_config = IndexBuildConfig(
        embed_model_name=config.embedding_model_path,
        data_files=tuple(source_discovery.retrieval_files),
        index_dir=config.index_dir,
        tmp_path=config.tmp_path,
        subset=config.subset,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        title_token_boost=config.title_token_boost,
        ocr_enabled=config.ocr_enabled,
    )
    # 这里会负责“加载已有索引”或“首次构建 FAISS + BM25 双路索引”。
    vectordb, bm25_index, bm25_documents = load_or_build_retrieval_indexes(
        config=build_config,
        embedding_model=embedding_model,
        force_rebuild=config.force_rebuild_index,
        faiss_build_batch_size=config.faiss_build_batch_size,
    )
    return AdvancedRetriever(
        vectordb=vectordb,
        bm25_index=bm25_index,
        bm25_documents=bm25_documents,
        alpha=config.alpha,
        vector_top_k=config.vector_top_k,
        bm25_top_k=config.bm25_top_k,
        hybrid_top_k=config.hybrid_top_k,
        final_top_k=config.final_top_k,
        rrf_k=config.rrf_k,
        reranker_model_name=config.reranker_model_name,
        rerank_enabled=config.rerank_enabled,
        rerank_cache_size=config.rerank_cache_size,
        rerank_batch_size=config.rerank_batch_size,
    )


def build_library_sql_tool(config: AppConfig) -> LibrarySqlTool | None:
    """按配置构建图书馆库存数据库工具。"""
    if not config.library_db_url:
        return None

    sql_config = LibrarySqlConfig(
        db_url=config.library_db_url,
        books_table=config.library_books_table,
        inventory_table=config.library_inventory_table,
        book_id_column=config.library_book_id_column,
        inventory_book_id_column=config.library_inventory_book_id_column,
        title_column=config.library_title_column,
        isbn_column=config.library_isbn_column,
        available_count_column=config.library_available_count_column,
        total_count_column=config.library_total_count_column,
        shelf_location_column=config.library_shelf_location_column,
        status_column=config.library_status_column,
    )
    return LibrarySqlTool(sql_config)


def build_agent(config: AppConfig) -> ReflectiveConversationAgent:
    """Builds the reflective conversation agent."""
    http_client = httpx.Client(
        trust_env=config.llm_use_env_proxy,
        timeout=httpx.Timeout(config.llm_timeout_sec),
    )
    model = OpenAIServerModel(
        model_id=config.llm_model_id,
        api_base=config.llm_base_url,
        api_key=config.llm_api_key,
        flatten_messages_as_text=True,
        client_kwargs={
            "http_client": http_client,
        },
    )
    # Agent 现在拿到的是增强版检索工具：支持混合召回和精排。
    retriever_tool = SemanticRetriever(build_retriever(config))
    web_search_tool = ProxyAwareWebSearchTool(
        proxy_config=WebSearchProxyConfig(
            use_env_proxy=config.web_search_use_env_proxy,
            http_proxy=config.web_search_http_proxy,
            https_proxy=config.web_search_https_proxy,
            all_proxy=config.web_search_all_proxy,
            no_proxy=config.web_search_no_proxy,
        )
    )
    tools = [retriever_tool, web_search_tool]
    sql_tool = build_library_sql_tool(config)
    if sql_tool is not None:
        tools.append(sql_tool)
    tool_calling_agent = ToolCallingAgent(
        tools=tools,
        model=model,
        instructions=AGENT_EXECUTION_INSTRUCTIONS,
    )
    return ReflectiveConversationAgent(
        model=model,
        agent=tool_calling_agent,
        retriever_tool=retriever_tool,
        memory=ConversationMemory(model=model),
    )


def interactive_loop(agent: ReflectiveConversationAgent) -> None:
    """Runs the CLI query loop."""
    while True:
        try:
            query = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not query:
            continue

        # 保持原来的交互式体验，但底层已经换成新的检索管线。
        response = agent.run(query)
        print("Response:", response)


def main() -> None:
    """Runs the application."""
    config = load_config()
    configure_logging(config.log_level, config.log_dir)
    validate_config(config)
    # 启动日志里打印关键路径和 reranker 开关，便于确认当前运行配置。
    LOGGER.info(
        "Starting agentic RAG with data_path=%s index_dir=%s embed_device=%s ocr_enabled=%s llm_use_env_proxy=%s web_search_use_env_proxy=%s library_db_enabled=%s reranker=%s enabled=%s log_dir=%s",
        config.data_path,
        config.index_dir,
        config.embedding_device,
        config.ocr_enabled,
        config.llm_use_env_proxy,
        config.web_search_use_env_proxy,
        bool(config.library_db_url),
        config.reranker_model_name,
        config.rerank_enabled,
        config.log_dir,
    )
    agent = build_agent(config)
    interactive_loop(agent)


if __name__ == "__main__":
    main()
