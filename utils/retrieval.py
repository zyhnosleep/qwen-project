"""Retrieval utilities for hybrid search and reranking."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
import gc
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import datasets
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.processing import load_knowledge_documents

LOGGER = logging.getLogger(__name__)
INDEX_SCHEMA_VERSION = 2
FAISS_SUBDIR = "faiss"
BM25_FILENAME = "bm25.pkl"
MANIFEST_FILENAME = "manifest.json"
LEGACY_FAISS_FILES = ("index.faiss", "index.pkl")
BUILD_STAGE_BM25_READY = "bm25_ready"
BUILD_STAGE_COMPLETE = "complete"


@dataclass(frozen=True)
class IndexBuildConfig:
    """Stores the retrieval index build configuration."""

    embed_model_name: str
    data_files: tuple[str, ...]
    index_dir: str
    tmp_path: str
    subset: int
    chunk_size: int
    chunk_overlap: int
    title_token_boost: int
    ocr_enabled: bool


@dataclass(frozen=True)
class QueryProfile:
    """描述一次查询的关键词倾向，便于动态调节混合检索权重。"""

    raw_query: str
    normalized_query: str
    phrase_query: str
    query_terms: tuple[str, ...]
    identifier_terms: tuple[str, ...]
    contains_identifier: bool
    has_delimited_phrase: bool
    is_short_query: bool
    looks_like_title_lookup: bool
    effective_alpha: float


@dataclass
class ScoredChunk:
    """Represents a retrieved chunk and all retrieval scores."""

    document: Document
    hybrid_score: float = 0.0
    final_score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    hybrid_rank: int | None = None
    rerank_rank: int | None = None

    @property
    def chunk_id(self) -> str:
        """Returns the stable chunk identifier."""
        return str(self.document.metadata.get("chunk_id", ""))

    def to_payload(self, rank: int) -> dict[str, Any]:
        """Serializes the scored chunk for tool output."""
        metadata = dict(self.document.metadata)
        return {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "content": self.document.page_content,
            "relevance_score": round(float(self.final_score), 6),
            "scores": {
                "hybrid": round(float(self.hybrid_score), 6),
                "vector": None if self.vector_score is None else round(float(self.vector_score), 6),
                "bm25": None if self.bm25_score is None else round(float(self.bm25_score), 6),
                "rerank": None if self.rerank_score is None else round(float(self.rerank_score), 6),
            },
            "ranks": {
                "vector": self.vector_rank,
                "bm25": self.bm25_rank,
                "hybrid": self.hybrid_rank,
                "rerank": self.rerank_rank,
            },
            "source": {
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "source": metadata.get("source"),
                "source_file": metadata.get("source_file"),
                "source_type": metadata.get("source_type"),
                "record_id": metadata.get("record_id"),
                "chunk_index": metadata.get("chunk_index"),
                "start_index": metadata.get("start_index"),
                "page_number": metadata.get("page_number"),
                "block_type": metadata.get("block_type"),
                "citation_hint": metadata.get("citation_hint"),
                "citation_text": _build_citation_text(metadata),
                "image_caption": metadata.get("image_caption"),
                "tags": metadata.get("tags", []),
            },
        }


class ScoreCache:
    """A tiny LRU cache for reranker query-document scores."""

    def __init__(self, max_size: int = 4096) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, float] = OrderedDict()

    def get(self, key: str) -> float | None:
        """Returns a cached score if present."""
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: str, value: float) -> None:
        """Stores a cached score."""
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._max_size:
            self._data.popitem(last=False)


def _load_jieba() -> Any:
    try:
        import jieba
    except ImportError as exc:
        raise ImportError(
            "Hybrid retrieval requires `jieba`. Install it with `pip install jieba`."
        ) from exc
    return jieba


def _load_bm25_class() -> Any:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "Hybrid retrieval requires `rank-bm25`. Install it with `pip install rank-bm25`."
        ) from exc
    return BM25Okapi


def _load_flag_reranker() -> Any:
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise ImportError(
            "Reranking requires `FlagEmbedding`. Install it with `pip install FlagEmbedding`."
        ) from exc
    return FlagReranker


def _resolve_local_hf_snapshot(model_name_or_path: str) -> str:
    """优先把 Hugging Face 仓库名解析成本地快照目录，避免运行时联网探测。"""
    raw_path = Path(model_name_or_path)
    if raw_path.exists():
        return str(raw_path)

    # 仓库名形如 `org/repo` 时，尝试定位本地 huggingface hub 缓存。
    if "/" not in model_name_or_path:
        return model_name_or_path

    org_name, repo_name = model_name_or_path.split("/", 1)
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_cache_dir = cache_root / f"models--{org_name}--{repo_name}" / "snapshots"
    if not model_cache_dir.exists():
        return model_name_or_path

    # 选择最新一个快照目录作为本地模型路径。
    snapshot_dirs = sorted(
        [path for path in model_cache_dir.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not snapshot_dirs:
        return model_name_or_path
    return str(snapshot_dirs[0])


def _safe_text(value: Any) -> str:
    """Converts nullable fields into normalized strings."""
    if value is None:
        return ""
    return str(value).strip()


def _collapse_whitespace(text: str) -> str:
    """压缩连续空白，方便做标题匹配与日志展示。"""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_tags(raw_tags: Any) -> list[str]:
    """Normalizes optional tag fields from heterogeneous sources."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        candidates = re.split(r"[,;/|]", raw_tags)
        return [tag.strip() for tag in candidates if tag.strip()]
    if isinstance(raw_tags, Sequence):
        tags: list[str] = []
        for item in raw_tags:
            text = _safe_text(item)
            if text:
                tags.append(text)
        return tags
    return [_safe_text(raw_tags)] if _safe_text(raw_tags) else []


def _build_citation_text(metadata: dict[str, Any]) -> str:
    """Builds a citation-ready source string for downstream answer formatting."""
    title = _safe_text(metadata.get("title")) or _safe_text(metadata.get("source")) or "未命名资料"
    page_number = metadata.get("page_number")
    if isinstance(page_number, int):
        return f"来源于《{title}》第{page_number}页。"
    return f"来源于《{title}》。"


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizes Chinese text for BM25 retrieval."""
    jieba = _load_jieba()
    normalized = _safe_text(text).lower()
    if not normalized:
        return []

    tokens: list[str] = []
    for token in jieba.lcut_for_search(normalized):
        token = token.strip()
        if token:
            tokens.append(token)

    # Preserve exact alphanumeric strings such as ISBNs and model names.
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9._:/-]*", normalized))
    return tokens


def _extract_identifier_terms(text: str) -> tuple[str, ...]:
    """抽取 ISBN、型号、编号类强关键词，给精确检索更高优先级。"""
    normalized = _safe_text(text).lower()
    candidates = re.findall(r"[a-z0-9][a-z0-9._:/-]{5,}", normalized)
    unique_terms = tuple(dict.fromkeys(candidates))
    return unique_terms


def _extract_query_terms(text: str) -> tuple[str, ...]:
    """抽取查询中的中文/英文短语，用于标题和标签的额外加权。"""
    normalized = _collapse_whitespace(_safe_text(text).lower())
    if not normalized:
        return ()

    # 中文连续片段、英文单词、数字串都保留下来，用于标题命中判断。
    candidates = re.findall(r"[\u4e00-\u9fff]+|[a-z]+|\d+", normalized)
    filtered = [item for item in candidates if len(item) >= 2]
    if not filtered and normalized:
        filtered = [normalized]
    return tuple(dict.fromkeys(filtered))


def build_query_profile(query: str, base_alpha: float) -> QueryProfile:
    """分析查询意图，并给出当前查询应使用的关键词权重。"""
    normalized_query = _collapse_whitespace(_safe_text(query))
    phrase_query = normalized_query.strip("\"'“”‘’《》")
    identifier_terms = _extract_identifier_terms(normalized_query)
    query_terms = _extract_query_terms(phrase_query)
    has_delimited_phrase = phrase_query != normalized_query or any(
        mark in normalized_query for mark in ['"', "'", "“", "”", "《", "》"]
    )
    is_short_query = len(phrase_query) <= 18
    looks_like_title_lookup = is_short_query and " " not in phrase_query and "？" not in phrase_query and "?" not in phrase_query

    # 中文图书馆场景里，书名、ISBN、型号查询应更偏向 BM25，因此动态抬高 alpha。
    effective_alpha = base_alpha
    if identifier_terms:
        effective_alpha += 0.45
    if has_delimited_phrase:
        effective_alpha += 0.20
    if looks_like_title_lookup:
        effective_alpha += 0.15
    effective_alpha = min(max(effective_alpha, 1.0), 2.5)

    return QueryProfile(
        raw_query=query,
        normalized_query=normalized_query,
        phrase_query=phrase_query,
        query_terms=query_terms,
        identifier_terms=identifier_terms,
        contains_identifier=bool(identifier_terms),
        has_delimited_phrase=has_delimited_phrase,
        is_short_query=is_short_query,
        looks_like_title_lookup=looks_like_title_lookup,
        effective_alpha=effective_alpha,
    )


def _document_payload(document: Document) -> dict[str, Any]:
    """Serializes a LangChain document into a pickle-safe payload."""
    return {
        "page_content": document.page_content,
        "metadata": dict(document.metadata),
    }


def _document_from_payload(payload: dict[str, Any]) -> Document:
    """Deserializes a LangChain document payload."""
    return Document(
        page_content=payload["page_content"],
        metadata=payload.get("metadata", {}),
    )


def _has_legacy_faiss_index(index_dir: Path) -> bool:
    """判断目录里是否存在旧版单层 FAISS 索引文件。"""
    return all((index_dir / filename).exists() for filename in LEGACY_FAISS_FILES)


def _extract_documents_from_faiss(vectordb: FAISS) -> list[Document]:
    """从已存在的 FAISS docstore 中恢复文档，避免重新读取原始语料再切块。

    这里优先按 FAISS 自身的索引顺序恢复文档，保证迁移出的 BM25 文档列表稳定。
    """
    documents: list[Document] = []
    docstore = vectordb.docstore
    doc_mapping = getattr(docstore, "_dict", {})
    index_to_docstore_id = getattr(vectordb, "index_to_docstore_id", {})

    for _, docstore_id in sorted(index_to_docstore_id.items(), key=lambda item: item[0]):
        document = doc_mapping.get(docstore_id)
        if isinstance(document, Document):
            documents.append(document)

    return documents


def _build_bm25_index(
    documents: Sequence[Document],
    title_token_boost: int,
) -> tuple[Any, list[list[str]]]:
    """基于已有文档构建 BM25 索引与分词语料。"""
    tokenized_corpus = [
        _keyword_tokens_for_document(document, title_token_boost)
        for document in tqdm(documents, desc="Tokenizing BM25 corpus")
    ]
    BM25Okapi = _load_bm25_class()
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, tokenized_corpus


def build_chunked_documents(
    *,
    data_files: Sequence[str],
    embed_model_name: str,
    tmp_path: str,
    subset: int,
    chunk_size: int,
    chunk_overlap: int,
    ocr_enabled: bool = False,
) -> list[Document]:
    """兼容旧接口，实际逻辑转由 processing 模块统一处理。"""
    return load_knowledge_documents(
        source_files=data_files,
        embed_model_name=embed_model_name,
        tmp_path=tmp_path,
        subset=subset,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
    )


def _keyword_tokens_for_document(document: Document, title_token_boost: int) -> list[str]:
    """Builds a BM25 token stream with title-aware weighting."""
    title = _safe_text(document.metadata.get("title"))
    tags = _normalize_tags(document.metadata.get("tags"))
    body = document.page_content

    title_tokens = tokenize_for_bm25(title)
    tag_tokens = tokenize_for_bm25(" ".join(tags))
    body_tokens = tokenize_for_bm25(body)

    # 标题和标签在图书馆检索里通常比正文更能代表“是什么书”，因此做显式 boost。
    weighted_tokens = (
        title_tokens * max(title_token_boost, 1)
        + tag_tokens * max(title_token_boost, 1)
        + body_tokens
    )
    return weighted_tokens or body_tokens


def _save_manifest(manifest_path: Path, config: IndexBuildConfig) -> None:
    """Persists the retrieval build manifest."""
    _write_manifest(manifest_path, config, build_stage=BUILD_STAGE_COMPLETE)


def _write_manifest(
    manifest_path: Path,
    config: IndexBuildConfig,
    *,
    build_stage: str,
) -> None:
    """写入索引构建清单，并标记当前构建阶段。"""
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "build_config": asdict(config),
                "build_stage": build_stage,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Loads the retrieval build manifest."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _normalized_build_config_payload(config: IndexBuildConfig) -> dict[str, Any]:
    """把运行时配置转成和 JSON 清单一致的可比较结构。"""
    # dataclass 里的 tuple 经 JSON 序列化后会变成 list，这里先做一次标准化。
    return json.loads(json.dumps(asdict(config), ensure_ascii=False))


def _manifest_matches(
    manifest_path: Path,
    config: IndexBuildConfig,
    *,
    build_stage: str | None = None,
) -> bool:
    """Checks whether the persisted manifest matches the runtime config."""
    if not manifest_path.exists():
        return False
    try:
        manifest = _load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Failed to read retrieval manifest at %s.", manifest_path)
        return False

    stage_matches = build_stage is None or manifest.get("build_stage", BUILD_STAGE_COMPLETE) == build_stage
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION or not stage_matches:
        return False

    stored_config = dict(manifest.get("build_config", {}))
    current_config = _normalized_build_config_payload(config)
    for key, value in current_config.items():
        if key not in stored_config:
            # 对新增加的配置项做向后兼容：旧 manifest 没写时，只在“默认安全值”下视为兼容。
            if key == "ocr_enabled" and value is False:
                continue
            return False
        if stored_config[key] != value:
            return False
    return True


def _save_bm25_payload(
    bm25_path: Path,
    documents: Sequence[Document],
    tokenized_corpus: Sequence[Sequence[str]],
) -> None:
    """先落盘 BM25 阶段产物，便于后续从中断点恢复。"""
    with bm25_path.open("wb") as handle:
        pickle.dump(
            {
                "documents": [_document_payload(document) for document in documents],
                "tokenized_corpus": list(tokenized_corpus),
            },
            handle,
        )


def _load_bm25_payload(bm25_path: Path) -> dict[str, Any]:
    """读取已落盘的 BM25 结果，用于恢复构建。"""
    with bm25_path.open("rb") as handle:
        return pickle.load(handle)


def _restore_documents_from_bm25_payload(bm25_payload: dict[str, Any]) -> list[Document]:
    """从已保存的 BM25 payload 恢复文档对象。"""
    return [_document_from_payload(item) for item in bm25_payload["documents"]]


def _iter_batches(items: Sequence[Document], batch_size: int) -> Sequence[Sequence[Document]]:
    """按固定批大小切分文档序列。"""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _build_faiss_incrementally(
    *,
    documents: Sequence[Document],
    embedding_model: Any,
    faiss_dir: Path,
    batch_size: int,
) -> FAISS:
    """分批构建 FAISS，降低一次性 embedding 全量文档的内存峰值。"""
    vectordb: FAISS | None = None
    batch_count = math.ceil(len(documents) / batch_size)

    for batch_index, document_batch in enumerate(
        tqdm(_iter_batches(documents, batch_size), total=batch_count, desc="Building FAISS index"),
        start=1,
    ):
        batch_documents = list(document_batch)
        # 首批用 from_documents 初始化索引，后续批次用 add_documents 增量追加。
        if vectordb is None:
            vectordb = FAISS.from_documents(
                documents=batch_documents,
                embedding=embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            vectordb.add_documents(batch_documents)

        LOGGER.info(
            "FAISS incremental build batch=%d/%d batch_size=%d",
            batch_index,
            batch_count,
            len(batch_documents),
        )
        # 每批结束后尽快释放 Python 对象和 CUDA 缓存，避免常驻峰值过高。
        del batch_documents
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if vectordb is None:
        raise ValueError("No documents were available for FAISS incremental build.")

    vectordb.save_local(str(faiss_dir))
    return vectordb


def load_or_build_retrieval_indexes(
    *,
    config: IndexBuildConfig,
    embedding_model: Any,
    force_rebuild: bool = False,
    faiss_build_batch_size: int = 512,
) -> tuple[FAISS, Any, list[Document]]:
    """Loads or rebuilds both FAISS and BM25 indexes.

    Args:
        config: Runtime retrieval build configuration.
        embedding_model: Embedding model instance passed to FAISS.

    Returns:
        A tuple of `(vectordb, bm25_index, bm25_documents)`.
    """
    index_dir = Path(config.index_dir)
    faiss_dir = index_dir / FAISS_SUBDIR
    bm25_path = index_dir / BM25_FILENAME
    manifest_path = index_dir / MANIFEST_FILENAME

    # force_rebuild=true 时，明确跳过所有缓存和旧索引迁移逻辑，直接做完整重建。
    if (
        not force_rebuild
        and faiss_dir.exists()
        and bm25_path.exists()
        and _manifest_matches(manifest_path, config, build_stage=BUILD_STAGE_COMPLETE)
    ):
        LOGGER.info("Loading cached FAISS and BM25 indexes from %s.", index_dir)
        vectordb = FAISS.load_local(
            str(faiss_dir),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        bm25_payload = _load_bm25_payload(bm25_path)
        BM25Okapi = _load_bm25_class()
        documents = _restore_documents_from_bm25_payload(bm25_payload)
        bm25_index = BM25Okapi(bm25_payload["tokenized_corpus"])
        return vectordb, bm25_index, documents

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and _has_legacy_faiss_index(index_dir):
        # 如果用户已经有旧版 FAISS 索引，就直接迁移，避免再次进行 chunking knowledge base。
        LOGGER.info("Migrating legacy FAISS index at %s into hybrid retrieval format.", index_dir)
        vectordb = FAISS.load_local(
            str(index_dir),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        documents = _extract_documents_from_faiss(vectordb)
        if not documents:
            raise ValueError("Legacy FAISS index was loaded, but no documents could be recovered.")

        bm25_index, tokenized_corpus = _build_bm25_index(documents, config.title_token_boost)

        # 迁移后保存成新目录结构，后续启动就能直接复用，不再走迁移分支。
        _save_bm25_payload(bm25_path, documents, tokenized_corpus)
        _write_manifest(manifest_path, config, build_stage=BUILD_STAGE_BM25_READY)
        vectordb.save_local(str(faiss_dir))
        _write_manifest(manifest_path, config, build_stage=BUILD_STAGE_COMPLETE)
        LOGGER.info("Legacy FAISS migration finished with %d cached chunks.", len(documents))
        return vectordb, bm25_index, documents

    if (
        not force_rebuild
        and bm25_path.exists()
        and _manifest_matches(manifest_path, config, build_stage=BUILD_STAGE_BM25_READY)
    ):
        # 如果上一次已经完成 chunking + BM25，但在 FAISS 阶段被杀掉，这里直接恢复。
        LOGGER.info("Resuming FAISS build from cached BM25 payload at %s.", bm25_path)
        bm25_payload = _load_bm25_payload(bm25_path)
        documents = _restore_documents_from_bm25_payload(bm25_payload)
        BM25Okapi = _load_bm25_class()
        bm25_index = BM25Okapi(bm25_payload["tokenized_corpus"])
        vectordb = _build_faiss_incrementally(
            documents=documents,
            embedding_model=embedding_model,
            faiss_dir=faiss_dir,
            batch_size=max(faiss_build_batch_size, 1),
        )
        _write_manifest(manifest_path, config, build_stage=BUILD_STAGE_COMPLETE)
        LOGGER.info("Resumed FAISS build finished with %d cached chunks.", len(documents))
        return vectordb, bm25_index, documents

    # 走到这里说明是首次构建，或者用户显式要求完整重建。
    LOGGER.info("Building FAISS and BM25 indexes at %s. force_rebuild=%s", index_dir, force_rebuild)

    documents = build_chunked_documents(
        data_files=config.data_files,
        embed_model_name=config.embed_model_name,
        tmp_path=config.tmp_path,
        subset=config.subset,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        ocr_enabled=config.ocr_enabled,
    )
    if not documents:
        raise ValueError("No chunked documents were produced from the knowledge base.")

    bm25_index, tokenized_corpus = _build_bm25_index(documents, config.title_token_boost)
    # BM25 和文档列表先落盘；如果后续 FAISS 构建被系统杀掉，下次可以直接恢复到这一阶段。
    _save_bm25_payload(bm25_path, documents, tokenized_corpus)
    _write_manifest(manifest_path, config, build_stage=BUILD_STAGE_BM25_READY)
    del tokenized_corpus
    gc.collect()

    vectordb = _build_faiss_incrementally(
        documents=documents,
        embedding_model=embedding_model,
        faiss_dir=faiss_dir,
        batch_size=max(faiss_build_batch_size, 1),
    )
    _write_manifest(manifest_path, config, build_stage=BUILD_STAGE_COMPLETE)
    LOGGER.info("Indexed %d chunks.", len(documents))
    return vectordb, bm25_index, documents


class AdvancedRetriever:
    """Runs hybrid retrieval followed by cross-encoder reranking."""

    def __init__(
        self,
        *,
        vectordb: FAISS,
        bm25_index: Any,
        bm25_documents: Sequence[Document],
        alpha: float = 1.25,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        hybrid_top_k: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        rerank_enabled: bool = True,
        rerank_cache_size: int = 4096,
        rerank_batch_size: int = 8,
    ) -> None:
        self._vectordb = vectordb
        self._bm25_index = bm25_index
        self._bm25_documents = list(bm25_documents)
        self._alpha = alpha
        self._vector_top_k = vector_top_k
        self._bm25_top_k = bm25_top_k
        self._hybrid_top_k = hybrid_top_k
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        self._reranker_model_name = reranker_model_name
        self._rerank_enabled = rerank_enabled
        self._rerank_batch_size = rerank_batch_size
        self._reranker: Any | None = None
        self._rerank_cache = ScoreCache(max_size=rerank_cache_size)
        # 记录最近一次检索摘要，方便 Tool 返回给上层 Agent 和人工排查。
        self._last_summary: dict[str, Any] = {}

    def setup(self) -> None:
        """Optionally preloads the reranker."""
        if self._rerank_enabled:
            self._ensure_reranker_loaded()

    def retrieve(self, query: str) -> list[ScoredChunk]:
        """Retrieves the final ranked documents for a user query.

        Args:
            query: User query.

        Returns:
            A list of reranked chunks sorted by final score.
        """
        results, _ = self.retrieve_with_summary(query)
        return results

    def retrieve_with_summary(self, query: str) -> tuple[list[ScoredChunk], dict[str, Any]]:
        """返回结果和检索摘要，便于日志、调试和 Tool 结构化输出。"""
        hybrid_results, summary = self.hybrid_search(query)
        if not hybrid_results:
            self._last_summary = summary
            return [], summary
        if not self._rerank_enabled:
            final_results = hybrid_results[: self._final_top_k]
            summary.update(
                {
                    "rerank_applied": False,
                    "rerank_status": "disabled",
                    "returned_count": len(final_results),
                }
            )
            self._last_summary = summary
            return final_results, summary
        final_results, rerank_summary = self._rerank(query, hybrid_results)
        summary.update(rerank_summary)
        self._last_summary = summary
        return final_results, summary

    def hybrid_search(self, query: str) -> tuple[list[ScoredChunk], dict[str, Any]]:
        """Retrieves hybrid candidates via FAISS, BM25, and weighted RRF."""
        query = _safe_text(query)
        if not query:
            return [], {"query": query, "hybrid_candidate_count": 0}

        # 查询画像用于动态调节 sparse/dense 融合权重。
        query_profile = build_query_profile(query, self._alpha)
        fused: dict[str, ScoredChunk] = {}

        vector_results = self._vector_search(query)
        for result in vector_results:
            chunk = fused.setdefault(
                result.chunk_id,
                ScoredChunk(document=result.document),
            )
            chunk.vector_score = result.vector_score
            chunk.vector_rank = result.vector_rank
            chunk.hybrid_score += 1.0 / (self._rrf_k + (result.vector_rank or 0))

        bm25_results = self._bm25_search(query)
        for result in bm25_results:
            chunk = fused.setdefault(
                result.chunk_id,
                ScoredChunk(document=result.document),
            )
            chunk.bm25_score = result.bm25_score
            chunk.bm25_rank = result.bm25_rank
            # 对 ISBN / 书名 / 标签导向的查询，BM25 分支会被动态放大。
            chunk.hybrid_score += query_profile.effective_alpha / (self._rrf_k + (result.bm25_rank or 0))

        # 融合完成后，再加一层标题/标签命中加分，让“馆藏式精确查询”更稳定。
        for chunk in fused.values():
            chunk.hybrid_score += self._metadata_bonus(query_profile, chunk.document)

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                item.hybrid_score,
                item.bm25_score if item.bm25_score is not None else -math.inf,
                item.vector_score if item.vector_score is not None else -math.inf,
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            item.hybrid_rank = rank
            item.final_score = item.hybrid_score

        top_ranked = ranked[: self._hybrid_top_k]
        LOGGER.info(
            "Hybrid search query=%r alpha=%.2f vector_hits=%d bm25_hits=%d fused_hits=%d top_ids=%s",
            query,
            query_profile.effective_alpha,
            len(vector_results),
            len(bm25_results),
            len(top_ranked),
            [item.chunk_id for item in top_ranked[:5]],
        )
        summary = {
            "query": query,
            "effective_alpha": round(query_profile.effective_alpha, 4),
            "query_profile": {
                "contains_identifier": query_profile.contains_identifier,
                "has_delimited_phrase": query_profile.has_delimited_phrase,
                "is_short_query": query_profile.is_short_query,
                "looks_like_title_lookup": query_profile.looks_like_title_lookup,
                "identifier_terms": list(query_profile.identifier_terms),
                "query_terms": list(query_profile.query_terms),
            },
            "vector_candidate_count": len(vector_results),
            "bm25_candidate_count": len(bm25_results),
            "hybrid_candidate_count": len(top_ranked),
        }
        return top_ranked, summary

    def _vector_search(self, query: str) -> list[ScoredChunk]:
        """Runs the dense retrieval branch."""
        # fetch_k 取更大一点，让 FAISS 在召回阶段保留更多候选，再由后续融合和精排裁决。
        docs_and_scores = self._vectordb.similarity_search_with_relevance_scores(
            query,
            k=self._vector_top_k,
            fetch_k=max(self._vector_top_k * 4, 40),
        )
        results: list[ScoredChunk] = []
        for rank, (document, score) in enumerate(docs_and_scores, start=1):
            results.append(
                ScoredChunk(
                    document=document,
                    vector_score=float(score),
                    vector_rank=rank,
                )
            )
        return results

    def _bm25_search(self, query: str) -> list[ScoredChunk]:
        """Runs the sparse retrieval branch."""
        query_tokens = list(dict.fromkeys(tokenize_for_bm25(query)))
        if not query_tokens:
            return []

        scores = list(self._bm25_index.get_scores(query_tokens))
        scored_indices = [(index, float(score)) for index, score in enumerate(scores)]
        positive_hits = [item for item in scored_indices if item[1] > 0]
        ranked_pairs = sorted(
            positive_hits or scored_indices,
            key=lambda item: item[1],
            reverse=True,
        )[: self._bm25_top_k]

        results: list[ScoredChunk] = []
        for rank, (index, score) in enumerate(ranked_pairs, start=1):
            results.append(
                ScoredChunk(
                    document=self._bm25_documents[index],
                    bm25_score=score,
                    bm25_rank=rank,
                )
            )
        return results

    def _metadata_bonus(self, query_profile: QueryProfile, document: Document) -> float:
        """为标题、标签、编号命中添加额外加分，强化馆藏精确检索场景。"""
        title = _collapse_whitespace(_safe_text(document.metadata.get("title")).lower())
        tags = [_safe_text(tag).lower() for tag in _normalize_tags(document.metadata.get("tags"))]
        page_content = _collapse_whitespace(document.page_content.lower())
        query = query_profile.phrase_query.lower()
        bonus = 0.0

        if query and title:
            if title == query:
                bonus += 0.12
            elif query in title:
                bonus += 0.08
            elif title.startswith(query):
                bonus += 0.05

        # 标签是次于标题的重要字段，命中后给轻量奖励，避免压过主排序。
        for term in query_profile.query_terms:
            if any(term in tag for tag in tags):
                bonus += 0.015

        # ISBN / 编号类命中对图书检索很关键，因此单独拉高。
        if query_profile.contains_identifier and any(
            identifier in page_content for identifier in query_profile.identifier_terms
        ):
            bonus += 0.08

        return min(bonus, 0.20)

    def _ensure_reranker_loaded(self) -> None:
        """Loads the reranker lazily to keep startup overhead manageable."""
        if self._reranker is not None:
            return

        FlagReranker = _load_flag_reranker()
        use_fp16 = torch.cuda.is_available()
        resolved_model_path = _resolve_local_hf_snapshot(self._reranker_model_name)
        try:
            LOGGER.info(
                "Loading reranker model=%s resolved_path=%s use_fp16=%s.",
                self._reranker_model_name,
                resolved_model_path,
                use_fp16,
            )
            # 优先走本地缓存快照目录，避免 FlagEmbedding 再去 Hugging Face 做在线探测。
            self._reranker = FlagReranker(resolved_model_path, use_fp16=use_fp16)
        except RuntimeError as exc:
            # GPU 装载失败时先尝试退回更稳妥的非 fp16 方式，避免直接让整条链路中断。
            if use_fp16 and "out of memory" in str(exc).lower():
                LOGGER.warning("Reranker fp16 load failed with OOM, retrying in full precision.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._reranker = FlagReranker(resolved_model_path, use_fp16=False)
            else:
                raise

    def _rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
    ) -> tuple[list[ScoredChunk], dict[str, Any]]:
        """Reranks hybrid candidates with a cross-encoder model."""
        try:
            self._ensure_reranker_loaded()
        except Exception as exc:
            # 精排不可用时退回 Hybrid 结果，保证主流程可继续工作。
            LOGGER.warning("Failed to load reranker, fallback to hybrid search: %s", exc)
            fallback = list(candidates[: self._final_top_k])
            for rank, item in enumerate(fallback, start=1):
                item.rerank_rank = rank
                item.final_score = item.hybrid_score
            return fallback, {
                "rerank_applied": False,
                "rerank_status": f"load_failed:{type(exc).__name__}",
                "returned_count": len(fallback),
                "top1_changed": False,
            }

        if self._reranker is None:
            fallback = list(candidates[: self._final_top_k])
            return fallback, {
                "rerank_applied": False,
                "rerank_status": "unavailable",
                "returned_count": len(fallback),
                "top1_changed": False,
            }

        uncached_pairs: list[list[str]] = []
        uncached_keys: list[str] = []
        for candidate in candidates:
            key = self._cache_key(query, candidate.document.page_content)
            cached_score = self._rerank_cache.get(key)
            if cached_score is not None:
                candidate.rerank_score = cached_score
                continue
            uncached_pairs.append([query, candidate.document.page_content])
            uncached_keys.append(key)

        if uncached_pairs:
            try:
                computed_scores = self._compute_rerank_scores(uncached_pairs)
            except RuntimeError as exc:
                # 推理阶段显存不足时清缓存并降级，避免一次 OOM 把检索全打断。
                if "out of memory" in str(exc).lower():
                    LOGGER.warning("Reranker inference OOM, fallback to hybrid search: %s", exc)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    fallback = list(candidates[: self._final_top_k])
                    return fallback, {
                        "rerank_applied": False,
                        "rerank_status": "oom_fallback",
                        "returned_count": len(fallback),
                        "top1_changed": False,
                    }
                raise
            for key, candidate, score in zip(
                uncached_keys,
                [item for item in candidates if item.rerank_score is None],
                computed_scores,
                strict=True,
            ):
                candidate.rerank_score = score
                self._rerank_cache.set(key, score)

        reranked = sorted(
            candidates,
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else -math.inf,
                item.hybrid_score,
            ),
            reverse=True,
        )
        for rank, item in enumerate(reranked, start=1):
            item.rerank_rank = rank
            item.final_score = item.rerank_score if item.rerank_score is not None else item.hybrid_score

        final_results = reranked[: self._final_top_k]
        self._log_rerank_lift(query, candidates, final_results)
        return final_results, {
            "rerank_applied": True,
            "rerank_status": "ok",
            "returned_count": len(final_results),
            # top1_changed 能直观看出 reranker 是否真的改写了最终首条证据。
            "top1_changed": bool(candidates and final_results and candidates[0].chunk_id != final_results[0].chunk_id),
            "hybrid_top1_chunk_id": candidates[0].chunk_id if candidates else None,
            "rerank_top1_chunk_id": final_results[0].chunk_id if final_results else None,
            "cache_size": len(self._rerank_cache._data),
            "uncached_pair_count": len(uncached_pairs),
        }

    def _compute_rerank_scores(self, query_document_pairs: Sequence[Sequence[str]]) -> list[float]:
        """Computes normalized reranker scores for query-document pairs."""
        assert self._reranker is not None
        try:
            scores = self._reranker.compute_score(
                list(query_document_pairs),
                batch_size=self._rerank_batch_size,
                normalize=True,
            )
        except TypeError:
            scores = self._reranker.compute_score(
                list(query_document_pairs),
                batch_size=self._rerank_batch_size,
            )
            if isinstance(scores, (float, int)):
                scores = [scores]
            scores = [self._sigmoid(float(score)) for score in scores]

        if isinstance(scores, (float, int)):
            scores = [scores]
        return [float(score) for score in scores]

    def _log_rerank_lift(
        self,
        query: str,
        hybrid_results: Sequence[ScoredChunk],
        reranked_results: Sequence[ScoredChunk],
    ) -> None:
        """Logs how reranking changed the ranking order."""
        hybrid_rank_map = {item.chunk_id: item.hybrid_rank for item in hybrid_results}
        rerank_summary = [
            {
                "chunk_id": item.chunk_id,
                "hybrid_rank": hybrid_rank_map.get(item.chunk_id),
                "rerank_rank": item.rerank_rank,
                "hybrid_score": round(float(item.hybrid_score), 6),
                "rerank_score": None if item.rerank_score is None else round(float(item.rerank_score), 6),
                "title": item.document.metadata.get("title"),
            }
            for item in reranked_results
        ]
        LOGGER.info("Rerank summary query=%r results=%s", query, rerank_summary)

    @staticmethod
    def _cache_key(query: str, document_text: str) -> str:
        """Creates a stable cache key for a query-document pair."""
        payload = f"{query}\0{document_text}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Normalizes an arbitrary score into the [0, 1] range."""
        return 1.0 / (1.0 + math.exp(-value))


class SemanticRetriever(Tool):
    """Hybrid semantic retriever with cross-encoder reranking.

    Use this tool for local knowledge-base retrieval when the query needs
    fact-grounded answers, entity lookup, academic consultation, title or ISBN
    matching, or multi-hop factual verification. The tool combines FAISS dense
    retrieval, BM25 keyword retrieval, weighted RRF fusion, and BGE
    cross-encoder reranking, then returns the most relevant chunks with
    relevance scores and source metadata.
    """

    name = "semantic_retriever"
    description = (
        "Hybrid semantic retriever for local fact grounding. Use it when you need "
        "high-precision retrieval over the knowledge base for academic questions, "
        "book-title lookup, ISBN lookup, terminology matching, or factual verification. "
        "The tool combines FAISS dense retrieval, BM25 keyword retrieval, weighted RRF "
        "fusion, and cross-encoder reranking before returning the top evidence "
        "chunks with relevance scores and source metadata."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "A concise retrieval query describing the target fact, title, ISBN, concept, or evidence need.",
        }
    }
    output_type = "object"
    output_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "chunk_id": {"type": "string"},
                        "content": {"type": "string"},
                        "relevance_score": {"type": "number"},
                        "scores": {"type": "object"},
                        "ranks": {"type": "object"},
                        "source": {"type": "object"},
                    },
                },
            },
            "retrieval_summary": {"type": "object"},
        },
    }

    def __init__(self, retriever: AdvancedRetriever, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._retriever = retriever

    def setup(self) -> None:
        """Loads heavy runtime dependencies lazily on first tool use."""
        self._retriever.setup()
        self.is_initialized = True

    def forward(self, query: str) -> dict[str, Any]:
        """Returns ranked local evidence for the given query."""
        # 这里返回的不只是片段本身，还带上检索摘要，方便 Agent 和人工一起看链路效果。
        results, retrieval_summary = self._retriever.retrieve_with_summary(query)
        payload = [item.to_payload(rank=index) for index, item in enumerate(results, start=1)]
        return {
            "query": query,
            "results": payload,
            "retrieval_summary": {
                **retrieval_summary,
                "top_source_titles": [item.document.metadata.get("title") for item in results],
                "supports_hybrid_search": True,
                "supports_cross_encoder_reranking": True,
            },
        }
