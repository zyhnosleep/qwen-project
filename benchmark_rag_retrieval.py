"""Benchmark retrieval quality and latency for the RAG stack."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document

from qwen_agentic_rag import build_retriever, configure_logging, load_config
from utils.retrieval import ScoredChunk, _extract_identifier_terms, _safe_text


@dataclass(frozen=True)
class BenchmarkTask:
    """一条检索评测任务。"""

    query: str
    query_type: str
    target_record_id: str
    target_title: str


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark vector, hybrid, and rerank retrieval.")
    parser.add_argument("--max-docs", type=int, default=300, help="Maximum unique source documents to sample.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K used for Recall@K and Top-1 evaluation.")
    parser.add_argument(
        "--output",
        type=str,
        default="outs/rag_retrieval_benchmark.json",
        help="Path to save benchmark metrics as JSON.",
    )
    return parser.parse_args()


def _pick_benchmark_tasks(documents: list[Document], max_docs: int) -> list[BenchmarkTask]:
    """从现有知识库文档中抽取一组可自动评估的查询。"""
    unique_by_record: dict[str, Document] = {}
    for document in documents:
        record_id = _safe_text(document.metadata.get("record_id"))
        title = _safe_text(document.metadata.get("title"))
        if not record_id or not title or record_id in unique_by_record:
            continue
        unique_by_record[record_id] = document
        if len(unique_by_record) >= max_docs:
            break

    tasks: list[BenchmarkTask] = []
    for record_id, document in unique_by_record.items():
        title = _safe_text(document.metadata.get("title"))
        tasks.append(
            BenchmarkTask(
                query=title,
                query_type="title",
                target_record_id=record_id,
                target_title=title,
            )
        )

        # 如果正文中能抽出 ISBN/编号，就额外增加一条“精确术语”测试任务。
        identifier_terms = _extract_benchmark_identifier_terms(document)
        if identifier_terms:
            tasks.append(
                BenchmarkTask(
                    query=identifier_terms[0],
                    query_type="identifier",
                    target_record_id=record_id,
                    target_title=title,
                )
            )

    return tasks


def _extract_benchmark_identifier_terms(document: Document) -> list[str]:
    """为 benchmark 只保留真正有意义的精确术语，如 ISBN、型号、编号。"""
    content_lines = [line.strip() for line in document.page_content.splitlines() if line.strip()]
    # 跳过我们在 chunk 前面人为拼接的 Title / Tags 前缀，避免采样出无效查询 `title:`。
    filtered_text = "\n".join(
        line for line in content_lines if not line.lower().startswith(("title:", "tags:"))
    )
    candidates = list(_extract_identifier_terms(filtered_text))

    # 优先保留更像 ISBN / 型号的 token：既包含数字，也尽量包含字母或连接符。
    preferred: list[str] = []
    for token in candidates:
        normalized = token.lower()
        if normalized in {"title", "title:", "tags", "tags:"}:
            continue
        has_digit = any(char.isdigit() for char in token)
        if not has_digit:
            continue
        if token.endswith(":"):
            continue
        preferred.append(token)

    # 去重并限制长度，避免很长的垃圾串影响 benchmark。
    deduped: list[str] = []
    seen: set[str] = set()
    for token in preferred:
        normalized = token.lower()
        if normalized in seen:
            continue
        if len(token) > 32:
            continue
        seen.add(normalized)
        deduped.append(token)
    return deduped


def _normalize_result_record_ids(results: list[Any]) -> list[str]:
    """把不同检索接口的返回统一转成 record_id 列表。"""
    record_ids: list[str] = []
    for item in results:
        if isinstance(item, ScoredChunk):
            record_ids.append(_safe_text(item.document.metadata.get("record_id")))
        elif isinstance(item, Document):
            record_ids.append(_safe_text(item.metadata.get("record_id")))
        elif isinstance(item, tuple) and item and isinstance(item[0], Document):
            record_ids.append(_safe_text(item[0].metadata.get("record_id")))
        else:
            raise TypeError(f"Unsupported retrieval result type: {type(item)!r}")
    return record_ids


def _evaluate_variant(
    *,
    tasks: list[BenchmarkTask],
    top_k: int,
    search_fn: Callable[[str], list[Any]],
) -> dict[str, Any]:
    """评估单个检索变体的召回、Top-1 和平均延迟。"""
    latencies_ms: list[float] = []
    hits_at_k: list[int] = []
    hits_at_1: list[int] = []
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for task in tasks:
        start = time.perf_counter()
        results = search_fn(task.query)
        latency_ms = (time.perf_counter() - start) * 1000.0
        record_ids = _normalize_result_record_ids(results)

        hit_at_k = int(task.target_record_id in record_ids[:top_k])
        hit_at_1 = int(bool(record_ids) and record_ids[0] == task.target_record_id)

        latencies_ms.append(latency_ms)
        hits_at_k.append(hit_at_k)
        hits_at_1.append(hit_at_1)
        by_type[task.query_type]["latency_ms"].append(latency_ms)
        by_type[task.query_type]["recall_at_k"].append(hit_at_k)
        by_type[task.query_type]["top1_accuracy"].append(hit_at_1)

    metrics = {
        "query_count": len(tasks),
        "recall_at_k": round(sum(hits_at_k) / len(hits_at_k), 4) if hits_at_k else 0.0,
        "top1_accuracy": round(sum(hits_at_1) / len(hits_at_1), 4) if hits_at_1 else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
        "p95_latency_ms": round(_percentile(latencies_ms, 95), 2) if latencies_ms else 0.0,
        "by_query_type": {},
    }
    for query_type, values in by_type.items():
        metrics["by_query_type"][query_type] = {
            "recall_at_k": round(sum(values["recall_at_k"]) / len(values["recall_at_k"]), 4),
            "top1_accuracy": round(sum(values["top1_accuracy"]) / len(values["top1_accuracy"]), 4),
            "avg_latency_ms": round(statistics.mean(values["latency_ms"]), 2),
        }
    return metrics


def _percentile(values: list[float], percentile: float) -> float:
    """计算简单百分位，避免额外引入 numpy 依赖。"""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(max(int(round((percentile / 100.0) * (len(sorted_values) - 1))), 0), len(sorted_values) - 1)
    return sorted_values[index]


def main() -> None:
    """Runs the benchmark."""
    args = parse_args()
    config = load_config()
    configure_logging(config.log_level, config.log_dir)

    retriever = build_retriever(config)
    documents = retriever._bm25_documents  # noqa: SLF001
    tasks = _pick_benchmark_tasks(documents, args.max_docs)
    if not tasks:
        raise ValueError("No benchmark tasks could be created from the current knowledge base.")

    vector_k = max(args.top_k, retriever._vector_top_k)  # noqa: SLF001

    results = {
        "config": {
            "top_k": args.top_k,
            "max_docs": args.max_docs,
            "index_dir": config.index_dir,
            "embedding_device": config.embedding_device,
            "rerank_enabled": config.rerank_enabled,
        },
        "task_summary": {
            "task_count": len(tasks),
            "title_queries": sum(1 for task in tasks if task.query_type == "title"),
            "identifier_queries": sum(1 for task in tasks if task.query_type == "identifier"),
        },
        "metrics": {},
    }

    results["metrics"]["vector_only"] = _evaluate_variant(
        tasks=tasks,
        top_k=args.top_k,
        search_fn=lambda query: retriever._vectordb.similarity_search_with_relevance_scores(  # noqa: SLF001
            query,
            k=vector_k,
            fetch_k=max(vector_k * 4, 40),
        )[: args.top_k],
    )
    results["metrics"]["hybrid"] = _evaluate_variant(
        tasks=tasks,
        top_k=args.top_k,
        search_fn=lambda query: retriever.hybrid_search(query)[0][: args.top_k],
    )
    results["metrics"]["hybrid_rerank"] = _evaluate_variant(
        tasks=tasks,
        top_k=args.top_k,
        search_fn=lambda query: retriever.retrieve_with_summary(query)[0][: args.top_k],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
