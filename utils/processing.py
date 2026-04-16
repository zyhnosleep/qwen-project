"""Document processing helpers for parquet and layout-aware PDF ingestion."""

from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import datasets
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.runtime import configure_runtime_environment

LOGGER = logging.getLogger(__name__)
SUPPORTED_RETRIEVAL_SUFFIXES = {".parquet", ".pdf"}
DEFAULT_OCR_DPI = 320
configure_runtime_environment(Path(__file__).resolve().parents[1])


@dataclass(frozen=True)
class KnowledgeSourceDiscovery:
    """描述一次知识源扫描的结果。"""

    retrieval_files: tuple[str, ...]
    parquet_files: tuple[str, ...]
    pdf_files: tuple[str, ...]


@dataclass(frozen=True)
class LayoutBlock:
    """表示一个带版面语义的文档块。"""

    content: str
    metadata: dict[str, Any]
    block_type: str


def discover_knowledge_sources(data_path: str) -> KnowledgeSourceDiscovery:
    """扫描目录，找到可用于检索构建的 parquet 和 PDF 文件。"""
    root_path = Path(data_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Knowledge data path does not exist: {data_path}")

    parquet_files: list[str] = []
    pdf_files: list[str] = []
    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_RETRIEVAL_SUFFIXES:
            continue
        if file_path.suffix.lower() == ".parquet":
            parquet_files.append(str(file_path.resolve()))
        elif file_path.suffix.lower() == ".pdf":
            pdf_files.append(str(file_path.resolve()))

    retrieval_files = tuple(parquet_files + pdf_files)
    if not retrieval_files:
        raise FileNotFoundError(f"No parquet or PDF files were found under {data_path}.")

    return KnowledgeSourceDiscovery(
        retrieval_files=retrieval_files,
        parquet_files=tuple(parquet_files),
        pdf_files=tuple(pdf_files),
    )


def load_knowledge_documents(
    *,
    source_files: Sequence[str],
    embed_model_name: str,
    tmp_path: str,
    subset: int,
    chunk_size: int,
    chunk_overlap: int,
    ocr_enabled: bool,
) -> list[Document]:
    """统一加载 parquet 与 PDF，并输出适合索引构建的 Document 列表。"""
    splitter = build_text_splitter(
        embed_model_name=embed_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    parquet_files = [path for path in source_files if Path(path).suffix.lower() == ".parquet"]
    pdf_files = [path for path in source_files if Path(path).suffix.lower() == ".pdf"]

    documents: list[Document] = []
    if parquet_files:
        documents.extend(
            _load_parquet_documents(
                parquet_files=parquet_files,
                tmp_path=tmp_path,
                subset=subset,
                splitter=splitter,
            )
        )
    if pdf_files:
        for pdf_path in pdf_files:
            documents.extend(
                _load_pdf_documents(
                    pdf_path=pdf_path,
                    splitter=splitter,
                    ocr_enabled=ocr_enabled,
                )
            )

    deduped_documents = dedupe_documents(documents)
    LOGGER.info("Loaded %d knowledge chunks after deduplication.", len(deduped_documents))
    return deduped_documents


def build_text_splitter(
    *,
    embed_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """构建统一的文本切分器。"""
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(embed_model_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ".", " ", ""],
    )


def _load_parquet_documents(
    *,
    parquet_files: Sequence[str],
    tmp_path: str,
    subset: int,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """加载 parquet 语料，并按当前文本切分策略生成 chunk。"""
    dataset = datasets.load_dataset(
        "parquet",
        data_files=list(parquet_files),
        split="train",
        cache_dir=tmp_path,
    )
    if subset > 0:
        dataset = dataset.select(range(subset))

    documents: list[Document] = []
    for record in tqdm(dataset, desc="Chunking knowledge base"):
        record_id = _safe_text(record.get("id")) or _safe_text(record.get("uniqueKey"))
        title = _safe_text(record.get("title"))
        content = _safe_text(record.get("content"))
        url = _safe_text(record.get("url"))
        tags = _normalize_tags(record.get("tags"))
        if not content:
            continue

        # parquet 语料没有页码概念，因此 page_number 统一留空。
        base_block = LayoutBlock(
            content=_compose_prefixed_text(title=title, tags=tags, body=content),
            metadata={
                "title": title,
                "url": url,
                "record_id": record_id,
                "source": url or title or "local_knowledge_base",
                "source_file": url or "",
                "source_type": "parquet",
                "tags": tags,
                "page_number": None,
                "block_type": "text",
                "citation_hint": _build_citation_hint(title, None, "text"),
            },
            block_type="text",
        )
        documents.extend(_split_layout_block(base_block, splitter))

    return documents


def _load_pdf_documents(
    *,
    pdf_path: str,
    splitter: RecursiveCharacterTextSplitter,
    ocr_enabled: bool,
) -> list[Document]:
    """加载复杂 PDF，保留页码、表格和图片说明信息。"""
    _configure_unstructured_local_models()
    source_path = Path(pdf_path)
    text_layer_blocks = _extract_pdf_text_layer_blocks(source_path)
    text_layer_available = _has_strong_text_layer(text_layer_blocks)
    nltk_resources_ready = _has_required_nltk_resources()

    # 有强文字层时，正文优先走文字层；版面解析只做最佳努力补充，失败不应阻断主流程。
    strategy = "fast" if text_layer_available else ("ocr_only" if ocr_enabled else "hi_res")
    LOGGER.info(
        "Parsing PDF file=%s strategy=%s text_layer_available=%s nltk_ready=%s",
        pdf_path,
        strategy,
        text_layer_available,
        nltk_resources_ready,
    )
    elements: Sequence[Any] = []
    partition_pdf = _load_partition_pdf() if not text_layer_available else None
    if partition_pdf is not None:
        if not nltk_resources_ready:
            LOGGER.warning(
                "Skipping unstructured layout parsing for file=%s because required NLTK resources are unavailable.",
                pdf_path,
            )
        else:
            try:
                elements = _partition_pdf_with_strategy(
                    partition_pdf=partition_pdf,
                    source_path=source_path,
                    strategy=strategy,
                )
            except Exception as exc:
                # 当前环境如果缺少 punkt/nltk 资源或布局模型依赖，就回退到 fast，再不行就交给 OCR fallback。
                LOGGER.warning(
                    "Layout-aware PDF parsing failed for file=%s strategy=%s, fallback to fast. error=%s",
                    pdf_path,
                    strategy,
                    exc,
                )
                try:
                    elements = _partition_pdf_with_strategy(
                        partition_pdf=partition_pdf,
                        source_path=source_path,
                        strategy="fast",
                    )
                except Exception as fast_exc:
                    LOGGER.warning(
                        "Fast PDF parsing also failed for file=%s. Falling back to OCR/text layer only. error=%s",
                        pdf_path,
                        fast_exc,
                    )
                    elements = []

    layout_blocks: list[LayoutBlock] = []
    if text_layer_available:
        # 有文字层时优先使用 PDF 自带文本，通常比 hi_res 识别出来的正文更干净。
        layout_blocks.extend(text_layer_blocks)

    if elements:
        layout_blocks.extend(
            _elements_to_layout_blocks(
                elements=elements,
                source_path=source_path,
                include_text=not text_layer_available,
            )
        )

    if not layout_blocks:
        # 对扫描版或 PPT 渲染型 PDF，版面模型失败后再走一次纯 OCR 兜底，保证至少能按页抽出文本。
        LOGGER.warning(
            "PDF parsing returned no usable layout blocks for file=%s. Falling back to plain OCR page extraction.",
            pdf_path,
        )
        layout_blocks = _ocr_fallback_layout_blocks(source_path)
    else:
        layout_blocks = _supplement_layout_blocks_with_ocr(
            layout_blocks=layout_blocks,
            source_path=source_path,
            prefer_existing_text=text_layer_available,
        )

    documents: list[Document] = []
    for block in layout_blocks:
        documents.extend(_split_layout_block(block, splitter))
    if documents:
        return documents
    raise RuntimeError(
        "PDF parsing returned no elements. This file likely requires a richer OCR/layout stack "
        "or different parser settings."
    )


def _partition_pdf_with_strategy(
    *,
    partition_pdf: Any,
    source_path: Path,
    strategy: str,
) -> Sequence[Any]:
    """兼容不同 Unstructured 版本签名，并按指定策略解析 PDF。"""
    try:
        return partition_pdf(
            filename=str(source_path),
            strategy=strategy,
            infer_table_structure=True,
            include_page_breaks=False,
            extract_image_block_types=["Image", "Table"],
            hi_res_model_name="yolox",
        )
    except TypeError:
        # 部分版本没有 `extract_image_block_types` 参数，做一次向后兼容。
        return partition_pdf(
            filename=str(source_path),
            strategy=strategy,
            infer_table_structure=True,
            include_page_breaks=False,
            hi_res_model_name="yolox",
        )


def _ocr_fallback_layout_blocks(source_path: Path) -> list[LayoutBlock]:
    """在 Unstructured 版面解析失败时，逐页做 OCR 抽取文本。"""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "OCR fallback requires `pdf2image`, but it is unavailable."
        ) from exc

    try:
        # 优先兼容当前环境中已安装的 unstructured OCR 包装器。
        from unstructured_pytesseract import image_to_data, image_to_string
    except ImportError:
        try:
            from pytesseract import image_to_data, image_to_string
        except ImportError as exc:
            raise RuntimeError(
                "OCR fallback requires `pytesseract` or `unstructured_pytesseract`, but neither is available."
            ) from exc

    ocr_languages = _select_ocr_languages()
    images = convert_from_path(str(source_path), dpi=DEFAULT_OCR_DPI)
    layout_blocks: list[LayoutBlock] = []
    for page_number, image in enumerate(images, start=1):
        processed_image = _preprocess_image_for_ocr(image)
        block_texts = _ocr_page_blocks(
            image=processed_image,
            image_to_data=image_to_data,
            image_to_string=image_to_string,
            languages=ocr_languages,
        )
        if not block_texts:
            continue

        for block_index, block_text in enumerate(block_texts):
            layout_blocks.append(
                LayoutBlock(
                    content=block_text,
                    metadata={
                        "title": source_path.stem,
                        "url": "",
                        "record_id": f"{source_path.stem}:page:{page_number}:ocr:{block_index}",
                        "source": str(source_path),
                        "source_file": str(source_path),
                        "source_type": "pdf",
                        "page_number": page_number,
                        "block_type": "ocr_text",
                        "table_html": None,
                        "image_caption": None,
                        "citation_hint": _build_citation_hint(source_path.stem, page_number, "text"),
                        "ocr_fallback": True,
                        "ocr_languages": ocr_languages,
                        "tags": [],
                    },
                    block_type="text",
                )
            )
    return layout_blocks


def _extract_pdf_text_layer_blocks(source_path: Path) -> list[LayoutBlock]:
    """优先从 PDF 自带文字层抽取正文，适合简历/论文等文本型 PDF。"""
    from pypdf import PdfReader

    reader = PdfReader(str(source_path))
    layout_blocks: list[LayoutBlock] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = _clean_layout_text(page.extract_text() or "")
        for block_index, block_text in enumerate(_split_text_layer_blocks(raw_text)):
            layout_blocks.append(
                LayoutBlock(
                    content=block_text,
                    metadata={
                        "title": source_path.stem,
                        "url": "",
                        "record_id": f"{source_path.stem}:page:{page_number}:textlayer:{block_index}",
                        "source": str(source_path),
                        "source_file": str(source_path),
                        "source_type": "pdf",
                        "page_number": page_number,
                        "block_type": "text",
                        "table_html": None,
                        "image_caption": None,
                        "citation_hint": _build_citation_hint(source_path.stem, page_number, "text"),
                        "text_layer": True,
                        "tags": [],
                    },
                    block_type="text",
                )
            )
    return layout_blocks


def _split_text_layer_blocks(raw_text: str) -> list[str]:
    """把文字层文本按自然段聚合成较稳定的块。"""
    if not raw_text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", raw_text) if part.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in raw_text.splitlines() if line.strip()]

    blocks: list[str] = []
    buffer: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        if buffer and current_length + len(paragraph) > 700:
            blocks.append("\n".join(buffer))
            buffer = [paragraph]
            current_length = len(paragraph)
            continue
        buffer.append(paragraph)
        current_length += len(paragraph)

    if buffer:
        blocks.append("\n".join(buffer))
    return blocks


def _has_strong_text_layer(layout_blocks: Sequence[LayoutBlock]) -> bool:
    """判断文字层是否足够强，值得优先用作正文来源。"""
    if not layout_blocks:
        return False
    total_length = sum(_alpha_numeric_length(block.content) for block in layout_blocks)
    strong_pages = {
        int(block.metadata.get("page_number") or 1)
        for block in layout_blocks
        if _alpha_numeric_length(block.content) >= 40
    }
    return total_length >= 120 and bool(strong_pages)


def _has_required_nltk_resources() -> bool:
    """检查当前环境是否具备 Unstructured 常用的 NLTK 资源。"""
    try:
        import nltk
    except ImportError:
        return False

    resource_paths = (
        "tokenizers/punkt_tab/english/",
        "taggers/averaged_perceptron_tagger_eng/",
    )
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            return False
    return True


def _supplement_layout_blocks_with_ocr(
    *,
    layout_blocks: list[LayoutBlock],
    source_path: Path,
    prefer_existing_text: bool,
) -> list[LayoutBlock]:
    """对缺页或文本极弱的页自动补 OCR，兼顾版面解析与可读性。"""
    total_pages = _pdf_page_count(source_path)
    blocks_by_page: dict[int, list[LayoutBlock]] = defaultdict(list)
    for block in layout_blocks:
        page_number = int(block.metadata.get("page_number") or 1)
        blocks_by_page[page_number].append(block)

    pages_needing_ocr: set[int] = set()
    for page_number in range(1, total_pages + 1):
        page_blocks = blocks_by_page.get(page_number, [])
        if not page_blocks:
            pages_needing_ocr.add(page_number)
            continue

        # 若 hi_res 只抽到极少碎片文字，也补一层 OCR，避免只拿到标题或页脚。
        visible_text = "".join(block.content for block in page_blocks if block.block_type == "text")
        if _alpha_numeric_length(visible_text) < 24:
            pages_needing_ocr.add(page_number)

    if not pages_needing_ocr:
        return layout_blocks

    LOGGER.info("Supplementing OCR for pages=%s file=%s", sorted(pages_needing_ocr), source_path)
    ocr_blocks = _ocr_fallback_layout_blocks(source_path)
    for block in ocr_blocks:
        page_number = int(block.metadata.get("page_number") or 1)
        if page_number in pages_needing_ocr:
            # 对于已经有可靠文字层的页，只在缺页/弱页场景下追加 OCR，而不覆盖原正文。
            if prefer_existing_text and any(
                existing_block.metadata.get("text_layer")
                for existing_block in blocks_by_page.get(page_number, [])
            ):
                blocks_by_page[page_number].append(block)
                continue
            blocks_by_page[page_number].append(block)

    merged_blocks: list[LayoutBlock] = []
    for page_number in range(1, total_pages + 1):
        merged_blocks.extend(blocks_by_page.get(page_number, []))
    return merged_blocks


def _pdf_page_count(source_path: Path) -> int:
    """读取 PDF 页数。"""
    from pypdf import PdfReader

    reader = PdfReader(str(source_path))
    return len(reader.pages)


def _alpha_numeric_length(text: str) -> int:
    """估计文本里有效字符数量。"""
    return sum(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in text)


def _configure_unstructured_local_models() -> None:
    """把 Unstructured 依赖的 HF 模型重定向到本地缓存，避免运行时联网探测。"""
    try:
        from unstructured_inference.models import tables
    except ImportError:
        return

    local_table_model = _resolve_local_hf_snapshot("microsoft/table-transformer-structure-recognition")
    if Path(local_table_model).exists():
        tables.DEFAULT_MODEL = local_table_model
        # 如果 table agent 还没初始化，让它后续直接走本地快照目录。
        tables_agent = getattr(tables, "tables_agent", None)
        if tables_agent is not None and getattr(tables_agent, "model", None) is None:
            LOGGER.info("Configured local table structure model path=%s", local_table_model)


def _resolve_local_hf_snapshot(model_name_or_path: str) -> str:
    """把 Hugging Face 仓库名优先解析为本地快照目录。"""
    raw_path = Path(model_name_or_path)
    if raw_path.exists():
        return str(raw_path)
    if "/" not in model_name_or_path:
        return model_name_or_path

    org_name, repo_name = model_name_or_path.split("/", 1)
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_cache_dir = cache_root / f"models--{org_name}--{repo_name}" / "snapshots"
    if not model_cache_dir.exists():
        return model_name_or_path
    snapshot_dirs = sorted(
        [path for path in model_cache_dir.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return str(snapshot_dirs[0]) if snapshot_dirs else model_name_or_path


def _select_ocr_languages() -> str:
    """自动选择当前系统可用的 OCR 语言包。"""
    available = _available_tesseract_languages()
    preferred = []
    for language in ("chi_sim", "chi_tra", "eng"):
        if language in available:
            preferred.append(language)
    if not preferred:
        preferred = ["eng"]
    if "chi_sim" not in preferred:
        LOGGER.warning(
            "Chinese OCR language pack `chi_sim` is not available. OCR quality for Chinese PDF pages may be degraded."
        )
    return "+".join(preferred)


def _available_tesseract_languages() -> set[str]:
    """读取系统里当前可用的 tesseract 语言包。"""
    try:
        completed = subprocess.run(
            ["tesseract", "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    languages = {
        line.strip()
        for line in completed.stdout.splitlines()[1:]
        if line.strip()
    }
    return languages


def _preprocess_image_for_ocr(image: Any) -> Any:
    """做一层简单的 OCR 图像增强，提高中文课件类 PDF 的识别稳定性。"""
    from PIL import ImageFilter, ImageOps

    grayscale = image.convert("L")
    enhanced = ImageOps.autocontrast(grayscale)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    # 二值化阈值略偏中间，兼顾 PPT 截图的浅色背景和深色文字。
    thresholded = sharpened.point(lambda pixel: 255 if pixel > 165 else 0)
    return thresholded


def _ocr_page_blocks(
    *,
    image: Any,
    image_to_data: Any,
    image_to_string: Any,
    languages: str,
) -> list[str]:
    """按页内块/行聚合 OCR 结果，避免把整页挤成一大段文本。"""
    config = "--oem 1 --psm 6"
    try:
        raw_tsv = image_to_data(image, lang=languages, config=config)
    except Exception:
        raw_tsv = ""

    blocks: dict[tuple[int, int], list[str]] = defaultdict(list)
    if raw_tsv:
        reader = csv.DictReader(raw_tsv.splitlines(), delimiter="\t")
        for row in reader:
            text = _clean_ocr_text(row.get("text", ""))
            if not text:
                continue
            confidence = _safe_float(row.get("conf"))
            # 过滤低置信度噪声块。
            if confidence is not None and confidence < 35:
                continue
            block_num = int(_safe_int(row.get("block_num"), 0))
            line_num = int(_safe_int(row.get("line_num"), 0))
            blocks[(block_num, line_num)].append(text)

    page_blocks = [
        _clean_ocr_text(" ".join(tokens))
        for _, tokens in sorted(blocks.items(), key=lambda item: item[0])
    ]
    page_blocks = [block for block in page_blocks if block]
    page_blocks = _merge_short_ocr_blocks(page_blocks)

    # 如果 `image_to_data` 没拿到足够稳定的块，再回退到整页 OCR。
    if not page_blocks:
        full_page_text = _clean_ocr_text(image_to_string(image, lang=languages, config=config))
        if full_page_text:
            page_blocks = [full_page_text]
    return page_blocks


def _merge_short_ocr_blocks(page_blocks: Sequence[str]) -> list[str]:
    """把过短 OCR 块合并成更有语义的段落，降低碎片噪声。"""
    merged_blocks: list[str] = []
    buffer: list[str] = []
    current_length = 0

    for block in page_blocks:
        block_length = _alpha_numeric_length(block)
        if buffer and current_length + block_length > 120:
            merged_blocks.append(_clean_ocr_text(" ".join(buffer)))
            buffer = [block]
            current_length = block_length
            continue

        buffer.append(block)
        current_length += block_length
        if current_length >= 28:
            merged_blocks.append(_clean_ocr_text(" ".join(buffer)))
            buffer = []
            current_length = 0

    if buffer:
        merged_blocks.append(_clean_ocr_text(" ".join(buffer)))

    return [block for block in merged_blocks if _alpha_numeric_length(block) >= 6]


def _load_partition_pdf() -> Any:
    """按需加载 Unstructured PDF 分区器。"""
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise ImportError(
            "PDF ingestion requires `unstructured[pdf]`. Install it before indexing PDF documents."
        ) from exc
    return partition_pdf


def _elements_to_layout_blocks(
    elements: Sequence[Any],
    source_path: Path,
    *,
    include_text: bool = True,
) -> list[LayoutBlock]:
    """把 Unstructured 元素转成保留版面语义的块。"""
    blocks: list[LayoutBlock] = []
    elements_by_page: dict[int, list[Any]] = defaultdict(list)
    for element in elements:
        page_number = int(getattr(getattr(element, "metadata", None), "page_number", 1) or 1)
        elements_by_page[page_number].append(element)

    for page_number, page_elements in sorted(elements_by_page.items(), key=lambda item: item[0]):
        text_buffer: list[str] = []
        for index, element in enumerate(page_elements):
            category = _element_category(element)
            text = _clean_layout_text(_element_text(element))
            if category in {"FigureCaption"}:
                # Caption 由图片块消费，不独立生成 chunk，避免重复噪声。
                continue

            if category == "Table":
                _flush_text_buffer(
                    text_buffer=text_buffer,
                    blocks=blocks,
                    source_path=source_path,
                    page_number=page_number,
                )
                table_html = getattr(getattr(element, "metadata", None), "text_as_html", None)
                blocks.append(
                    LayoutBlock(
                        content=text or _clean_layout_text(table_html or ""),
                        metadata={
                            "title": source_path.stem,
                            "url": "",
                            "record_id": f"{source_path.stem}:page:{page_number}:table:{index}",
                            "source": str(source_path),
                            "source_file": str(source_path),
                            "source_type": "pdf",
                            "page_number": page_number,
                            "block_type": "table",
                            "table_html": table_html,
                            "image_caption": None,
                            "citation_hint": _build_citation_hint(source_path.stem, page_number, "table"),
                            "tags": [],
                        },
                        block_type="table",
                    )
                )
                continue

            if category in {"Image", "Figure"}:
                _flush_text_buffer(
                    text_buffer=text_buffer,
                    blocks=blocks,
                    source_path=source_path,
                    page_number=page_number,
                )
                caption = _nearest_caption(page_elements, index)
                image_text = text or ""
                combined_content = _compose_image_block_text(image_text, caption)
                if _is_meaningful_image_block(combined_content, caption):
                    blocks.append(
                        LayoutBlock(
                            content=combined_content,
                            metadata={
                                "title": source_path.stem,
                                "url": "",
                                "record_id": f"{source_path.stem}:page:{page_number}:image:{index}",
                                "source": str(source_path),
                                "source_file": str(source_path),
                                "source_type": "pdf",
                                "page_number": page_number,
                                "block_type": "image",
                                "table_html": None,
                                "image_caption": caption,
                                "citation_hint": _build_citation_hint(source_path.stem, page_number, "figure"),
                                "tags": [],
                            },
                            block_type="image",
                        )
                    )
                continue

            if include_text and text:
                text_buffer.append(text)

        _flush_text_buffer(
            text_buffer=text_buffer,
            blocks=blocks,
            source_path=source_path,
            page_number=page_number,
        )

    return blocks


def _split_layout_block(
    block: LayoutBlock,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """按块类型切分，确保表格与图片说明不会被拆散。"""
    if not block.content.strip():
        return []

    if block.block_type in {"table", "image"}:
        return [Document(page_content=block.content, metadata=dict(block.metadata))]

    return splitter.create_documents(
        texts=[block.content],
        metadatas=[dict(block.metadata)],
    )


def dedupe_documents(documents: Sequence[Document]) -> list[Document]:
    """按内容、页码和来源去重，避免重复 chunk 污染索引。"""
    deduped: list[Document] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for document in documents:
        metadata = document.metadata
        dedupe_key = (
            _safe_text(metadata.get("title")),
            _safe_text(document.page_content),
            _safe_text(metadata.get("source_file")),
            _safe_text(metadata.get("block_type")),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(document)
    return deduped


def _flush_text_buffer(
    *,
    text_buffer: list[str],
    blocks: list[LayoutBlock],
    source_path: Path,
    page_number: int,
) -> None:
    """把同页普通文字缓冲区刷成一个文本块。"""
    if not text_buffer:
        return
    content = "\n".join(part for part in text_buffer if part).strip()
    if content:
        blocks.append(
            LayoutBlock(
                content=content,
                metadata={
                    "title": source_path.stem,
                    "url": "",
                    "record_id": f"{source_path.stem}:page:{page_number}:text:{len(blocks)}",
                    "source": str(source_path),
                    "source_file": str(source_path),
                    "source_type": "pdf",
                    "page_number": page_number,
                    "block_type": "text",
                    "table_html": None,
                    "image_caption": None,
                    "citation_hint": _build_citation_hint(source_path.stem, page_number, "text"),
                    "tags": [],
                },
                block_type="text",
            )
        )
    text_buffer.clear()


def _compose_prefixed_text(title: str, tags: Sequence[str], body: str) -> str:
    """给普通文本补标题和标签前缀。"""
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    if body:
        parts.append(body)
    return "\n".join(parts)


def _compose_image_block_text(image_text: str, caption: str | None) -> str:
    """生成图片块文本，使 Agent 至少能读到图片说明。"""
    parts: list[str] = []
    if image_text:
        parts.append(image_text)
    if caption:
        parts.append(f"Image Caption: {caption}")
    return "\n".join(part for part in parts if part).strip()


def _is_meaningful_image_block(image_text: str, caption: str | None) -> bool:
    """过滤掉误判出来的短噪声图片块。"""
    if caption and _alpha_numeric_length(caption) >= 6:
        return True
    return _alpha_numeric_length(image_text) >= 10


def _nearest_caption(page_elements: Sequence[Any], image_index: int) -> str | None:
    """寻找与图片最接近的 Caption。"""
    for offset in (1, -1, 2, -2):
        neighbor_index = image_index + offset
        if not 0 <= neighbor_index < len(page_elements):
            continue
        neighbor = page_elements[neighbor_index]
        if _element_category(neighbor) == "FigureCaption":
            text = _clean_layout_text(_element_text(neighbor))
            if text:
                return text
    return None


def _element_category(element: Any) -> str:
    """安全地拿到 Unstructured 元素类别。"""
    category = getattr(element, "category", None)
    if category:
        return str(category)
    return element.__class__.__name__


def _element_text(element: Any) -> str:
    """安全地拿到 Unstructured 元素文本。"""
    text = getattr(element, "text", None)
    if text:
        return str(text)
    metadata = getattr(element, "metadata", None)
    if metadata is not None:
        html_text = getattr(metadata, "text_as_html", None)
        if html_text:
            return str(html_text)
    return ""


def _build_citation_hint(title: str, page_number: int | None, block_type: str) -> str:
    """构造面向 Agent 的引用提示。"""
    if page_number is None:
        return f"该信息来自《{title}》的{block_type}内容。"
    block_label = {
        "table": "表格",
        "figure": "图片说明",
        "image": "图片说明",
        "text": "正文",
    }.get(block_type, block_type)
    return f"该信息位于《{title}》第 {page_number} 页的{block_label}中。"


def _safe_text(value: Any) -> str:
    """把任意可选字段转成可比较的字符串。"""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_tags(raw_tags: Any) -> list[str]:
    """标准化 tags 字段。"""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        candidates = re.split(r"[,;/|]", raw_tags)
        return [tag.strip() for tag in candidates if tag.strip()]
    if isinstance(raw_tags, Iterable):
        tags: list[str] = []
        for item in raw_tags:
            text = _safe_text(item)
            if text:
                tags.append(text)
        return tags
    text = _safe_text(raw_tags)
    return [text] if text else []


def _clean_layout_text(text: str) -> str:
    """简单清洗版面文本，去掉多余空白。"""
    return re.sub(r"\n{3,}", "\n\n", _safe_text(text))


def _clean_ocr_text(text: str) -> str:
    """清洗 OCR 文本中的常见噪声。"""
    cleaned = _safe_text(text)
    cleaned = cleaned.replace("|", " ").replace("¦", " ")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # 过滤掉几乎全是符号的噪声行。
    alpha_numeric_count = sum(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in cleaned)
    if alpha_numeric_count == 0:
        return ""
    return cleaned.strip()


def _safe_int(value: Any, default: int) -> int:
    """把 OCR 表格里的数字字段安全转成 int。"""
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_float(value: Any) -> float | None:
    """把 OCR 置信度字段安全转成 float。"""
    try:
        return float(str(value).strip())
    except Exception:
        return None
