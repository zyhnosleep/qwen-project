"""Additional tools for agentic RAG workflows."""

from __future__ import annotations

import os
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any

from smolagents import Tool, WebSearchTool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LibrarySqlConfig:
    """图书馆数据库工具的配置。"""

    db_url: str
    books_table: str = "books"
    inventory_table: str = "inventory"
    book_id_column: str = "id"
    inventory_book_id_column: str = "book_id"
    title_column: str = "title"
    isbn_column: str = "isbn"
    available_count_column: str = "available_count"
    total_count_column: str = "total_count"
    shelf_location_column: str = "shelf_location"
    status_column: str = "status"
    max_rows: int = 10


@dataclass(frozen=True)
class WebSearchProxyConfig:
    """Web 搜索专用代理配置。"""

    use_env_proxy: bool = False
    http_proxy: str = ""
    https_proxy: str = ""
    all_proxy: str = ""
    no_proxy: str = "127.0.0.1,localhost"


class LibrarySqlTool(Tool):
    """安全的图书馆实时库存查询工具。

    适合查询书籍余量、架位、在馆状态等结构化信息。工具只允许使用预设的
    SQL 模板，并通过参数绑定执行，避免 Agent 直接生成任意 SQL 带来的风险。
    如果查询失败，工具会返回结构化报错信息与建议，方便 Agent 进行自我修正。
    """

    name = "library_sql_tool"
    description = (
        "Query the live library inventory database for book availability, shelf location, "
        "and circulation status. Use this tool when the user asks about real-time stock, "
        "shelf position, or whether a specific book or ISBN is available now."
    )
    inputs = {
        "query_kind": {
            "type": "string",
            "description": (
                "One of: inventory_by_title, inventory_by_isbn, status_by_title, status_by_isbn. "
                "Leave empty to let the tool infer based on whether isbn is provided."
            ),
        },
        "book_name": {
            "type": "string",
            "description": "Book title keyword used in title-based inventory queries.",
        },
        "isbn": {
            "type": "string",
            "description": "ISBN or catalogue identifier used in exact inventory queries.",
        },
    }
    output_type = "object"
    output_schema = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "query_kind": {"type": "string"},
            "rows": {"type": "array"},
            "error": {"type": "object"},
            "retry_hint": {"type": "string"},
            "available_templates": {"type": "array"},
        },
    }

    def __init__(self, config: LibrarySqlConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._engine: Engine | None = None

    def setup(self) -> None:
        """初始化数据库连接。"""
        self._engine = create_engine(self._config.db_url, future=True)
        self.is_initialized = True

    def forward(
        self,
        query_kind: str,
        book_name: str,
        isbn: str,
    ) -> dict[str, Any]:
        """执行参数化 SQL 模板查询。"""
        assert self._engine is not None

        normalized_query_kind = self._resolve_query_kind(query_kind, book_name, isbn)
        available_templates = sorted(self._query_templates().keys())
        try:
            sql_template = self._query_templates()[normalized_query_kind]
        except KeyError:
            return {
                "success": False,
                "query_kind": normalized_query_kind,
                "rows": [],
                "error": {
                    "type": "UnsupportedQueryKind",
                    "message": f"Unsupported query_kind: {normalized_query_kind}",
                },
                "retry_hint": "请选择 inventory_by_title / inventory_by_isbn / status_by_title / status_by_isbn 之一。",
                "available_templates": available_templates,
            }

        params = self._build_params(normalized_query_kind, book_name, isbn)
        try:
            with self._engine.connect() as connection:
                result = connection.execute(text(sql_template), params)
                rows = [dict(row._mapping) for row in result.fetchall()]
            return {
                "success": True,
                "query_kind": normalized_query_kind,
                "rows": rows,
                "error": None,
                "retry_hint": "" if rows else "没有查到结果。可以尝试更准确的书名、ISBN，或切换另一种 query_kind。",
                "available_templates": available_templates,
            }
        except SQLAlchemyError as exc:
            # 这里返回详细报错和修正建议，让 Agent 有机会换模板或改参数再次调用。
            LOGGER.warning("LibrarySqlTool query failed: %s", exc)
            return {
                "success": False,
                "query_kind": normalized_query_kind,
                "rows": [],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "retry_hint": (
                    "SQL 查询执行失败。请检查 query_kind 是否匹配、参数是否为空，"
                    "或尝试改用 ISBN 精确查询 / 书名模糊查询。"
                ),
                "available_templates": available_templates,
            }

    def _resolve_query_kind(self, query_kind: str, book_name: str, isbn: str) -> str:
        """在 Agent 未显式指定 query_kind 时自动推断模板。"""
        query_kind = query_kind.strip()
        if query_kind:
            return query_kind
        if isbn.strip():
            return "inventory_by_isbn"
        if book_name.strip():
            return "inventory_by_title"
        return "inventory_by_title"

    def _build_params(self, query_kind: str, book_name: str, isbn: str) -> dict[str, Any]:
        """构造模板参数。"""
        if query_kind.endswith("_by_isbn"):
            return {
                "isbn": isbn.strip(),
                "limit": self._config.max_rows,
            }
        return {
            "book_name": f"%{book_name.strip()}%",
            "limit": self._config.max_rows,
        }

    def _query_templates(self) -> dict[str, str]:
        """生成可安全执行的参数化 SQL 模板。"""
        books_table = _safe_identifier(self._config.books_table)
        inventory_table = _safe_identifier(self._config.inventory_table)
        book_id_column = _safe_identifier(self._config.book_id_column)
        inventory_book_id_column = _safe_identifier(self._config.inventory_book_id_column)
        title_column = _safe_identifier(self._config.title_column)
        isbn_column = _safe_identifier(self._config.isbn_column)
        available_count_column = _safe_identifier(self._config.available_count_column)
        total_count_column = _safe_identifier(self._config.total_count_column)
        shelf_location_column = _safe_identifier(self._config.shelf_location_column)
        status_column = _safe_identifier(self._config.status_column)

        base_select = f"""
            SELECT
                b.{book_id_column} AS book_id,
                b.{title_column} AS title,
                b.{isbn_column} AS isbn,
                i.{available_count_column} AS available_count,
                i.{total_count_column} AS total_count,
                i.{shelf_location_column} AS shelf_location,
                i.{status_column} AS status
            FROM {books_table} AS b
            LEFT JOIN {inventory_table} AS i
              ON b.{book_id_column} = i.{inventory_book_id_column}
        """

        return {
            "inventory_by_title": f"""
                {base_select}
                WHERE lower(b.{title_column}) LIKE lower(:book_name)
                ORDER BY i.{available_count_column} DESC
                LIMIT :limit
            """,
            "inventory_by_isbn": f"""
                {base_select}
                WHERE b.{isbn_column} = :isbn
                ORDER BY i.{available_count_column} DESC
                LIMIT :limit
            """,
            "status_by_title": f"""
                {base_select}
                WHERE lower(b.{title_column}) LIKE lower(:book_name)
                ORDER BY b.{title_column} ASC
                LIMIT :limit
            """,
            "status_by_isbn": f"""
                {base_select}
                WHERE b.{isbn_column} = :isbn
                LIMIT :limit
            """,
        }


def _safe_identifier(identifier: str) -> str:
    """限制 SQL 标识符只能是安全字符，防止模板层注入。"""
    identifier = identifier.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return identifier


class ProxyAwareWebSearchTool(WebSearchTool):
    """为 WebSearchTool 单独绑定代理设置，不污染其他外部请求。"""

    def __init__(
        self,
        proxy_config: WebSearchProxyConfig,
        max_results: int = 10,
        engine: str = "duckduckgo",
    ) -> None:
        super().__init__(max_results=max_results, engine=engine)
        self._proxy_config = proxy_config
        self._env_lock = Lock()

    def forward(self, query: str) -> str:
        """在专用代理上下文里执行外网搜索。"""
        with self._proxy_environment():
            return super().forward(query)

    @contextmanager
    def _proxy_environment(self):
        """临时注入 web search 专用代理，执行完成后恢复现场。"""
        with self._env_lock:
            previous_values = {
                "http_proxy": os.environ.get("http_proxy"),
                "https_proxy": os.environ.get("https_proxy"),
                "all_proxy": os.environ.get("all_proxy"),
                "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
                "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
                "ALL_PROXY": os.environ.get("ALL_PROXY"),
                "NO_PROXY": os.environ.get("NO_PROXY"),
            }
            try:
                if self._proxy_config.use_env_proxy:
                    self._apply_proxy_config()
                else:
                    self._clear_proxy_env()
                yield
            finally:
                for key, value in previous_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def _apply_proxy_config(self) -> None:
        """把 web search 代理设置写入环境变量。"""
        proxy_pairs = {
            "http_proxy": self._proxy_config.http_proxy,
            "https_proxy": self._proxy_config.https_proxy or self._proxy_config.http_proxy,
            "all_proxy": self._proxy_config.all_proxy,
            "HTTP_PROXY": self._proxy_config.http_proxy,
            "HTTPS_PROXY": self._proxy_config.https_proxy or self._proxy_config.http_proxy,
            "ALL_PROXY": self._proxy_config.all_proxy,
            "NO_PROXY": self._proxy_config.no_proxy,
        }
        for key, value in proxy_pairs.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    @staticmethod
    def _clear_proxy_env() -> None:
        """清理代理环境变量，供不走代理的 web search 使用。"""
        for key in (
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        ):
            os.environ.pop(key, None)
