"""Runtime environment helpers for mirrors, caches, and optional resources."""

from __future__ import annotations

import logging
import os
from pathlib import Path


LOGGER = logging.getLogger(__name__)
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_NLTK_PACKAGES = {
    "punkt_tab": "tokenizers/punkt_tab/english/",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng/",
}
_MISSING_NLTK_WARNING_EMITTED = False


def configure_runtime_environment(project_dir: str | Path) -> None:
    """配置运行时环境，使模型优先走镜像与本地缓存。"""
    project_dir = Path(project_dir).resolve()
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # 把 NLTK 数据目录固定到项目内，便于后续离线部署时统一管理。
    nltk_data_dir = project_dir / "data" / "nltk_data"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NLTK_DATA", str(nltk_data_dir))
    _configure_nltk_runtime(nltk_data_dir)

    LOGGER.info(
        "Runtime environment configured hf_endpoint=%s nltk_data=%s",
        os.environ.get("HF_ENDPOINT"),
        os.environ.get("NLTK_DATA"),
    )


def _configure_nltk_runtime(nltk_data_dir: Path) -> None:
    """确保 NLTK 能看到项目内资源目录，并在缺资源时自动下载。"""
    global _MISSING_NLTK_WARNING_EMITTED
    try:
        import nltk
    except ImportError:
        LOGGER.warning("NLTK is not installed; skipping NLTK runtime configuration.")
        return

    nltk_data_dir_str = str(nltk_data_dir)
    if nltk_data_dir_str not in nltk.data.path:
        # 直接把项目内路径插到最前，避免运行时优先命中系统里半残缺的数据目录。
        nltk.data.path.insert(0, nltk_data_dir_str)

    missing_packages: list[str] = []
    for package_name, resource_path in DEFAULT_NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing_packages.append(package_name)

    if not missing_packages:
        return

    auto_download_enabled = os.getenv("NLTK_AUTO_DOWNLOAD", "false").strip().lower() in {"1", "true", "yes", "on"}
    if auto_download_enabled:
        LOGGER.warning(
            "Missing NLTK resources detected: %s. Attempting auto-download into %s.",
            missing_packages,
            nltk_data_dir,
        )
        for package_name in missing_packages:
            try:
                nltk.download(package_name, download_dir=nltk_data_dir_str, quiet=True)
            except Exception as exc:
                LOGGER.warning("Failed to auto-download NLTK package %s: %s", package_name, exc)
        return

    # 默认不主动联网下载，避免在离线/弱网环境里每次启动都刷出一串下载失败日志。
    if not _MISSING_NLTK_WARNING_EMITTED:
        LOGGER.warning(
            "Missing NLTK resources detected: %s. Auto-download is disabled; PDF layout parsing will "
            "gracefully fall back when these resources are required. Set NLTK_AUTO_DOWNLOAD=true to enable.",
            missing_packages,
        )
        _MISSING_NLTK_WARNING_EMITTED = True
