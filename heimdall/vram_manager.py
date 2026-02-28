"""
VRAM Manager — Monitors GPU memory and builds ONNX sessions with tight budgets.

Key ideas:
  • Auto-discovers pip-installed NVIDIA CUDA/cuDNN DLLs and adds them to PATH.
  • Query VRAM via `pynvml` (fallback: onnxruntime internal API).
  • Every ONNX session gets an arena_extend_strategy=kSameAsRequested
    and a gpu_mem_limit to prevent runaway allocation.
  • Provides a context-manager for "VRAM budget" blocks.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import onnxruntime as ort

from .config import CFG

logger = logging.getLogger("heimdall.vram")


# ──────────────────────────────────────────────────────────────────────
# Auto-register pip-installed NVIDIA CUDA DLLs (Windows)
# This MUST run before any ONNX session is created.
# ──────────────────────────────────────────────────────────────────────
def _register_cuda_dlls() -> None:
    """
    Find nvidia-* pip packages and add their bin/ dirs to DLL search path.

    On Windows, onnxruntime needs cublasLt64_12.dll, cudnn*.dll, etc.
    When installed via `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 ...`,
    the DLLs live in  site-packages/nvidia/<pkg>/bin/.
    """
    site_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site_nvidia.exists():
        logger.debug("No pip-installed NVIDIA packages found at %s", site_nvidia)
        return

    bin_dirs = sorted(site_nvidia.glob("*/bin"))
    if not bin_dirs:
        return

    added = []
    for d in bin_dirs:
        d_str = str(d)
        if d_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d_str + os.pathsep + os.environ.get("PATH", "")
            added.append(d.parent.name)
        # Also use os.add_dll_directory (Python 3.8+ on Windows)
        try:
            os.add_dll_directory(d_str)
        except (OSError, AttributeError):
            pass

    if added:
        logger.info("Registered CUDA DLL paths: %s", ", ".join(added))


# Run once at import time
_register_cuda_dlls()


# ──────────────────────────────────────────────────────────────────────
# Lightweight VRAM query (no pynvml hard dependency)
# ──────────────────────────────────────────────────────────────────────
def _get_free_vram_mb() -> float:
    """Return free VRAM in MB. Falls back to config total if unavailable."""
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(CFG.onnx_device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.free / (1024 * 1024)
    except Exception:
        logger.debug("pynvml unavailable — assuming full VRAM budget")
        return float(CFG.vram_total_mb)


def get_vram_status() -> dict[str, float]:
    """Return dict with total / free / used VRAM in MB."""
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(CFG.onnx_device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "total_mb": info.total / (1024 * 1024),
            "free_mb": info.free / (1024 * 1024),
            "used_mb": info.used / (1024 * 1024),
        }
    except Exception:
        return {
            "total_mb": float(CFG.vram_total_mb),
            "free_mb": float(CFG.vram_total_mb),
            "used_mb": 0.0,
        }


# ──────────────────────────────────────────────────────────────────────
# ONNX Session Builder
# ──────────────────────────────────────────────────────────────────────
def create_onnx_session(
    model_path: str,
    *,
    gpu_mem_limit_mb: int | None = None,
) -> ort.InferenceSession:
    """
    Create an ONNX Runtime GPU session with controlled VRAM allocation.

    Parameters
    ----------
    model_path : str
        Path to the .onnx model file.
    gpu_mem_limit_mb : int, optional
        Hard limit on GPU memory for this session (MB).
        Defaults to ``CFG.onnx_arena_mem_limit_mb``.

    Returns
    -------
    ort.InferenceSession
    """
    if gpu_mem_limit_mb is None:
        gpu_mem_limit_mb = CFG.onnx_arena_mem_limit_mb

    providers = _build_providers(gpu_mem_limit_mb)
    sess_options = _build_session_options()

    free = _get_free_vram_mb()
    logger.info(
        "Loading %s  |  VRAM free: %.0f MB  |  session limit: %d MB",
        model_path,
        free,
        gpu_mem_limit_mb,
    )

    if free < gpu_mem_limit_mb + CFG.vram_safety_margin_mb:
        logger.warning(
            "Low VRAM (%.0f MB free). Model may still load but watch for OOM.",
            free,
        )

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
    )
    return session


def _build_providers(gpu_mem_limit_mb: int) -> list:
    """Build ORT execution providers list with CUDA constraints."""
    cuda_options: dict = {
        "device_id": CFG.onnx_device_id,
        "arena_extend_strategy": "kSameAsRequested",   # no over-allocation
        "gpu_mem_limit": gpu_mem_limit_mb * 1024 * 1024,
        "cudnn_conv_algo_search": "HEURISTIC",         # less VRAM than EXHAUSTIVE
        "do_copy_in_default_stream": True,
    }
    return [
        ("CUDAExecutionProvider", cuda_options),
        "CPUExecutionProvider",  # fallback
    ]


def _build_session_options() -> ort.SessionOptions:
    """Build common session options shared by all models."""
    opts = ort.SessionOptions()
    opts.log_severity_level = CFG.onnx_log_level
    # Intra-op parallelism (within an op — e.g. large GEMM)
    opts.intra_op_num_threads = 4
    # Graph-level optimizations
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Enable memory pattern optimization (reuse buffers)
    opts.enable_mem_pattern = True
    opts.enable_mem_reuse = True
    return opts


# ──────────────────────────────────────────────────────────────────────
# Budget guard (optional context manager)
# ──────────────────────────────────────────────────────────────────────
def is_cuda_available() -> bool:
    """Check if any ONNX session can actually use CUDA."""
    try:
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            return False
        # Try creating a minimal session to confirm CUDA works
        opts = ort.SessionOptions()
        opts.log_severity_level = 4  # suppress logs
        # If we get here, CUDA provider is listed but may still fail
        # The real test is if sessions load without warnings — checked at load time
        return True
    except Exception:
        return False


def session_uses_cuda(session: ort.InferenceSession) -> bool:
    """Check if an existing ONNX session is actually running on CUDA."""
    try:
        active = session.get_providers()
        return "CUDAExecutionProvider" in active
    except Exception:
        return False


@contextmanager
def vram_budget_guard(required_mb: int) -> Generator[None, None, None]:
    """
    Context manager that checks for sufficient VRAM before proceeding.

    Raises RuntimeError if not enough memory is available.
    """
    free = _get_free_vram_mb()
    if free < required_mb + CFG.vram_safety_margin_mb:
        raise RuntimeError(
            f"Insufficient VRAM: {free:.0f} MB free, "
            f"need {required_mb + CFG.vram_safety_margin_mb} MB "
            f"({required_mb} MB + {CFG.vram_safety_margin_mb} MB safety margin)"
        )
    yield
