from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class MineruResult:
    text: str
    metadata: Dict[str, Any]


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    v = val.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def run_mineru(
    pdf_path: Path,
    out_dir: Path,
    *,
    parse_method: str | None = None,
    lang: str | None = None,
    table_enable: bool | None = None,
    formula_enable: bool | None = None,
) -> MineruResult:
    """Run MinerU on a single PDF with GPU detection and return Markdown text.

    This follows the pattern proven to work in ocr_bench.
    """
    try:
        from mineru.cli.common import do_parse, read_fn  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("mineru is not installed: pip install mineru") from exc

    # Resolve device mode: auto|cuda|cpu
    requested = (os.environ.get("MINERU_DEVICE_MODE") or "auto").strip().lower()
    effective = requested
    cuda_available = False
    gpu_name: Optional[str] = None
    gpu_mem_gb: Optional[float] = None
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        cuda_available = False

    if requested in (None, "", "auto"):
        effective = "cuda" if cuda_available else "cpu"
    elif requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "MINERU_DEVICE_MODE=cuda requested but CUDA not available. Install CUDA torch and run with GPU runtime."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_bytes = read_fn(pdf_path)
    pdf_name = pdf_path.stem

    # Defaults can be overridden by env or function args
    p_method = (parse_method or os.environ.get("MINERU_PARSE_METHOD") or "auto").strip().lower()
    p_lang = (lang or os.environ.get("MINERU_LANG") or "en").strip()
    p_table = table_enable if table_enable is not None else _env_bool("MINERU_TABLE_ENABLE", True)
    p_formula = formula_enable if formula_enable is not None else _env_bool("MINERU_FORMULA_ENABLE", True)

    do_parse(
        output_dir=str(out_dir),
        pdf_file_names=[pdf_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[p_lang],
        backend="pipeline",
        parse_method=p_method,
        formula_enable=p_formula,
        table_enable=p_table,
        start_page_id=0,
        end_page_id=None,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=False,
        device_mode=effective,
    )

    md_path = out_dir / pdf_name / parse_method / f"{pdf_name}.md"
    if not md_path.exists():
        raise RuntimeError(f"MinerU did not produce expected Markdown at {md_path}")

    text = md_path.read_text(encoding="utf-8")
    meta = {
        "markdown_path": str(md_path),
        "device_mode": effective,
        "gpu": {"available": cuda_available, "name": gpu_name, "mem_gb": gpu_mem_gb},
        "parse_method": p_method,
        "lang": p_lang,
        "table_enable": p_table,
        "formula_enable": p_formula,
    }

    # Write a small summary for debugging
    try:
        (out_dir / "mineru_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

    return MineruResult(text=text, metadata=meta)


def warmup_mineru(*,
                  parse_method: str | None = None,
                  lang: str | None = None,
                  table_enable: bool | None = False,
                  formula_enable: bool | None = False,
                  device_mode: str | None = None,
                  tmp_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Trigger MinerU model initialization and weight downloads.

    Generates a tiny in-memory PDF and runs a minimal parse with most dumps disabled.
    This reduces first-ingest latency in production.
    """
    try:
        from mineru.cli.common import do_parse  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("mineru is not installed: pip install mineru") from exc

    # Optionally override device mode for warmup
    if device_mode:
        os.environ["MINERU_DEVICE_MODE"] = device_mode

    # Resolve device mode: auto|cuda|cpu
    requested = (os.environ.get("MINERU_DEVICE_MODE") or "auto").strip().lower()
    effective = requested
    cuda_available = False
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if requested in (None, "", "auto"):
        effective = "cuda" if cuda_available else "cpu"
    elif requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "MINERU_DEVICE_MODE=cuda requested but CUDA not available. Install CUDA torch and run with GPU runtime."
        )

    # Create a tiny single-page PDF in memory using PyMuPDF
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF required for warmup") from exc

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "MinerU warmup")
    pdf_bytes = doc.tobytes()
    doc.close()

    out_base = Path(tmp_dir or os.environ.get("INDEX_DIR", "/indices")) / "_warmup"
    out_base.mkdir(parents=True, exist_ok=True)

    # Effective config
    p_method = (parse_method or os.environ.get("MINERU_PARSE_METHOD") or "auto").strip().lower()
    p_lang = (lang or os.environ.get("MINERU_LANG") or "en").strip()
    p_table = table_enable if table_enable is not None else _env_bool("MINERU_TABLE_ENABLE", True)
    p_formula = formula_enable if formula_enable is not None else _env_bool("MINERU_FORMULA_ENABLE", True)

    # Run a minimal parse over a single page
    do_parse(
        output_dir=str(out_base),
        pdf_file_names=["warmup"],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[p_lang],
        backend="pipeline",
        parse_method=p_method,
        formula_enable=p_formula,
        table_enable=p_table,
        start_page_id=0,
        end_page_id=1,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=False,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=False,
        device_mode=effective,
    )

    return {
        "ok": True,
        "device_mode": effective,
        "parse_method": p_method,
        "lang": p_lang,
        "table_enable": p_table,
        "formula_enable": p_formula,
        "output_dir": str(out_base),
    }
