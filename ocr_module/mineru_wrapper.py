from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

import fitz  # type: ignore


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


_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\(([^)]+)\)")


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip():
            return int(value.strip())
    except (TypeError, ValueError):
        return None
    return None


def _extract_images_from_markdown(markdown_text: str, *, chunk: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not markdown_text:
        return entries
    for match in _IMAGE_PATTERN.finditer(markdown_text):
        src = (match.group(1) or "").strip()
        if not src or src.startswith("data:") or "://" in src:
            continue
        cleaned = src.lstrip("./")
        entries.append({"chunk": chunk, "path": cleaned})
    return entries


def _extract_images_from_content_list(content_data: Any, *, chunk: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def _walk(node: Any, context: Dict[str, Any]) -> None:
        if isinstance(node, dict):
            next_ctx = context
            for key in ("page", "page_id", "page_index", "page_number"):
                if key in node:
                    page_val = _coerce_int(node.get(key))
                    if page_val is not None:
                        next_ctx = {**context, "page": page_val}
                        break

            node_type = str(node.get("type") or node.get("category") or "").strip().lower()
            candidate_path: Optional[str] = None
            for key in ("image_path", "img_path", "img", "path", "value", "source_path"):
                val = node.get(key)
                if isinstance(val, str) and "images" in val:
                    candidate_path = val.strip()
                    break
            if candidate_path:
                cleaned = candidate_path.lstrip("./")
                entry: Dict[str, Any] = {
                    "chunk": chunk,
                    "page": next_ctx.get("page"),
                    "type": node_type or "image",
                    "path": cleaned,
                }
                if "bbox" in node:
                    entry["bbox"] = node["bbox"]
                elif "box" in node:
                    entry["bbox"] = node["box"]
                elif "position" in node:
                    entry["bbox"] = node["position"]
                if "width" in node:
                    entry["width"] = node.get("width")
                if "height" in node:
                    entry["height"] = node.get("height")
                results.append(entry)
            for value in node.values():
                _walk(value, next_ctx)
        elif isinstance(node, list):
            for item in node:
                _walk(item, context)
        elif isinstance(node, str):
            if "images" in node:
                cleaned = node.strip().lstrip("./")
                results.append({"chunk": chunk, "page": context.get("page"), "type": "image", "path": cleaned})

    _walk(content_data, {})
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for item in results:
        key = (item.get("path"), item.get("page"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _copy_chunk_images(chunk_images_dir: Path, final_images_dir: Path) -> List[str]:
    copied: List[str] = []
    if not chunk_images_dir.exists():
        return copied
    final_images_dir.mkdir(parents=True, exist_ok=True)
    for image_file in chunk_images_dir.iterdir():
        if not image_file.is_file():
            continue
        dest = final_images_dir / image_file.name
        if not dest.exists():
            shutil.copy2(image_file, dest)
        relative_path = f"images/{image_file.name}"
        copied.append(relative_path)
    return copied


def run_mineru(
    pdf_path: Path,
    out_dir: Path,
    *,
    parse_method: str | None = None,
    lang: str | None = None,
    table_enable: bool | None = None,
    formula_enable: bool | None = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MineruResult:
    """Run MinerU on a single PDF with GPU detection and return Markdown text.

    This follows the pattern proven to work in ocr_bench.
    """
    try:
        from mineru.cli.common import do_parse  # type: ignore
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

    pdf_name = pdf_path.stem

    # Defaults can be overridden by env or function args
    p_method = (parse_method or os.environ.get("MINERU_PARSE_METHOD") or "auto").strip().lower()
    p_lang = (lang or os.environ.get("MINERU_LANG") or "en").strip()
    p_table = table_enable if table_enable is not None else _env_bool("MINERU_TABLE_ENABLE", True)
    p_formula = formula_enable if formula_enable is not None else _env_bool("MINERU_FORMULA_ENABLE", True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    if total_pages == 0:
        raise RuntimeError("PDF contains no pages")

    chunk_pages = max(1, int(os.environ.get("MINERU_PROGRESS_CHUNK_PAGES", "4")))
    chunk_root = out_dir / "_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    combined_parts: List[str] = []
    chunk_records: List[Dict[str, Any]] = []
    all_image_entries: List[Dict[str, Any]] = []
    collected_images: List[str] = []
    collected_content_lists: List[str] = []

    final_dir = out_dir / pdf_name / p_method
    final_dir.mkdir(parents=True, exist_ok=True)
    final_images_dir = final_dir / "images"
    content_list_dir = final_dir / "content_lists"

    def emit(stage: str, processed: int) -> None:
        if not progress_cb:
            return
        total = total_pages
        percent = max(0.0, min(100.0, (processed / total) * 100.0))
        progress_cb({
            "stage": stage,
            "percent": percent,
            "current": processed,
            "total": total,
        })

    emit("parsing", 0)

    for chunk_index, start in enumerate(range(0, total_pages, chunk_pages)):
        end = min(total_pages, start + chunk_pages)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        chunk_bytes = chunk_doc.tobytes()
        chunk_doc.close()

        chunk_name = f"{pdf_name}_part_{chunk_index + 1}"
        chunk_dir = chunk_root / chunk_name
        chunk_dir.mkdir(parents=True, exist_ok=True)

        do_parse(
            output_dir=str(chunk_dir),
            pdf_file_names=[chunk_name],
            pdf_bytes_list=[chunk_bytes],
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
            f_dump_content_list=True,
            device_mode=effective,
        )

        chunk_md_path = chunk_dir / chunk_name / p_method / f"{chunk_name}.md"
        if not chunk_md_path.exists():
            raise RuntimeError(f"MinerU did not produce expected Markdown at {chunk_md_path}")
        chunk_markdown = chunk_md_path.read_text(encoding="utf-8")
        combined_parts.append(chunk_markdown)

        chunk_content_list_path = chunk_dir / chunk_name / p_method / "content_list.json"
        chunk_image_dir = chunk_dir / chunk_name / p_method / "images"

        chunk_image_paths = _copy_chunk_images(chunk_image_dir, final_images_dir)
        collected_images.extend(chunk_image_paths)

        chunk_content_entry: Optional[Path] = None
        if chunk_content_list_path.exists():
            content_list_dir.mkdir(parents=True, exist_ok=True)
            chunk_content_entry = content_list_dir / f"{chunk_name}.json"
            shutil.copy2(chunk_content_list_path, chunk_content_entry)
            collected_content_lists.append(f"content_lists/{chunk_content_entry.name}")
            try:
                loaded_content = json.loads(chunk_content_list_path.read_text(encoding="utf-8"))
                images_from_content = _extract_images_from_content_list(loaded_content, chunk=chunk_name)
                all_image_entries.extend(images_from_content)
            except Exception:
                pass
        else:
            markdown_images = _extract_images_from_markdown(chunk_markdown, chunk=chunk_name)
            all_image_entries.extend(markdown_images)

        chunk_records.append(
            {
                "chunk": chunk_name,
                "page_start": start + 1,
                "page_end": end,
                "content_list": f"content_lists/{chunk_content_entry.name}" if chunk_content_entry else None,
                "images_copied": len(chunk_image_paths),
            }
        )

        emit("parsing", end)
        shutil.rmtree(chunk_dir, ignore_errors=True)

    doc.close()

    md_path = final_dir / f"{pdf_name}.md"
    text = "\n\n".join(part.strip() for part in combined_parts if part.strip())
    if not text:
        raise RuntimeError("MinerU produced no text output")
    md_path.write_text(text, encoding="utf-8")
    shutil.rmtree(chunk_root, ignore_errors=True)
    meta = {
        "markdown_path": str(md_path),
        "device_mode": effective,
        "gpu": {"available": cuda_available, "name": gpu_name, "mem_gb": gpu_mem_gb},
        "parse_method": p_method,
        "lang": p_lang,
        "table_enable": p_table,
        "formula_enable": p_formula,
        "pages": total_pages,
        "chunk_pages": chunk_pages,
        "chunks": len(combined_parts),
        "assets": {
            "base_dir": str(final_dir),
            "images_dir": "images",
            "image_files": list(dict.fromkeys(collected_images)),
            "content_lists": collected_content_lists,
        },
        "image_blocks": all_image_entries,
        "chunk_records": chunk_records,
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

    default_base = Path(os.environ.get("INDEX_DIR", "/app_data/runtime")) / "_warmup"
    out_base = Path(tmp_dir) if tmp_dir is not None else default_base
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
