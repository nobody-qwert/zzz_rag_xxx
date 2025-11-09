import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from mineru_wrapper import run_mineru, warmup_mineru

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


DATA_DIR = Path(os.environ.get("DATA_DIR", "/app_data/docs"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/app_data/runtime"))
OCR_OUTPUT_DIR = Path(os.environ.get("OCR_OUTPUT_DIR") or (INDEX_DIR / "ocr_outputs"))
OCR_INCOMING_DIR = Path(os.environ.get("OCR_INCOMING_DIR") or (INDEX_DIR / "incoming"))
OCR_WARMUP_DIR = Path(os.environ.get("OCR_WARMUP_DIR") or (INDEX_DIR / "warmup"))
for path in (DATA_DIR, INDEX_DIR, OCR_OUTPUT_DIR, OCR_INCOMING_DIR, OCR_WARMUP_DIR):
    path.mkdir(parents=True, exist_ok=True)

MINERU_PARSE_METHOD = (os.environ.get("MINERU_PARSE_METHOD") or "auto").strip().lower()
MINERU_LANG = (os.environ.get("MINERU_LANG") or "en").strip()
MINERU_TABLE_ENABLE = _env_bool("MINERU_TABLE_ENABLE", True)
MINERU_FORMULA_ENABLE = _env_bool("MINERU_FORMULA_ENABLE", True)
MINERU_WARMUP_ON_STARTUP = _env_bool("MINERU_WARMUP_ON_STARTUP", False)


class ParseResponse(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MINERU_WARMUP_ON_STARTUP:
        async def _do_warmup() -> None:
            try:
                info = await asyncio.to_thread(
                    warmup_mineru,
                    parse_method=MINERU_PARSE_METHOD,
                    lang=MINERU_LANG,
                    table_enable=MINERU_TABLE_ENABLE,
                    formula_enable=MINERU_FORMULA_ENABLE,
                    tmp_dir=OCR_WARMUP_DIR,
                )
                logger.info("MinerU warmup finished: %s", info)
            except Exception as exc:
                logger.warning("MinerU warmup failed: %s", exc)

        asyncio.create_task(_do_warmup())
    yield


app = FastAPI(title="MinerU OCR Module", version="0.1.0", lifespan=lifespan)


async def _persist_upload(doc_hash: str, upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".bin"
    token = uuid4().hex
    dest_path = OCR_INCOMING_DIR / f"{doc_hash}_{token}{suffix}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest_path.open("wb") as out_file:
            while True:
                chunk = await upload.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
    except Exception:
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return dest_path


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
async def parse_document(
    doc_hash: str = Form(...),
    file: UploadFile = File(...),
    parse_method: Optional[str] = Form(None),
    lang: Optional[str] = Form(None),
    table_enable: Optional[bool] = Form(None),
    formula_enable: Optional[bool] = Form(None),
) -> ParseResponse:
    try:
        pdf_path = await _persist_upload(doc_hash, file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {exc}") from exc

    selected_parse_method = (parse_method or MINERU_PARSE_METHOD or "auto").strip().lower()
    selected_lang = (lang or MINERU_LANG or "en").strip()
    selected_table = table_enable if table_enable is not None else MINERU_TABLE_ENABLE
    selected_formula = formula_enable if formula_enable is not None else MINERU_FORMULA_ENABLE
    out_dir = OCR_OUTPUT_DIR / doc_hash

    try:
        result = await asyncio.to_thread(
            run_mineru,
            pdf_path,
            out_dir,
            parse_method=selected_parse_method,
            lang=selected_lang,
            table_enable=selected_table,
            formula_enable=selected_formula,
        )
    except Exception as exc:
        logger.exception("MinerU parse failed for %s", pdf_path)
        raise HTTPException(status_code=500, detail=f"MinerU parsing failed: {exc}") from exc
    finally:
        pdf_path.unlink(missing_ok=True)

    return ParseResponse(text=result.text, metadata=result.metadata)


@app.post("/warmup")
async def warmup_route() -> Dict[str, Any]:
    try:
        info = await asyncio.to_thread(
            warmup_mineru,
            parse_method=MINERU_PARSE_METHOD,
            lang=MINERU_LANG,
            table_enable=MINERU_TABLE_ENABLE,
            formula_enable=MINERU_FORMULA_ENABLE,
            tmp_dir=OCR_WARMUP_DIR,
        )
        return {"warmup_complete": True, **info}
    except Exception as exc:
        logger.exception("MinerU warmup failed")
        raise HTTPException(status_code=500, detail=f"MinerU warmup failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
