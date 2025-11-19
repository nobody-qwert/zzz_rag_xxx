from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class GPUPhaseManager:
    """Coordinate which heavy GPU service (LLM or OCR) owns memory."""

    def __init__(
        self,
        llm_control_url: Optional[str],
        ocr_control_url: Optional[str],
        *,
        timeout: float = 30.0,
    ) -> None:
        self._llm_url = (llm_control_url or "").strip() or None
        self._ocr_url = (ocr_control_url or "").strip() or None
        self._timeout = max(1.0, float(timeout))
        self._lock = asyncio.Lock()
        self._state: str = "llm"
        self._last_error: Optional[str] = None

    async def switch_to_ocr(self, reason: Optional[str] = None) -> None:
        await self._transition(target="ocr", reason=reason)

    async def switch_to_llm(self, reason: Optional[str] = None) -> None:
        await self._transition(target="llm", reason=reason)

    async def ensure_llm_ready(self) -> None:
        await self.switch_to_llm(reason="ensure_llm_ready")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "llm_control_url": self._llm_url,
            "ocr_control_url": self._ocr_url,
            "last_error": self._last_error,
        }

    async def _transition(self, target: str, reason: Optional[str]) -> None:
        if target not in {"llm", "ocr"}:
            raise ValueError(f"Unknown GPU phase target: {target}")
        urls_missing = self._llm_url is None or self._ocr_url is None
        if urls_missing:
            # No-op when control URLs are not configured
            self._state = target
            return
        async with self._lock:
            if self._state == target:
                # Refresh to ensure service is still alive
                await self._refresh_current(target)
                return
            logger.info("Switching GPU phase to %s (reason=%s)", target, reason)
            try:
                if target == "ocr":
                    await self._call_control(self._llm_url, "unload")
                    await self._call_control(self._ocr_url, "load")
                else:
                    await self._call_control(self._ocr_url, "unload")
                    await self._call_control(self._llm_url, "load")
                self._state = target
                self._last_error = None
            except Exception as exc:  # pragma: no cover - network errors
                self._last_error = str(exc)
                logger.error("GPU phase transition to %s failed: %s", target, exc)
                raise

    async def _refresh_current(self, target: str) -> None:
        try:
            if target == "ocr" and self._ocr_url:
                await self._call_control(self._ocr_url, "status", method="GET", ignore_missing=True)
            elif target == "llm" and self._llm_url:
                await self._call_control(self._llm_url, "status", method="GET", ignore_missing=True)
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("GPU phase refresh failed (%s): %s", target, exc)

    async def _call_control(
        self,
        base_url: str,
        action: str,
        *,
        method: str = "POST",
        ignore_missing: bool = False,
    ) -> None:
        url = f"{base_url.rstrip('/')}/{action.strip('/')}"
        timeout = httpx.Timeout(self._timeout, connect=min(5.0, self._timeout))
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                if method.upper() == "GET":
                    resp = await client.get(url)
                else:
                    resp = await client.post(url)
                resp.raise_for_status()
            except Exception:
                if ignore_missing:
                    raise
                raise
