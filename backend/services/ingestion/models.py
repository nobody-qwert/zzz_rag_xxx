from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class QueuedBatchDoc:
    doc_path: Path
    doc_hash: str
    display_name: str
    job_record_id: Optional[str] = None


@dataclass
class QueuedBatchJob:
    job_id: str
    docs: List[QueuedBatchDoc]
