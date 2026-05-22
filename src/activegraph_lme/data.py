from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LMEInstance:
    question_id: str
    question_type: str  # may end with _abs (abstention)
    question: str
    answer: str
    question_date: str
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[dict[str, Any]]]  # [[{role, content, has_answer?}, ...], ...]
    answer_session_ids: list[str]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs") or "_abs" in self.question_id


def load_dataset(path: str | Path) -> list[LMEInstance]:
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    out: list[LMEInstance] = []
    for entry in raw:
        out.append(
            LMEInstance(
                question_id=entry["question_id"],
                question_type=entry["question_type"],
                question=entry["question"],
                answer=entry["answer"],
                question_date=entry.get("question_date", ""),
                haystack_session_ids=entry.get("haystack_session_ids", []),
                haystack_dates=entry.get("haystack_dates", []),
                haystack_sessions=entry.get("haystack_sessions", []),
                answer_session_ids=entry.get("answer_session_ids", []),
            )
        )
    return out


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
