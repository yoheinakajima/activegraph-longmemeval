"""Stage 1 semantic-memory ActiveGraph system (no supersession).

Pipeline:
    ingest()
      1. Build the base turn graph via the package-native build_graph()
         (Turn + Session objects, precedes + cooccurrence relations).
      2. Register an @llm_behavior that reacts to one synthetic
         ``session.extract_request`` event per session and writes Fact
         objects + ``mentions`` relations through the same package
         ``ag.Graph``. Drive the reaction loop with
         ``runtime.run_until_idle()`` (single-threaded, FIFO).
      3. Facts are UNRESOLVED: contradictions coexist as separate Fact
         objects. Supersession (Stage 2) is intentionally out of scope.

    retrieve()
      Lexical IDF-overlap over the SAME pool that assemble() now iterates
      (turns + facts) — no separate "facts section". Same token budget as
      the deterministic systems so accuracy comparisons aren't confounded
      by context size.

Extractor / reader model identity
    The extractor @llm_behavior requests the SAME alias the reader
    requests (``claude-sonnet-4-5``); the API-resolved dated snapshot is
    captured per-call from the ``llm.responded`` event payload and stored
    in meta as ``extractor_model_resolved`` so a reviewer can confirm it
    matches (or doesn't match) the reader's resolved snapshot pinned in
    manifest.json. Extractor and reader intentionally share this alias;
    representational self-preference is a disclosed, uncontrolled
    residual (acceptable because the judge — not the reader — scores, so
    there is no scoring circularity). The extractor is non-GPT by
    construction (the judge is GPT-4o).

Run-identity invariants
    The reaction loop fires behaviors in registration order (FIFO,
    single-threaded). Replay is not reproducible without knowing which
    behaviors were registered, in what order; meta records
    ``behaviors_registered`` accordingly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import activegraph as ag
from activegraph.runtime.behavior_graph import BehaviorGraph
from pydantic import BaseModel, Field

from ..activegraph.graph import IngestState, Turn, build_graph
from ..activegraph.retrieve import (
    AssemblyResult,
    assemble,
    _tokenize_query,
)
from ..activegraph.stoplist import STOPLIST
from ..data import LMEInstance
from .base import AssembledContext


log = logging.getLogger(__name__)


_EXTRACT_REQUEST_TYPE = "session.extract_request"
_EXTRACTOR_BEHAVIOR_NAME = "sem_extract_facts_from_session"
_EXTRACTOR_BEHAVIOR_NAME_ASSISTANT = "sem_extract_facts_from_session_assistant"

# Each extractor behavior emits Facts authored by a different conversational
# role. The role is stamped onto every Fact's ``data["role"]`` and is part of
# the cache key (the same session text now yields two extraction results).
# user-centric facts answer "what did the USER tell us"; assistant-centric
# facts answer "what did the ASSISTANT previously say/recommend/compute" —
# the latter is what ss-assistant questions need and the original
# user-only extractor never produced.
_BEHAVIOR_ROLE = {
    _EXTRACTOR_BEHAVIOR_NAME: "user",
    _EXTRACTOR_BEHAVIOR_NAME_ASSISTANT: "assistant",
}
# Reverse map (role -> behavior name) for tagging cache-hit Fact writes
# with the actor that would have produced them on a live miss.
_ROLE_BEHAVIOR = {role: name for name, role in _BEHAVIOR_ROLE.items()}


# ---- LLM output schema ------------------------------------------------------


class _ExtractedFact(BaseModel):
    """One atomic fact the model believes the session establishes."""

    text: str = Field(
        ...,
        description=(
            "A single self-contained statement about the user or their world "
            "(<=160 chars). Do not include speculation, questions, or hedges."
        ),
    )
    mentioned_turn_idxs: list[int] = Field(
        default_factory=list,
        description=(
            "0-based indices of the turns in THIS session that establish the "
            "fact. Use [] only when the fact is implied by the session as a "
            "whole rather than any single turn."
        ),
    )


class _ExtractedFactList(BaseModel):
    facts: list[_ExtractedFact] = Field(default_factory=list)


# ---- extraction behavior (module-level so registration order is stable) ----


_EXTRACT_PROMPT_TEMPLATE = """\
{system}

Session under review:
{event}

Task:
{instruction}

Rules:
- Only extract facts that the session text directly supports.
- One claim per fact. Split compound claims.
- Extract ONLY facts about the user's world: their preferences,
  possessions, plans, identity, relationships, history, habits, opinions.
  Exclude facts that merely record the user asking the assistant a
  question, requesting help, or describe the topic of an
  assistant-provided answer. If the fact is only true because the user
  asked the assistant about something, exclude it. Include a fact only
  if it would still be true even if this conversation never happened.
  In particular, do NOT emit facts of the form "The user asked ...",
  "The user requested ...", "The user wanted to learn / know ...",
  "The user is learning how to ...", or "The user is interested in
  <topic the assistant explained>".
- Prefer durable facts about the user (preferences, identity, history,
  ongoing situations) over single-occasion small talk.
- Use neutral third-person phrasing ("The user ...").
- Do not emit two facts that express the same claim at different
  granularities; keep the most specific single version.
- If the session contains no extractable facts, return {{"facts": []}}.
"""


# Assistant-authored counterpart to _EXTRACT_PROMPT_TEMPLATE. Same envelope
# (system / event / instruction / Rules) and the SAME _ExtractedFactList
# schema, but it extracts what the ASSISTANT contributed rather than facts
# about the user. ss-assistant questions ("what did you recommend for X?",
# "what number did you give me?") are answered by these facts; the
# user-centric extractor emits nothing answer-bearing for them.
_EXTRACT_PROMPT_TEMPLATE_ASSISTANT = """\
{system}

Session under review:
{event}

Task:
{instruction}

Rules:
- Only extract facts that the session text directly supports.
- One claim per fact. Split compound claims.
- Extract ONLY facts about what the ASSISTANT contributed in this session:
  the things the assistant recommended, suggested, computed, calculated,
  defined, explained, named, listed, or told the user. These are facts
  about the assistant's OUTPUT, the content a later question might ask the
  assistant to recall ("what did you recommend / say / compute?").
- Use neutral third-person phrasing beginning with "The assistant ..."
  ("The assistant recommended ...", "The assistant computed ...",
  "The assistant told the user that ...", "The assistant suggested ...").
- Capture the SUBSTANCE the assistant provided, not the act of helping:
  prefer "The assistant recommended the Osprey Atmos 65 backpack" over
  "The assistant helped the user pick a backpack". Include the specific
  values/names/items so the fact is answer-bearing on its own.
- Do NOT extract facts about the user (their preferences, possessions,
  plans, identity) — the companion user-fact extractor covers those.
- Do not emit two facts that express the same claim at different
  granularities; keep the most specific single version.
- If the assistant contributed no substantive recommendation, computation,
  or statement of fact, return {{"facts": []}}.
"""


def _write_facts_to_graph(
    bgraph: Any,
    parsed: _ExtractedFactList,
    *,
    session_id: str,
    session_date: str,
    session_idx: int,
    turn_object_ids: list[str],
    role: str,
) -> None:
    """Write Fact + mentions edges for one session's extraction result.

    Shared by both the live @llm_behavior handlers (on a cache miss,
    after the LLM call) and the in-memory session-extraction cache
    (on a cache hit, skipping the LLM call). Identical output by
    construction — both paths feed the same `parsed` shape into the
    same writes, so a cached hit produces byte-identical Fact ids and
    mentions edges to a live miss for the same session text.

    ``role`` ("user" | "assistant") is the authorship of the facts in
    ``parsed`` (which behavior produced them). It is stamped onto every
    Fact's ``data["role"]`` AND mixed into the fact-id hash so a user
    fact and an assistant fact that happen to share text in the same
    session get distinct ids (they are distinct memories).

    Determinism for replay: fact ids are content-hashed
    (``fact:<sha256(session_id|role|text)[:16]>``) so the same session
    text re-extracted into the same (role, text) yields the same
    fact_id — regardless of whether the parsed value came from the
    live API or the cache.
    """
    n_turns = len(turn_object_ids)
    for fact in parsed.facts:
        text = (fact.text or "").strip()
        if not text:
            continue
        h = hashlib.sha256(
            f"{session_id}|{role}|{text}".encode("utf-8")
        ).hexdigest()[:16]
        fact_id = f"fact:{h}"

        fact_obj = bgraph.add_object(
            "Fact",
            {
                "fact_id": fact_id,
                "text": text,
                "session_id": session_id,
                "session_date": session_date,
                "session_idx": session_idx,
                "role": role,
                "source": "llm_extract",
            },
        )

        for idx in fact.mentioned_turn_idxs:
            if not isinstance(idx, int) or idx < 0 or idx >= n_turns:
                continue
            bgraph.add_relation(fact_obj.id, turn_object_ids[idx], "mentions")


@ag.llm_behavior(
    name=_EXTRACTOR_BEHAVIOR_NAME,
    on=[_EXTRACT_REQUEST_TYPE],
    output_schema=_ExtractedFactList,
    # model=None -> runtime resolves to the configured provider's
    # default_model at registration time. AnthropicProvider's default is
    # "claude-sonnet-4-5" (the SAME alias the reader uses), per the
    # package's resolution rules. We do NOT hardcode a dated snapshot;
    # the actually-served snapshot is pulled from the llm.responded event
    # payload after run_until_idle().
    temperature=0.0,
    max_tokens=2048,
    timeout_seconds=60.0,
    description=(
        "Extract atomic facts from a single haystack session. Emits one "
        "Fact object per claim with a stable content-hash id and "
        "`mentions` relations to the supporting turns."
    ),
    prompt_template=_EXTRACT_PROMPT_TEMPLATE,
)
def _sem_extract_handler(
    event: ag.Event,
    bgraph: Any,
    ctx: Any,
    parsed: _ExtractedFactList,
) -> None:
    """Live-API path (USER facts): invoked by the runtime on a cache miss
    after the LLM has produced a parsed _ExtractedFactList. All
    Fact-writing is delegated to :func:`_write_facts_to_graph` so the
    cache-hit path (which skips the LLM but still must produce identical
    writes) calls exactly the same code.
    """
    payload = event.payload or {}
    _write_facts_to_graph(
        bgraph,
        parsed,
        session_id=str(payload.get("session_id", "")),
        session_date=str(payload.get("session_date", "")),
        session_idx=int(payload.get("session_idx", 0)),
        turn_object_ids=list(payload.get("turn_object_ids", [])),
        role="user",
    )


@ag.llm_behavior(
    name=_EXTRACTOR_BEHAVIOR_NAME_ASSISTANT,
    on=[_EXTRACT_REQUEST_TYPE],
    output_schema=_ExtractedFactList,
    # Same alias / temperature / schema as the user extractor — the only
    # difference is the prompt template (assistant-centric) and the role
    # stamped on the resulting facts. Registered AFTER the user behavior so
    # the FIFO single-threaded reaction loop fires user-then-assistant
    # deterministically for each extract_request.
    temperature=0.0,
    max_tokens=2048,
    timeout_seconds=60.0,
    description=(
        "Extract atomic ASSISTANT-authored facts from a single haystack "
        "session (what the assistant recommended / computed / told the "
        "user). Emits one Fact object per claim with a stable content-hash "
        "id and `mentions` relations to the supporting turns."
    ),
    prompt_template=_EXTRACT_PROMPT_TEMPLATE_ASSISTANT,
)
def _sem_extract_handler_assistant(
    event: ag.Event,
    bgraph: Any,
    ctx: Any,
    parsed: _ExtractedFactList,
) -> None:
    """Live-API path (ASSISTANT facts). Mirror of :func:`_sem_extract_handler`
    but stamps ``role="assistant"`` on every written Fact."""
    payload = event.payload or {}
    _write_facts_to_graph(
        bgraph,
        parsed,
        session_id=str(payload.get("session_id", "")),
        session_date=str(payload.get("session_date", "")),
        session_idx=int(payload.get("session_idx", 0)),
        turn_object_ids=list(payload.get("turn_object_ids", [])),
        role="assistant",
    )


def _compute_prompt_sha256(prompt_template: str) -> str:
    """Full sha256 hex of the extraction prompt template.

    Used at two places: (1) stamped into the cache manifest at build
    time, and (2) recomputed at cache-load time and compared against
    the manifest to refuse stale caches built under a different
    prompt. Truncated form (first 16 hex chars) is the `extractor_signature`
    we expose in per-ingest meta — same input, different presentation.
    """
    return hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()


def _compute_combined_prompt_sha256() -> str:
    """Canonical extractor signature over the FULL behavior set (both the
    user and assistant prompt templates, in registration order).

    This is the value stamped into a cache manifest and re-checked at load
    time. Adding the assistant behavior changes this signature, so the
    committed seed-A (built under the user-only signature) will be REFUSED
    by the manifest guard under the new code — exactly the intended
    invalidation. seed-A-v2 is built under, and pinned to, this combined
    signature.
    """
    blob = (
        _EXTRACT_PROMPT_TEMPLATE
        + "\x00"
        + _EXTRACT_PROMPT_TEMPLATE_ASSISTANT
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """sha256 of a file's bytes, hex-encoded. Used to stamp the cache
    file's checksum into the manifest + the CHECKSUMS.sha256 sidecar
    so the committed artifact's integrity is verifiable."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- persistent extraction cache --------------------------------------------


class CacheManifestMismatchError(RuntimeError):
    """Raised when a persistent extraction cache's manifest does not
    match the current prompt template + extractor model alias.

    A committed cache must never silently serve facts from an old
    prompt or a different model — that would corrupt the experiment
    by mixing extractions from two different generators. The guard
    fires at cache-load time (system construction) so the failure is
    visible before any API spend happens.
    """


class _PersistentExtractionCache:
    """JSONL-backed per-session extraction cache with a sidecar manifest.

    File layout under ``cache_dir``::

        seed-{seed}.jsonl              # one JSON object per line:
                                       #   {"session_id", "content_sha256", "parsed"}
        seed-{seed}.manifest.json      # provenance + invalidation guard
        CHECKSUMS.sha256               # sha256sum-format integrity record

    Lookup key
        (session_id, content_sha256) — full sha256 of the session text
        the prompt is rendered over. The prompt template and extractor
        model alias are NOT in the per-entry key; they're stamped on
        the manifest as global invariants and checked at load time.
        That lets the file format stay flat and the per-entry payload
        match the runtime's parsed schema 1:1.

    Invalidation guard (CONTRACT)
        On load, ``prompt_sha256`` and ``extractor_model_requested``
        from the manifest must match the current
        ``_EXTRACT_PROMPT_TEMPLATE`` and the system's configured model
        alias respectively. If either differs we raise
        :class:`CacheManifestMismatchError` with a message naming the
        mismatch — never silently serve facts produced under a stale
        prompt or a different model.

    Write-through semantics
        ``put()`` appends to the JSONL file synchronously and updates
        the in-memory dict. The manifest's ``n_entries`` /
        ``cache_file_sha256`` / ``extractor_model_resolved`` are
        re-stamped via :meth:`flush_manifest`, which the system calls
        at the end of each ingest (after the resolved snapshot has
        been pulled from the post-run event log).

    A cache hit produces byte-identical Fact and ``mentions`` writes
    to a live miss (the parsed value's schema is identical), so the
    replay/inspectability contract still holds: every cached Fact has
    a real ``object.created`` event in ``graph.events`` — the LLM
    lifecycle prefix (``llm.requested``/``llm.responded``) is what
    differs.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        seed: str,
        prompt_sha256: str,
        extractor_model_requested: str,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.prompt_sha256 = prompt_sha256
        self.extractor_model_requested = extractor_model_requested
        self.cache_path = self.cache_dir / f"seed-{seed}.jsonl"
        self.manifest_path = self.cache_dir / f"seed-{seed}.manifest.json"
        self.checksums_path = self.cache_dir / "CHECKSUMS.sha256"

        # In-memory hot dict — keys = (session_id, content_sha256, role).
        self._entries: dict[tuple[str, str, str], _ExtractedFactList] = {}
        # Resolved snapshot the API actually served, captured lazily
        # on the first miss within the lifetime of this cache (or
        # pulled from the manifest on load). All subsequent puts must
        # carry the same resolved snapshot; mid-cache drift raises.
        self.extractor_model_resolved: str | None = None
        # Track newly-added entries this process appended, for stats.
        self._n_appended_this_process: int = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_or_init()

    # ---- lifecycle ----------------------------------------------------------

    def _load_or_init(self) -> None:
        """Read the manifest + JSONL into memory. Refuse to serve a
        cache whose manifest was stamped under a different prompt or
        model alias — see :class:`CacheManifestMismatchError`.

        Tolerated states:
          - Neither exists       → fresh cache (manifest stamped on flush).
          - Both exist           → validate manifest, load entries.
          - Cache only           → PARTIAL BUILD. A parallel build under
                                   ACTIVEGRAPH_SEM_EXTRACT_CACHE_NO_FLUSH=1
                                   produces a JSONL of harvested entries
                                   but defers the manifest stamp to the
                                   parent process. Treat this as a
                                   resumable build: load the entries as
                                   trusted (they came from this process
                                   tree under the same prompt + alias)
                                   and let the next flush write the
                                   manifest.
          - Manifest only        → real error; refuse.
        """
        manifest_exists = self.manifest_path.exists()
        cache_exists = self.cache_path.exists()
        if manifest_exists and not cache_exists:
            raise CacheManifestMismatchError(
                f"manifest at {self.manifest_path} exists but cache_file "
                f"{self.cache_path} does not. Delete the manifest or restore "
                f"the cache file."
            )
        if not manifest_exists:
            if cache_exists:
                # Partial build: jsonl exists from a parallel build that
                # hasn't reached the parent's final flush. Load entries
                # (trusted: same code path emitted them) so subsequent
                # ingest()/put() / flush_manifest() calls can resume.
                with open(self.cache_path) as f:
                    for line_no, raw in enumerate(f, start=1):
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                            sid = str(obj["session_id"])
                            csum = str(obj["content_sha256"])
                            role = str(obj.get("role", "user"))
                            parsed = _ExtractedFactList.model_validate(obj["parsed"])
                        except Exception as e:  # noqa: BLE001
                            raise CacheManifestMismatchError(
                                f"corrupt entry at {self.cache_path}:{line_no}: {e!r}"
                            ) from e
                        self._entries[(sid, csum, role)] = parsed
            # Fresh or partial: manifest gets written on first flush.
            return

        with open(self.manifest_path) as f:
            manifest = json.load(f)
        man_prompt = str(manifest.get("prompt_sha256", ""))
        man_alias = str(manifest.get("extractor_model_requested", ""))
        if man_prompt != self.prompt_sha256:
            raise CacheManifestMismatchError(
                f"extraction cache at {self.cache_path} was built under a "
                f"DIFFERENT prompt (manifest prompt_sha256={man_prompt!r}, "
                f"current prompt_sha256={self.prompt_sha256!r}). Refusing "
                f"to serve stale facts. Regenerate the cache against the "
                f"current prompt, or point at a cache built for it."
            )
        if man_alias != self.extractor_model_requested:
            raise CacheManifestMismatchError(
                f"extraction cache at {self.cache_path} was built with a "
                f"DIFFERENT extractor model "
                f"(manifest extractor_model_requested={man_alias!r}, "
                f"current={self.extractor_model_requested!r}). Refusing "
                f"to serve facts produced by a different model. Regenerate "
                f"or point at the right cache."
            )
        self.extractor_model_resolved = manifest.get("extractor_model_resolved")

        # Load entries. Lines are JSON objects; tolerate trailing
        # newlines but reject malformed lines loudly so corruption
        # surfaces at load time, not on a downstream KeyError.
        with open(self.cache_path) as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    sid = str(obj["session_id"])
                    csum = str(obj["content_sha256"])
                    role = str(obj.get("role", "user"))
                    parsed = _ExtractedFactList.model_validate(obj["parsed"])
                except Exception as e:  # noqa: BLE001 — surface corruption
                    raise CacheManifestMismatchError(
                        f"corrupt entry at {self.cache_path}:{line_no}: {e!r}"
                    ) from e
                self._entries[(sid, csum, role)] = parsed

    # ---- read ----

    def get(
        self, session_id: str, content_sha256: str, role: str
    ) -> _ExtractedFactList | None:
        return self._entries.get((session_id, content_sha256, role))

    # ---- write ----

    def put(
        self,
        session_id: str,
        content_sha256: str,
        role: str,
        parsed: _ExtractedFactList,
        *,
        extractor_model_resolved: str | None,
    ) -> None:
        """Append a new entry (write-through) and update the
        in-memory hot dict. No-op if the key is already present
        (idempotent on re-extracts within a single process).

        The key now includes ``role`` so the user-fact and assistant-fact
        extraction results for the SAME session text are stored as two
        distinct entries.

        Append uses ``fcntl.flock`` for cross-process safety so a
        parallel build (multiple worker processes appending to the
        same JSONL) cannot interleave bytes mid-line. POSIX O_APPEND
        is atomic only up to PIPE_BUF (~4KB); a fact-list line can
        exceed that, so we lock.

        If ``extractor_model_resolved`` is provided and the cache
        has already pinned a different snapshot, raise — the cache
        must reflect ONE resolved snapshot end-to-end.
        """
        import fcntl

        key = (session_id, content_sha256, role)
        if key in self._entries:
            return
        if extractor_model_resolved:
            if self.extractor_model_resolved is None:
                self.extractor_model_resolved = extractor_model_resolved
            elif self.extractor_model_resolved != extractor_model_resolved:
                raise CacheManifestMismatchError(
                    f"extractor served multiple resolved snapshots within "
                    f"one cache lifetime: previously pinned "
                    f"{self.extractor_model_resolved!r}, now seeing "
                    f"{extractor_model_resolved!r}. Refusing to mix "
                    f"snapshots in a single cache file."
                )
        self._entries[key] = parsed
        line = json.dumps(
            {
                "session_id": session_id,
                "content_sha256": content_sha256,
                "role": role,
                "parsed": parsed.model_dump(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        with open(self.cache_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        self._n_appended_this_process += 1

    def flush_manifest(self) -> None:
        """(Re)write the manifest and CHECKSUMS sidecar to reflect
        the current cache file. Idempotent. Call after each ingest
        so the manifest's ``n_entries`` / ``cache_file_sha256`` /
        ``extractor_model_resolved`` stay in sync with the JSONL.

        Suppress per-ingest flush by setting
        ``ACTIVEGRAPH_SEM_EXTRACT_CACHE_NO_FLUSH=1``. Used by parallel
        builders where N worker processes append to the same JSONL —
        the parent process does the single final flush after all
        workers exit so the manifest isn't races to rewrite.
        """
        if os.environ.get("ACTIVEGRAPH_SEM_EXTRACT_CACHE_NO_FLUSH") == "1":
            return
        if not self.cache_path.exists():
            # Nothing to stamp yet (no misses written).
            return
        manifest = {
            "seed": self.seed,
            "prompt_sha256": self.prompt_sha256,
            "extractor_model_requested": self.extractor_model_requested,
            "extractor_model_resolved": self.extractor_model_resolved,
            "n_entries": len(self._entries),
            "cache_file": self.cache_path.name,
            "cache_file_sha256": _sha256_file(self.cache_path),
            "created_at": (
                self._created_at_or_now()
                if not self.manifest_path.exists()
                else self._existing_manifest_created_at()
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        # sha256sum-compatible format so `sha256sum -c CHECKSUMS.sha256`
        # works from data/sem_extract_cache/ for byte-integrity checks.
        # Only seed-A (the committed canonical artifact) is recorded
        # here — seed-B/C are gitignored variance samples and would
        # clobber the seed-A line if they wrote here too. Their JSONL
        # sha256 is still stamped into their own manifest.json.
        if self.seed == "A":
            with open(self.checksums_path, "w") as f:
                f.write(f"{manifest['cache_file_sha256']}  {self.cache_path.name}\n")

    def _created_at_or_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _existing_manifest_created_at(self) -> str:
        try:
            with open(self.manifest_path) as f:
                return str(json.load(f).get("created_at", self._created_at_or_now()))
        except Exception:  # noqa: BLE001 — defensive
            return self._created_at_or_now()

    # ---- stats ----

    def __len__(self) -> int:
        return len(self._entries)

    def n_appended_this_process(self) -> int:
        return self._n_appended_this_process


# ---- system state -----------------------------------------------------------


@dataclass
class _SemState:
    state: IngestState
    meta: dict[str, Any] = field(default_factory=dict)


# ---- system -----------------------------------------------------------------


class ActiveGraphSemExtractSystem:
    """Stage 1 semantic-memory ActiveGraph (no supersession).

    Public name: ``activegraph-sem-extract``.
    """

    name = "activegraph-sem-extract"

    def __init__(
        self,
        *,
        token_budget: int,
        min_token_length: int,
        min_session_cooccurrence: int,
        max_doc_freq_fraction: float,
        extractor_model: str = "claude-sonnet-4-5",
        extraction_cache_enabled: bool | None = None,
        extract_seed: str = "A-v2",
        extraction_cache_dir: str | Path | None = None,
    ) -> None:
        self.token_budget = token_budget
        self.min_token_length = min_token_length
        self.min_session_cooccurrence = min_session_cooccurrence
        self.max_doc_freq_fraction = max_doc_freq_fraction
        # The alias requested at extraction call time. The runtime sends
        # this to AnthropicProvider; the dated snapshot the API serves
        # is what we record into meta.extractor_model_resolved.
        self.extractor_model = extractor_model

        # Persistent per-session extraction cache. The cache file
        # (data/sem_extract_cache/seed-{A,B,C}.jsonl) is the canonical
        # frozen experiment input — seed-A is committed to the repo,
        # seed-B/C are gitignored variance samples. The system reads
        # the file at construction time and validates its manifest
        # (prompt_sha256 + extractor_model_requested) against the
        # current prompt template and configured alias; a mismatch
        # raises CacheManifestMismatchError before any API spend.
        #
        # Override the cache state via:
        #   ACTIVEGRAPH_SEM_EXTRACT_CACHE=0  -> disable persistence and
        #                                       run cacheless (in-memory
        #                                       only; useful for variance
        #                                       measurement or guard
        #                                       tests).
        #   extract_seed="B"                 -> point at seed-B file.
        #   extraction_cache_dir="..."       -> override the cache root
        #                                       (default: data/sem_extract_cache).
        if extraction_cache_enabled is None:
            extraction_cache_enabled = (
                os.environ.get("ACTIVEGRAPH_SEM_EXTRACT_CACHE", "1") != "0"
            )
        self._extraction_cache_enabled: bool = extraction_cache_enabled
        self._extract_seed: str = extract_seed
        self._extraction_cache_dir: Path = Path(
            extraction_cache_dir or "data/sem_extract_cache"
        )

        # Per-prompt content hash; doubles as the manifest's
        # invalidation key. Now computed over BOTH extractor prompts
        # (user + assistant) so adding the assistant behavior invalidates
        # any cache built under the user-only signature. Truncated form
        # (first 16 hex) goes into per-ingest meta as `extractor_signature`.
        self._prompt_sha256: str = _compute_combined_prompt_sha256()
        self._extractor_signature: str = self._prompt_sha256[:16]

        self._cache: _PersistentExtractionCache | None = None
        if self._extraction_cache_enabled:
            # Load (and validate) on construction. A mismatch raises
            # CacheManifestMismatchError here — before any LLM call —
            # so a reviewer sees the staleness immediately. Loads are
            # cheap (parse a few MB of JSONL) and idempotent across
            # ingests within one CLI invocation.
            self._cache = _PersistentExtractionCache(
                cache_dir=self._extraction_cache_dir,
                seed=self._extract_seed,
                prompt_sha256=self._prompt_sha256,
                extractor_model_requested=self.extractor_model,
            )

        # In-memory fallback when persistent cache is disabled
        # (variance/guard testing). Mirrors the persistent cache's
        # in-memory hot dict so the rest of ingest() doesn't need to
        # branch on which kind of cache is active. Keyed by
        # (session_id, content_sha256, role) like the persistent cache.
        self._memory_only_cache: dict[
            tuple[str, str, str], _ExtractedFactList
        ] = {}

        # Cumulative counters across all ingests on this instance
        # (per-ingest counts also go into each ingest's meta).
        self._cum_sessions_total: int = 0
        self._cum_sessions_extracted: int = 0
        self._cum_cache_hits: int = 0

    # ---- ingest -------------------------------------------------------------

    def ingest(self, instance: LMEInstance) -> _SemState:
        ingest_state = build_graph(
            instance.haystack_session_ids,
            instance.haystack_dates,
            instance.haystack_sessions,
            min_token_length=self.min_token_length,
            min_session_cooccurrence=self.min_session_cooccurrence,
            max_doc_freq_fraction=self.max_doc_freq_fraction,
        )

        # Behavior registration order is a run-identity invariant — the
        # package's reaction loop is FIFO single-threaded and behaviors
        # fire in registration order. Record it so a reviewer can verify
        # replay would re-emit events identically.
        behaviors = [_sem_extract_handler, _sem_extract_handler_assistant]
        behaviors_registered = [
            {"name": b.name, "on": list(b.on), "model": b.model}
            for b in behaviors
        ]

        llm_provider = _build_llm_provider(self.extractor_model)

        runtime = ag.Runtime(
            ingest_state.graph,
            behaviors=behaviors,
            llm_provider=llm_provider,
            seed=0,
        )
        # Replace the IngestState's runtime stub with the live one so
        # downstream replay/debug walks the same instance.
        ingest_state.runtime = runtime

        # Partition sessions into cache hits / misses BEFORE driving
        # the reaction loop. Misses emit a session.extract_request the
        # behavior fires on (live LLM call). Hits skip the emit (and so
        # skip the LLM call entirely) — we write their Facts directly
        # after run_until_idle, using the same _write_facts_to_graph
        # helper the live handler uses, so the per-session graph
        # mutations are identical to a live miss for the same session.
        sessions_by_id = self._group_sessions(ingest_state, instance)
        cache_hit_writes: list[tuple[dict, _ExtractedFactList, str]] = []
        # Map of miss event_id -> (session_id, content_sha256). Each miss
        # emits ONE extract_request that BOTH behaviors (user + assistant)
        # react to; harvesting derives the role from the responding
        # behavior name (see _harvest_misses_into_cache).
        miss_event_id_to_key: dict[str, tuple[str, str]] = {}
        n_sessions_total = 0
        n_cache_hits = 0
        n_sessions_extracted = 0

        def _lookup(
            sid_: str, csum_: str, role_: str
        ) -> _ExtractedFactList | None:
            if self._cache is not None:
                return self._cache.get(sid_, csum_, role_)
            if self._extraction_cache_enabled is False:
                return None
            return self._memory_only_cache.get((sid_, csum_, role_))

        for sid, sdate, s_idx in zip(
            instance.haystack_session_ids,
            instance.haystack_dates,
            range(len(instance.haystack_session_ids)),
        ):
            turn_views = sessions_by_id[sid]
            session_text = "\n".join(
                f"[turn {tv.turn_idx}] {tv.role}: {tv.content}" for tv in turn_views
            )
            n_sessions_total += 1

            content_sha256 = hashlib.sha256(
                session_text.encode("utf-8")
            ).hexdigest()
            cache_key = (sid, content_sha256)

            payload = {
                "session_id": sid,
                "session_date": sdate,
                "session_idx": s_idx,
                "session_text": session_text,
                "turn_object_ids": [tv.object_id for tv in turn_views],
                "n_turns": len(turn_views),
            }

            # BOTH roles must be cached for this session to skip the LLM.
            cached_by_role = {
                role: _lookup(sid, content_sha256, role)
                for role in ("user", "assistant")
            }
            if all(v is not None for v in cached_by_role.values()):
                # FULL CACHE HIT (both roles): skip the emit entirely so
                # NEITHER behavior fires. Defer Fact-writing per role until
                # after run_until_idle so the event log shows live misses
                # first (preserving FIFO order), then hits.
                n_cache_hits += 1
                for role, cached in cached_by_role.items():
                    cache_hit_writes.append((payload, cached, role))
            else:
                # MISS (one or both roles uncached): emit ONE event; both
                # behaviors run and produce both roles' facts live. A
                # partial-cache session re-extracts both roles — only
                # possible during an incremental build, never during the
                # all-hit replay the committed cache is built for.
                n_sessions_extracted += 1
                evt = ag.Event(
                    id=ingest_state.graph.ids.event(),
                    type=_EXTRACT_REQUEST_TYPE,
                    payload=payload,
                    actor="sem_extract_ingest",
                )
                miss_event_id_to_key[evt.id] = cache_key
                ingest_state.graph.emit(evt)

        # Drive the package's reaction loop. If extraction can't reach
        # the LLM (e.g. ANTHROPIC_API_KEY unset in this environment) the
        # behavior emits behavior.failed events and run_until_idle still
        # returns; we surface that to the caller via meta.n_facts == 0.
        try:
            runtime.run_until_idle()
        except Exception as exc:  # noqa: BLE001 — surface so harness can stop
            log.error("Extractor runtime.run_until_idle() raised: %r", exc)
            raise

        # Walk the post-run event log to pull the resolved extractor
        # snapshot (the dated model the API actually served). Needed
        # both for cache persistence (the manifest pins this) and for
        # the per-ingest meta record.
        extractor_resolved = _resolve_extractor_snapshot(ingest_state.graph)

        # Harvest miss results from the event log into the cache, then
        # write Facts for hits (using the same helper as the live
        # handler — identical graph mutations by construction).
        if miss_event_id_to_key:
            self._harvest_misses_into_cache(
                ingest_state.graph,
                miss_event_id_to_key,
                extractor_resolved,
            )

        # CACHE-HIT WRITES: still emit real object.created /
        # relation.created events through a BehaviorGraph tagged with
        # the extractor's actor name. The LLM call is skipped — the
        # graph mutations are not. This preserves the package's
        # replay/inspectability contract: every Fact in the graph has
        # a real object.created event sitting in graph.events.
        for payload, cached, role in cache_hit_writes:
            hit_bgraph = BehaviorGraph(
                ingest_state.graph,
                actor=_ROLE_BEHAVIOR.get(role, _EXTRACTOR_BEHAVIOR_NAME),
                caused_by=None,
                frame_id=None,
            )
            _write_facts_to_graph(
                hit_bgraph,
                cached,
                session_id=str(payload["session_id"]),
                session_date=str(payload["session_date"]),
                session_idx=int(payload["session_idx"]),
                turn_object_ids=list(payload["turn_object_ids"]),
                role=role,
            )

        # Update cumulative counters on the system instance.
        self._cum_sessions_total += n_sessions_total
        self._cum_sessions_extracted += n_sessions_extracted
        self._cum_cache_hits += n_cache_hits

        # Re-stamp the manifest + CHECKSUMS sidecar so the on-disk
        # cache_file_sha256 / n_entries / extractor_model_resolved
        # stay in sync with the JSONL file we've been appending to.
        # Cheap: rewrites two small files.
        if self._cache is not None and n_sessions_extracted > 0:
            self._cache.flush_manifest()

        n_facts = sum(1 for _ in ingest_state.graph.objects(type="Fact"))
        n_facts_user = sum(
            1 for o in ingest_state.graph.objects(type="Fact")
            if (o.data or {}).get("role") == "user"
        )
        n_facts_assistant = sum(
            1 for o in ingest_state.graph.objects(type="Fact")
            if (o.data or {}).get("role") == "assistant"
        )
        n_fact_events = sum(
            1 for e in ingest_state.graph.events if e.type == "object.created"
            and e.payload.get("object", {}).get("type") == "Fact"
        )
        n_mentions = len(ingest_state.graph.relations(type="mentions"))
        n_behavior_failed = sum(
            1 for e in ingest_state.graph.events if e.type == "behavior.failed"
        )

        meta = {
            **ingest_state.stats(),
            "n_facts": n_facts,
            "n_facts_user": n_facts_user,
            "n_facts_assistant": n_facts_assistant,
            "n_fact_events": n_fact_events,
            "n_mentions_edges": n_mentions,
            "n_behavior_failed": n_behavior_failed,
            "extractor_model_requested": self.extractor_model,
            "extractor_model_resolved": extractor_resolved,
            "behaviors_registered": behaviors_registered,
            # Session-extraction cache visibility. Per-ingest counts plus
            # cumulative-on-this-system-instance so a reviewer can
            # confirm `n_sessions_extracted + n_cache_hits == n_sessions_total`
            # and see the savings as the cache warms across questions.
            "extraction_cache_enabled": self._extraction_cache_enabled,
            "extractor_signature": self._extractor_signature,
            "prompt_sha256": self._prompt_sha256,
            "extract_seed": self._extract_seed,
            "extraction_cache_path": (
                str(self._cache.cache_path) if self._cache is not None else None
            ),
            "n_cache_entries_loaded_at_init": (
                len(self._cache) - self._cache.n_appended_this_process()
                if self._cache is not None else 0
            ),
            "n_cache_appended_this_process": (
                self._cache.n_appended_this_process()
                if self._cache is not None else 0
            ),
            "n_sessions_total": n_sessions_total,
            "n_sessions_extracted": n_sessions_extracted,
            "n_cache_hits": n_cache_hits,
            "cum_sessions_total": self._cum_sessions_total,
            "cum_sessions_extracted": self._cum_sessions_extracted,
            "cum_cache_hits": self._cum_cache_hits,
        }
        return _SemState(state=ingest_state, meta=meta)

    # ---- cache harvest ------------------------------------------------------

    def _harvest_misses_into_cache(
        self,
        graph: ag.Graph,
        miss_event_id_to_key: dict[str, tuple[str, str]],
        extractor_resolved: str | None,
    ) -> None:
        """Walk the post-run event log and persist each cache miss's
        parsed LLM output so the next time the same session text
        appears (within this run OR across `cli run` invocations,
        once the file is committed) it hits the cache instead of the
        API.

        Event chain (per the runtime's @llm_behavior invocation path):
          extract_request  ← caused_by ←  llm.requested  ← caused_by ←  llm.responded
        We look up llm.responded events for EITHER extractor behavior,
        walk back via ``caused_by`` two hops, and pull the parsed payload.
        The role is derived from which behavior responded so the user and
        assistant results for one extract_request land under distinct
        (session_id, content_sha256, role) keys.
        """
        events_by_id: dict[str, Any] = {e.id: e for e in graph.events}
        for ev in graph.events:
            if ev.type != "llm.responded":
                continue
            behavior_name = ev.payload.get("behavior")
            role = _BEHAVIOR_ROLE.get(behavior_name)
            if role is None:
                continue
            req = events_by_id.get(ev.caused_by) if ev.caused_by else None
            if req is None or req.type != "llm.requested":
                continue
            extract_req = events_by_id.get(req.caused_by) if req.caused_by else None
            if extract_req is None or extract_req.type != _EXTRACT_REQUEST_TYPE:
                continue
            cache_key = miss_event_id_to_key.get(extract_req.id)
            if cache_key is None:
                continue
            parsed_payload = ev.payload.get("parsed")
            if parsed_payload is None:
                continue
            try:
                parsed = _ExtractedFactList.model_validate(parsed_payload)
            except Exception:  # noqa: BLE001 — defensive; bad parse means no caching
                continue
            sid, csum = cache_key
            if self._cache is not None:
                self._cache.put(
                    sid,
                    csum,
                    role,
                    parsed,
                    extractor_model_resolved=extractor_resolved,
                )
            else:
                # Persistent cache disabled; fall back to the in-process
                # memory dict (variance/guard testing path).
                self._memory_only_cache[(sid, csum, role)] = parsed

    # ---- retrieve -----------------------------------------------------------

    def retrieve(
        self, state: _SemState, question: str, question_date: str
    ) -> AssembledContext:
        scores = _score_units_lexical(
            state.state, question, min_token_length=self.min_token_length
        )
        res: AssemblyResult = assemble(
            state.state, scores, token_budget=self.token_budget
        )
        # Partition selected ids by kind so the manifest meta surfaces both.
        selected_turn_ids = [uid for uid in res.selected_unit_ids if "#" in uid and not uid.startswith("fact:")]
        selected_fact_ids = [uid for uid in res.selected_unit_ids if uid.startswith("fact:")]
        meta = {
            **state.meta,
            "n_selected_turns": len(selected_turn_ids),
            "n_selected_facts": len(selected_fact_ids),
            "n_seeds": res.n_seeds,
            "n_temporal_expansions": res.n_expanded,
            "retrieval_signal": "lexical",
            "token_budget": self.token_budget,
        }
        return AssembledContext(text=res.text, truncated=res.truncated, meta=meta)

    # ---- helpers ------------------------------------------------------------

    def _group_sessions(
        self, state: IngestState, instance: LMEInstance
    ) -> dict[str, list[Turn]]:
        groups: dict[str, list[Turn]] = {sid: [] for sid in instance.haystack_session_ids}
        for t in state.turns:
            groups.setdefault(t.session_id, []).append(t)
        for sid in groups:
            groups[sid].sort(key=lambda tv: tv.turn_idx)
        return groups


# ---- scoring ----------------------------------------------------------------


def _score_units_lexical(
    state: IngestState, question: str, min_token_length: int
) -> dict[str, float]:
    """IDF-overlap score for every unit in the pool (turns + facts).

    Uses the existing turn-derived ``state.vocab_df`` for IDF weights so
    facts and turns share one yardstick. Facts get tokenized on the fly
    from their stored ``text``; turns reuse the cached ``turn_tokens``.
    Tokens outside the pruned vocab contribute 0 to both, exactly like
    score_lexical() does for turns alone.
    """
    q_tokens = _tokenize_query(question, min_token_length)
    n_turns = max(1, len(state.turns))
    idf: dict[str, float] = {}
    for tok in q_tokens:
        df = state.vocab_df.get(tok)
        if df is None:
            continue
        idf[tok] = math.log((n_turns + 1) / (df + 1)) + 1.0

    scores: dict[str, float] = {}
    for t in state.turns:
        toks = state.turn_tokens.get(t.turn_id, ())
        if not toks or not idf:
            scores[t.turn_id] = 0.0
            continue
        s = 0.0
        for tok in toks:
            w = idf.get(tok)
            if w is not None:
                s += w
        scores[t.turn_id] = s

    for obj in state.graph.objects(type="Fact"):
        data = obj.data or {}
        fact_id = str(data.get("fact_id") or obj.id)
        text = str(data.get("text", ""))
        if not text or not idf:
            scores[fact_id] = 0.0
            continue
        toks = _tokenize_query(text, min_token_length)
        s = 0.0
        for tok in toks:
            w = idf.get(tok)
            if w is not None:
                s += w
        scores[fact_id] = s

    return scores


# ---- provider + resolved-snapshot capture -----------------------------------


def _build_llm_provider(alias: str) -> Any:
    """Construct the AnthropicProvider the runtime uses to drive
    ``@llm_behavior`` calls.

    The alias is recorded on the behavior at registration time; the
    actually-served dated snapshot is captured per-call by
    :func:`_resolve_extractor_snapshot` after run_until_idle.
    """
    from activegraph.llm import AnthropicProvider

    if alias.lower().startswith("gpt") or "openai" in alias.lower():
        raise ValueError(
            f"Extractor must be non-GPT (the judge is GPT-4o); got alias={alias!r}."
        )
    return AnthropicProvider()


def _resolve_extractor_snapshot(graph: ag.Graph) -> str | None:
    """Pull the dated model snapshot the Anthropic API actually served
    from the most recent ``llm.responded`` event emitted by the
    extractor behavior.

    Returns ``None`` if no extractor LLM call was made (e.g. provider
    misconfigured, all behaviors failed). All ``llm.responded`` events
    emitted by the extractor must report the same resolved snapshot
    within a single run — we assert that here so a mid-run model
    rotation can't silently slip in.
    """
    snapshots: set[str] = set()
    for ev in graph.events:
        if ev.type != "llm.responded":
            continue
        # Filter to events caused by EITHER extractor behavior. Both the
        # actor and the behavior/behavior_name in the payload identify it.
        names = {
            ev.actor,
            ev.payload.get("behavior"),
            ev.payload.get("behavior_name"),
        }
        if names.isdisjoint(_BEHAVIOR_ROLE.keys()):
            continue
        model = ev.payload.get("model")
        if model:
            snapshots.add(str(model))
    if not snapshots:
        return None
    if len(snapshots) > 1:
        raise RuntimeError(
            "Extractor served multiple resolved snapshots in one ingest "
            f"({sorted(snapshots)!r}); refusing to record ambiguous provenance."
        )
    return next(iter(snapshots))
