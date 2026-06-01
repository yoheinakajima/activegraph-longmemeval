# Compile From the Log, Don't Replace It

### Hybrid Semantic Memory on LongMemEval-S: Stage-1 Ablation, Role-Aware Extraction, and a Synthesis With the Substrate

*Yohei Nakajima — May 2026*

*Technical report: an ablation on the ActiveGraph substrate. Naive LLM extraction regresses 16pp vs. the deterministic baseline. A hybrid design — LLM-extracted facts as a typed index over the event log, with provenance-anchored source turns reaching the reader — matches the substrate's accuracy and closes retrieval headroom on temporal-reasoning and knowledge-update. A role-aware extraction fix recovers the single-session-assistant category (0/5 → 4/5) and lifts overall accuracy to 0.92, with a single identifiable tradeoff: budget competition between user and assistant fact pools at compact context. Smoke-scoped (n=50); full-500 validation is the natural next step.*

---

## 1. Motivation and prior result

A previous post on this blog ([*Evidence Compilation Before Semantic Memory: ActiveGraph on LongMemEval-S*](https://activegraph.ai/blog/evidence-compilation-before-semantic-memory-longmemeval)) measured the deterministic retrieval substrate of [ActiveGraph](https://docs.activegraph.ai) on LongMemEval-S. Headline: **85.6% QA accuracy and 86.2% turn-level answer-in-context at 2,462 mean context tokens**, statistically tied with dense turn-RAG at matched budget, with no LLM-generated memory at ingest. The substrate works.

That paper deliberately deferred the obvious next question: **once the substrate is established, does adding LLM-generated semantic memory on top of it help?** It made one narrow prediction, conditioned on its own retrieval-side measurements:

- *Knowledge-update*: retrieval was near ceiling (≈95% turn-AIC, ≈99% session-AIC). The remaining failures were reader reconciliation, not retrieval. A semantic layer should help only if it *marks facts as superseded before assembly*.
- *Temporal-reasoning and multi-session*: there was real turn-level retrieval headroom (76.0% and 95.9% turn-AIC respectively). A typed semantic layer that links cross-session entities or normalizes dates before retrieval should help by surfacing the exact gold turns more often.

Concurrently, a different ActiveGraph harness, evaluated independently in [*Semantic Memory Forgot the Assistant: A 500-Question LongMemEval-S Debugging Report*](https://activegraph.ai/blog/semantic-memory-forgot-the-assistant), reported a specific failure mode of naive LLM extraction: when the writer's prompt is user-centric, the system retrieves nothing useful for questions about the *assistant's* prior outputs. That harness measured a *role-aware retention fix*: single-session-assistant accuracy moved 75.0% → 98.2% (+13 of 56, p = 0.0010); overall accuracy moved 83.4% → 84.8%. The mechanism is representational, not retrieval-algorithmic: facts derived from assistant-authored turns must be written under a different type before retrieval can surface them.

This post tests both threads against the post-1 substrate, on the same harness, at the same context budget, in a controlled ablation. The unit of comparison is the full evidence-compilation pipeline — indexing, scoring, expansion, packing, ordering, and rendering — held identical across systems except for how semantic facts (if any) participate.

The result, in one sentence: **semantic projections are most useful as a typed index over a deterministic event log, with provenance back to source spans, not as a replacement for the log.** The hybrid design that compiles back to source turns recovers the substrate's accuracy at matched budget, extends it on the categories post 1 predicted had headroom, and after a role-aware fix recovers the single-session-assistant category that naive extraction-based retrieval drops to zero.

The full-500 validation is deferred. Smoke n=50 is the right unit for an ablation that asks *which architecture* rather than *what is the headline number*, but it is not a leaderboard claim.

---

## 2. What is being compared

LongMemEval-S, cleaned `s` split, n = 500 questions in total. This experiment uses the published 50-question smoke subset committed in `config/smoke_ids.txt` of the `activegraph-longmemeval` harness. Across those 50 questions, the haystack consists of **2,345 unique (session_id, content_sha256) pairs**, with each question's haystack averaging ~48 sessions.

Fixed across all systems:

- **Reader**: Claude Sonnet 4.5 (`claude-sonnet-4-5`, resolved snapshot `claude-sonnet-4-5-20250929`), temperature 0.
- **Judge**: gpt-4o-2024-08-06, temperature 0. Same judge across all five systems compared here (a *cleaner* comparison than the deterministic-vs-LLM comparison reported in [the assistant-retention post](https://activegraph.ai/blog/semantic-memory-forgot-the-assistant), where the deterministic baseline used a different snapshot).
- **Embedding model**: text-embedding-3-small.
- **Retrieval budget**: 2,500 tokens (matched to post 1's compact regime). This is part of the claim.
- **Dataset seed**: 42.
- **Significance**: all comparisons are paired per-question; exact McNemar would apply if making leaderboard claims, but at smoke n=50 the cell counts are too small to be the primary support — per-type discordant *counts* are reported throughout.

### Systems

The five systems differ only in the memory-writer behavior at ingest and in how facts (if any) participate in assembly. All retrieval scoring is embedding-cosine over the relevant unit pool; all final context is rendered in chronological order under the 2,500-token budget.

| System | Memory writer | Retrieval pool | Reader sees |
|---|---|---|---|
| `activegraph-det-embedding` | none (raw turns) | Turn | scored turns |
| `activegraph-sem-extract` | LLM extraction (user-prompted, v1) | Fact | facts only |
| `activegraph-sem-index` | LLM extraction (user-prompted, v1) | Fact (scored) | provenance-anchored source turns only |
| `activegraph-sem-hybrid` (v1) | LLM extraction (user-prompted, v1) | Fact (scored) | fact-as-header + provenance-anchored source turn |
| `activegraph-sem-hybrid` (v2, role-aware) | LLM extraction (user-prompted + assistant-prompted) | Fact (scored, both roles) | fact-as-header + provenance-anchored source turn |

`sem-index` and `sem-hybrid` share the retrieval signal (the fact embedding score) and the provenance edges (`Fact → Turn` via `mentions`); they differ only in whether facts themselves appear in the rendered reader context. This isolates the question of whether the extracted abstraction adds reader-side value beyond serving as a retrieval index.

The det-embedding system is the post-1 substrate, not its retrieval-side AIC sidecar. It is the immediate baseline.

---

## 3. Methods: why the cache is committed

Naive LLM extraction at temperature 0 is not deterministic. Two independent passes of the same prompt over the same input, with identical model and seed parameters, produce overlapping but non-identical fact sets at the API level — driven by routing, batching, and other server-side non-determinism outside the user's control. This is a real property of LLM-based memory writers, not a configuration error.

To quantify before scoring, two independent extraction passes (`seed-A` and `seed-B`) were run over the smoke unique-session set, with the same prompt SHA `44826dbf73e3455f...` and the same resolved model snapshot. They were then diffed offline (no API spend):

- Both passes covered 2,346 unique (session_id, content_sha256) keys (parity).
- 58.1% of sessions produced *byte-identical* fact sets across passes.
- Median per-session fact-set Jaccard: 1.00. Mean: 0.78.
- Per-corpus stable-core (facts appearing in both passes): **79%**.
- Variance scaled with session content density (sessions with 0 facts: identical; sessions with 21+ facts: mean Jaccard 0.565).
- Symmetric only-A vs. only-B counts (2,540 vs. 2,497), indicating no systemic richness asymmetry.

A meaningful share of the apparent 41.5% symmetric-difference is paraphrastic, not semantic (e.g., one pass writes *"the user is allocating budget for venue, materials, and facilitators"*; the other appends *"…for their workshop"*: identical content under embedding scoring, zero overlap under exact-text matching). The 79% stable-core is therefore a conservative floor.

Given this measured non-determinism, the experiment freezes one extraction as a canonical artifact. `seed-A` (Stage-1 v1, user-only prompt) and `seed-A-v2` (Stage-1 v2, role-aware) are committed to the repository as JSONL files with stamped manifests recording `prompt_sha256`, `extractor_model_resolved`, file SHA-256, and timestamp. Both files are byte-integrity-checked via `CHECKSUMS.sha256`. A load-time guard refuses to use any cache whose prompt SHA or model snapshot does not match the current code; this prevents silent staleness if either is changed.

Zero-LLM-call replay was verified for both caches across the smoke set: all extract requests resolve to cache hits, with byte-identical `Fact` events emitted in identical FIFO order. The substrate operating on the frozen extraction is fully deterministic and replayable. The non-determinism is contained in a one-time write that is then version-pinned.

This is the property the post-1 paper claimed for its substrate, now preserved for the semantic layer above it: deterministic replay given the committed log of extracted facts, even when those facts came from a non-deterministic generative process.

The full 79%/Jaccard breakdown is in `.scratch_build/seed_diff_A_B.md`; the offline diff script is `scripts/diff_seeds.py`.

---

## 4. How each system writes and reads memory

### 4.1 Substrate (`activegraph-det-embedding`)

The substrate from post 1, unchanged. Sessions and turns ingested as events; turns scored against the query with embedding similarity; seed turns selected; expansion via temporal and co-occurrence structure; packing under the 2,500-token budget; chronological render.

No LLM-generated memory at ingest. The only LLM in the pipeline is the reader.

### 4.2 Naive extraction (`activegraph-sem-extract`)

A single `@llm_behavior`, `sem_extract_facts_from_session`, fires per session during ingest. Each call sends the full session text and a prompt instructing the model to return `_ExtractedFactList` — a Pydantic-typed list of facts of the form *"The user [verb] [object]…"*, with `mentioned_turn_idxs` linking each fact to the source turn(s) within the session. Each returned fact becomes a `Fact` node in the graph and a `mentions` edge to each named turn.

Extraction is event-sourced: each fact's creation is a recorded event in the append-only log, replayable byte-for-byte from the committed `seed-A.jsonl`.

The prompt was tuned through a single-question iteration cycle on `e47becba` (53 sessions; 387 baseline facts → 249 tuned, a 36% drop) to:
1. Exclude facts whose only basis is *"the user asked the assistant about X"* — facts that would not be true had the conversation never occurred.
2. Deduplicate facts that express the same claim at different granularities.

This is the prompt frozen at SHA `44826dbf73e3455f...` and used for the entire Stage-1 v1 cache.

At retrieval and assembly time, `activegraph-sem-extract` scores `Fact` nodes (not turns) against the question. The reader sees only fact text; raw source turns do not enter context unless smuggled in by token-budget-fill behavior at the assembler's tail.

### 4.3 Ablation arms (`activegraph-sem-index`, `activegraph-sem-hybrid` v1)

Both new systems were implemented behind a shared `_sem_compiled.py` module, reusing the existing `Scoreable` seam introduced when sem-extract was added. They use the same `seed-A` cache, the same fact pool, the same embedding scoring, and the same 2,500-token budget. They differ only in what `assemble()` emits to the reader.

**`activegraph-sem-index`** — facts are scored for retrieval, then replaced by their provenance-anchored source turn(s) (via `mentions` edges) in the rendered context. Deduplicate source turns. Render turns in chronological order. The reader sees *only turn text*, identical in format to det-embedding's output, but the *selection* came via the fact pool. This is the cleanest ablation: it isolates whether facts are a better *retrieval signal* than direct turn embedding.

**`activegraph-sem-hybrid` (v1)** — facts are scored, selected, and included in the rendered context as labeled headers above their provenance-anchored source turns. When multiple facts share a turn, the turn is rendered once with all relevant fact-headers stacked above it. Render entries in chronological order by `(session_date, session_idx, turn_idx)`. Budget accounting is unified: each *fact entry* costs `len(fact.text) + sum(len(turn.text) for turn in mentions)` tokens; high-scored facts therefore consume more budget because they bring their turns with them, biasing toward fewer-but-deeper context.

### 4.4 Role-aware extraction (`activegraph-sem-hybrid` v2)

The single observed structural failure of v1's hybrid (see §5) is a 0/5 on `single-session-assistant`: questions asking what the *assistant* previously said, recommended, calculated, or produced. The user-centric prompt produces no facts retrievable for assistant-authored questions.

The v2 fix adds a second `@llm_behavior`, `sem_extract_facts_from_session_assistant`, with an assistant-centric prompt ("the assistant recommended X" / "the assistant computed Y" / "the assistant told the user that Z"). Both behaviors fire per session in a FIFO single-threaded reaction loop, preserving determinism. Each fact carries `data["role"] = "user"` or `"assistant"`. Cache keys include role: `(session_id, content_sha256, role)`. Fact IDs hash in role: `fact:<sha256(session_id|role|text)>`.

The manifest's prompt signature is computed from the *combined* prompt set; the load-time guard correctly refuses the v1 cache under the v2 behavior set. The committed `seed-A.jsonl` is preserved for history; the canonical cache is `seed-A-v2.jsonl`.

Critically: **the assembler does not change.** Both `sem-index` and `sem-hybrid` retrieve over the full fact pool. The role field is metadata, not a filter. The embedding signal does the discrimination: questions about *what the assistant said* score higher against assistant-prefixed facts; questions about *what the user did* score higher against user-prefixed facts. The role-aware fix is a representational change at write time, not a retrieval-side change.

A 20-fact random sample of the assistant-fact pool (taken offline before scoring) showed the expected answer-bearing form: *"The assistant recommended Coursera's Python for Data Science specialization from University of Michigan"*; *"The assistant explained that Ripple Plant-Based Protein Powder comes in Chocolate, Vanilla, and Unflavored options"*; *"The assistant suggested race calendar websites like Active.com, RunningintheUSA.com, or CharityRunCalendar.com"*. Specific, named, retrievable.

The role-aware build was infrastructure-bumpy (a session-gated write-path bug initially wedged the parallel build; details footnoted) and was completed in ~110 minutes wall-clock for the smoke set with 8-way parallelism. ~4,690 cache entries total (2,345 sessions × 2 roles), four sessions stubbed as `{"facts": []}` on deterministic parse failure — byte-equivalent to the live `behavior.failed` path and matching the rate observed on the v1 build.

---

## 5. Results

### 5.1 Smoke result table

All five systems run against the same 50-question smoke subset, same judge, same reader, same budget.

| System | Overall | ss-user (7) | ss-asst (5) | ss-pref (3) | multi-sess (13) | temporal (14) | KU (8) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `activegraph-det-embedding` | 0.86 | 1.00 | 1.00 | 1.00 | 1.00 | 0.571 | 0.875 |
| `activegraph-sem-extract` | 0.72 | 0.571 | 1.00 | 1.00 | 0.769 | 0.500 | 0.875 |
| `activegraph-sem-index` | 0.82 | 1.00 | 0.00 | 1.00 | 0.846 | 0.857 | 1.000 |
| `activegraph-sem-hybrid` (v1) | 0.86 | 1.00 | 0.00 | 1.00 | 0.923 | 0.929 | 1.000 |
| `activegraph-sem-hybrid` (v2, role-aware) | **0.92** | **1.00** | **0.80** | **0.667** | **1.00** | **0.857** | **1.000** |

Per-category n shown in parentheses. Abstention accuracy is 1.0 on all systems (n=4); the remaining 46 questions are the substantive answerable set.

### 5.2 Finding 1 — Naive replacement loses; the ablation locates the gain

Naive `sem-extract` regresses 16 percentage points (0.86 → 0.72) at matched budget. The losses concentrate on `single-session-user` (1.00 → 0.571; −3 of 7 questions) and `multi-session` (1.00 → 0.769; −3 of 13). These are categories where the original turn text carried answer-bearing detail at granularities the extractor's prompt explicitly compresses ("the user has a Fitbit" preserves the brand but not necessarily the model number; "the user uses a foam roller after morning yoga" loses the source's exact phrasing of timing).

The two ablation arms isolate where the recovery comes from:

- `sem-index` recovers to 0.82, ten of the fourteen lost points. The retrieval pool of *facts* surfaces the right *turns* more reliably than direct turn embedding does. The reader sees only turn text, identical in format to the substrate's output. The gain is entirely on the retrieval side.

- `sem-hybrid` v1 recovers to 0.86, matching the substrate at matched budget, four points above `sem-index`. The gain over index is that the reader also benefits from seeing the abstracted fact text alongside the raw turn. The gain over substrate is the same retrieval-side gain `sem-index` showed, *plus* the within-budget reader-side benefit of typed abstraction.

Decomposing: **roughly two-thirds of the recovery from naive extraction's regression comes from facts being a smarter index over turns; the remaining third comes from facts also being visible to the reader.** Both effects are real, both are additive, neither subsumes the other.

### 5.3 Finding 2 — Retrieval headroom predicted by post 1 closes (with a caveat on mechanism)

Post 1's per-type AIC analysis (its Table 7) identified two categories with non-trivial turn-level retrieval headroom on the substrate: `multi-session` (76.0% turn-AIC, the right *session* was retrieved 95.9% of the time, but the right *turn* much less often) and `temporal-reasoning` (74.6% turn-AIC). Knowledge-update was already near-ceiling on the substrate (95.8% turn-AIC), with the remaining errors attributable to reader reconciliation, not retrieval.

On the smoke set, the substrate baseline reproduces these category positions (within sampling noise): `multi-session` 1.00 (the smoke happens to be easy here for the substrate), `temporal-reasoning` 0.571, `knowledge-update` 0.875.

`sem-hybrid` v1 lifts `temporal-reasoning` from 8/14 (0.571) to 13/14 (0.929), a count delta of +5 questions. It lifts `knowledge-update` from 7/8 (0.875) to 8/8 (1.000), a count delta of +1 question. Both are concrete counts on small n, but both also match the *direction* post 1's analysis predicted.

A careful note on `knowledge-update`: post 1's prediction was specifically that a semantic layer would help KU only via *supersession marking* (typing facts as "current" vs. "superseded" before assembly, easing reader reconciliation). Stage-1 in this work does **not** implement supersession; extracted facts coexist in the graph regardless of whether they conflict. The +1/8 KU gain therefore *cannot* be attributed to supersession. The most plausible mechanism, consistent with the AIC structure, is that the extraction normalizes the user's updated state into a single retrievable fact (e.g., the most recent "the user's pet is a cat named Mochi" surfaces ahead of an earlier "the user's pet is a dog") under embedding scoring, even without explicit supersession marking. This is a finding, not a confirmation: it suggests that the Stage-2 supersession mechanism may not be as load-bearing as post 1's narrow prediction implied, at least at compact budgets. Stage-2 supersession remains worth measuring, but the smoke result here is neutral-to-positive on whether KU still has supersession-shaped headroom.

The `temporal-reasoning` gain (+5/14 under v1; net +4/14 under v2 after one displacement regression) is more straightforwardly consistent with the retrieval-headroom hypothesis: dated facts ("the user's flight on April 3") surface more reliably under fact-pool scoring than raw turn text where the date is buried in conversation. n=14 is small enough that the result wants full-500 confirmation, but the direction and effect size are well within what post 1's analysis predicted.

### 5.4 Finding 3 — Independent reproduction of the assistant-coverage failure

Both `sem-hybrid` v1 and `sem-index` score **0/5 on single-session-assistant**. The substrate scores 5/5; naive `sem-extract` also scores 5/5. Two structurally different new systems, sharing only the user-centric extraction prompt and the fact-as-retrieval-signal architecture, fail on identically the same questions.

This is the [*Semantic Memory Forgot the Assistant*](https://activegraph.ai/blog/semantic-memory-forgot-the-assistant) finding, independently reproduced on the post-1 harness with a different ablation arm. The mechanism is the same: the user-centric extraction prompt produces no memories the system can retrieve when the question asks what the assistant said. Whether the reader sees facts (`hybrid`) or only turns (`index`) is irrelevant — the retrieval signal itself is biased the wrong way, because the fact pool contains nothing assistant-authored to match against an assistant-targeted query.

Why does naive `sem-extract` *not* fail here? Because at the 2,500-token budget on this smoke set, the naive system's assembler, lacking enough high-quality facts, fills the remaining budget with raw turns by default. Those turns happen to include the assistant's outputs verbatim, which the reader can recover the answer from. The naive system "succeeds" on `ss-assistant` not because its extraction is good, but because its extraction is poor enough that turns survive into context. Both new systems, with better retrieval signal and more selective filling, displace the assistant turns before the reader sees them.

Counterfactually: **without the `ss-assistant` 0/5 contribution, `sem-hybrid` v1 would land at ~0.96 (47-48/50)**, and `sem-index` at ~0.92 (~46/50). Both would substantially exceed the substrate's 0.86. The 0/5 is the single most consequential per-type cell in the table; recovering it is the entire delta between "Stage-1 ties the substrate" and "Stage-1 dominates the substrate."

### 5.5 Finding 4 — Role-aware extraction lands the recovery, with one identifiable tradeoff

`sem-hybrid` v2, scored against `seed-A-v2` (committed at `data/sem_extract_cache/seed-A-v2.jsonl`, manifest pinned to combined prompt SHA `603cd99e657a5412...`):

The role-aware fix lands the predicted recovery: `single-session-assistant` moves from 0/5 to 4/5, lifting overall accuracy from 0.86 to 0.92 (43/50 → 46/50). The mechanism is confirmed: adding assistant-typed facts to the hybrid retrieval pool is sufficient to recover the category, without any retrieval or assembler change. The embedding signal correctly routes assistant-targeted queries to assistant-prefixed facts, validating the design's premise that role discrimination is a representational property, not a retrieval-side filter.

Three per-category regressions accompany the gain, and per-question inspection reveals a single dominant mechanism rather than disparate noise:

*— `single-session-preference` (3/3 → 2/3, −1 question).* Question `07b6f563` asks for phone-accessory suggestions; the gold requires retrieving the user's earlier statement that they own an iPhone 13 Pro. The v2 hybrid's hypothesis explicitly states *"I don't have any information in our conversation history about what type of phone you have"* — the iPhone fact, surfaced under v1, did not survive into v2's retrieved context. Cause: budget competition between the user-fact pool and the new assistant-fact pool at the matched 2,500-token budget.

*— `temporal-reasoning` (13/14 → 12/14, −1 question).* Two questions failed under v2 that succeeded under v1. `gpt4_7abb270c` asks for the order of six museum visits; the reader returns five correctly ordered museums plus the explicit admission *"I can only confirm five museums from the conversation history, not six."* This is the same budget-competition mechanism: one of the six museum-visit facts was displaced from the 2,500-token context. `eac54add` asks for the user's significant business milestone "four weeks ago"; the reader correctly computes the time window (late February 2023) but selects the wrong event from that window, returning *"your influencer collaboration"* where the gold is *"I signed a contract with my first client."* This third failure is mechanism-ambiguous between retrieval-displacement and reader content-selection; an AIC-sidecar audit would isolate it.

Net effect: +4 on `ss-assistant`, +1 on `multi-session`, −1 on `ss-preference`, −2 on `temporal-reasoning`. Two of the three regressions, and possibly all three, share a single cause: **adding a second-role fact pool at a fixed retrieval budget dilutes per-question fact ranking, occasionally displacing answer-bearing user facts** that v1's single-pool retrieval surfaced. This is the principled tradeoff of role-aware extraction at compact budget, and is a category-redistribution finding rather than a uniform improvement.

The mechanism has a clear implication for the full-500 scale: at compact 2,500-token budget on smoke, single-displacement events are visible and consequential because each cell is small (n=3 to n=14). At full 500, the per-category n grows by an order of magnitude and single-displacement events average out; the budget-competition cost should be measurable but smaller proportionally, while the assistant-recovery benefit scales linearly. This predicts that the +6pp smoke gain is a *floor*, not a ceiling, for the full-500 result.

The 4/5 (not 5/5) on `ss-assistant` is consistent with one residual question whose assistant-authored content is not surfaced even with the role-aware extractor; the most plausible cause is the [Borges fidelity-loss pattern](https://activegraph.ai/blog/semantic-memory-forgot-the-assistant) — the extraction paraphrases the assistant's output, the topic survives, the exact span does not. A provenance-backed verbatim-fallback for quote/list/calculation questions is the natural next refinement.

---

## 6. What is and isn't claimed

### Claimed

1. **At matched 2,500-token budget on smoke n=50, naive LLM extraction underperforms the substrate by 16pp.** This regression is driven by information loss in single-session-user and multi-session categories where raw turn text carries answer-bearing detail.

2. **A hybrid design — LLM facts as a retrieval index over raw turns, with provenance-anchored turns reaching the reader — matches the substrate's overall accuracy and exceeds it on `temporal-reasoning` (+4/14 under v2) and `knowledge-update` (+1/8).**

3. **The architectural finding decomposes:** of the recovery from naive extraction's regression, approximately two-thirds is attributable to facts being a better *retrieval signal* (sem-index 0.82 vs sem-extract 0.72) and one-third to facts being *visible to the reader* (sem-hybrid 0.86 vs sem-index 0.82). Both effects are real and additive.

4. **Naive user-centric extraction independently reproduces the assistant-coverage failure** reported in [*Semantic Memory Forgot the Assistant*](https://activegraph.ai/blog/semantic-memory-forgot-the-assistant): single-session-assistant drops to 0/5 in both new ablation systems. A role-aware extraction fix recovers this category to 4/5 without any retrieval or assembler change. Overall accuracy lifts to 0.92 (46/50).

5. **The role-aware fix has one identifiable, principled tradeoff:** budget competition between user and assistant fact pools at compact context, evidenced by inspection of the failed `ss-preference` and `temporal-reasoning` questions. This tradeoff should scale sub-linearly with dataset size while the assistant-recovery benefit scales linearly, predicting that the +6pp smoke gain is a floor for full-500.

6. **Determinism is preserved.** LLM-extracted facts are written as events and frozen as committed artifacts with prompt + model-snapshot provenance. The substrate operating on the committed log is byte-identical on rerun (verified by zero-LLM-call replay).

### Not claimed

1. **This does not beat post 1's 85.6% full-500 substrate result.** Smoke n=50 and full 500 are not comparable; the per-category deltas are concrete counts (e.g., +4/14 on temporal) but the overall accuracy delta on smoke (0.86 → 0.92) is not. The smoke result that sem-hybrid v2 beats the substrate by 6pp is not yet replicated at full-500.

2. **Graph topology is not established as the causal mechanism.** As in post 1, the unit of comparison is the full evidence-compilation pipeline; this experiment does not isolate the contribution of graph structure vs. scoring vs. packing vs. role-aware typing.

3. **The Stage-2 supersession hypothesis is not tested.** Knowledge-update improved without supersession marking, weakening (but not falsifying) post 1's narrow prediction that supersession is the mechanism. Direct supersession measurement is a future experiment.

4. **The role-aware fix is not yet shown to survive full-500 evaluation.** The smoke result is consistent with the new post's full-500 fix (which moved ss-assistant from 75.0% → 98.2% on that different harness), but in-harness, in-budget, full-500 reproduction is the next step.

5. **The variance band is bounded but not fully measured at the score level.** Offline fact-level variance is 79% stable-core. A scored seed-A vs seed-B/C variance band on smoke was deferred when the architecture finding dominated the variance question; it remains a clean follow-up.

---

## 7. Limitations and follow-ups

**Sample size.** n=50 smoke. Per-category cells (n=3 to n=14) are concrete count effects but want full-500 confirmation. The post 1 paper noted that 50-question slices produce substantial sampling noise on per-type accuracy.

**Single seed.** Only `seed-A-v2` was scored. The offline fact-level variance measurement bounds the input noise, but the score-level variance band across independent extractions (seed-B-v2, seed-C-v2) was not measured. This is the cleanest cheap follow-up: independent extractions are ~$15 each at smoke, scored runs are ~$2 each, total ~$35 for a 3-point variance band on the headline number.

**Judge robustness.** A single judge (`gpt-4o-2024-08-06`) was used across all five systems compared here; this is internally consistent but not externally validated. A second-judge spot-check on discordant pairs (the questions where one system answers right and another wrong) is the standard robustness check.

**Causal attribution.** As above: the experiment shows that the hybrid pipeline beats the naive replacement pipeline. It does not isolate which specific component (the embedding-cosine over facts, the provenance link rendering, the chronological re-ordering, the typed role distinction) contributes how much. The next ablation layer would hold scoring fixed and vary each.

**Borges regression.** The new ActiveGraph post identified a fidelity-loss failure mode where extraction paraphrases the assistant's verbatim output and the exact span is lost (their case: a Borges quote). Stage-1 hybrid mitigates this *partially* by including the source turn alongside the fact header, but only if that source turn is in the selected pool. A provenance-backed verbatim-fallback experiment — for quote, list, code, and calculation questions, follow `mentions` edges back to the raw span regardless of budget pressure — is the natural extension. The 1/5 residual on `ss-assistant` under v2 is the immediate motivation.

**Budget competition.** The §5.5 per-question inspection identifies budget competition between user and assistant fact pools as the dominant tradeoff mechanism. A clean ablation would vary the retrieval budget (e.g., 2,500 vs 5,000 vs 10,000 tokens) holding everything else fixed, and measure whether the displacement regressions disappear at larger budgets while the assistant-recovery benefit holds.

**Stage-2.** Supersession (typed "current" vs. "stale" facts, resolved before assembly) was the second tier of the original semantic-memory plan. The KU result here (1.00 without supersession) suggests Stage-2 may have less headroom than post 1 expected, but the test for supersession is whether it improves multi-session questions where the *user's state* changed mid-conversation. That measurement remains.

### Immediate next experiment

The cheapest highest-leverage next step is the **full-500 scored run of `sem-hybrid` v2**:

- Extend `seed-A-v2.jsonl` from smoke (2,345 sessions × 2 roles) to full-500 (~19,000 sessions × 2 roles). Cost ~$30 in extraction. The same parallel build, same write-path, same role-aware behavior set.
- Score against `activegraph-det-embedding` on the same full-500 subset (matched judge, matched budget, matched seed). Cost ~$60 for both scored runs.
- Compute exact McNemar on the paired per-question correctness vectors, and the AIC sidecar significance on turn-level coverage, to match post 1's statistical rigor.

Total: ~$90 and ~6 hours of wall-clock to produce a paper-strength replication of the smoke finding.

---

## 8. Reproduction

Branch `claude/loving-bohr-4x2nK` at commit `[HEAD]`. Run dirs in `runs/`. Cache at `data/sem_extract_cache/seed-A-v2.jsonl` (committed `51990bf`); SHA pinned in `CHECKSUMS.sha256`.

```bash
git clone https://github.com/yoheinakajima/activegraph-longmemeval
cd activegraph-longmemeval
git checkout claude/loving-bohr-4x2nK
make setup && make data

# Verify the committed cache (no API spend).
uv run python scripts/verify_extract_cache.py --seed A-v2

# Reproduce the five-way smoke ablation.
uv run python -m activegraph_lme.cli run --system activegraph-det-embedding --dataset s --smoke
uv run python -m activegraph_lme.cli run --system activegraph-sem-extract --dataset s --smoke --extract-seed A
uv run python -m activegraph_lme.cli run --system activegraph-sem-index --dataset s --smoke --extract-seed A
uv run python -m activegraph_lme.cli run --system activegraph-sem-hybrid --dataset s --smoke --extract-seed A
uv run python -m activegraph_lme.cli run --system activegraph-sem-hybrid --dataset s --smoke --extract-seed A-v2

# Eval and sidecar each run.
for run_dir in runs/*activegraph-*__s__smoke; do
    uv run python -m activegraph_lme.cli eval --run-dir "$run_dir"
    uv run python scripts/aic_sidecar.py "$run_dir"
done
```

All five systems share reader, judge, embedding model, retrieval budget, and dataset seed. The committed `seed-A-v2.jsonl` produces zero LLM extraction calls during scored runs; headline numbers reproduce byte-for-byte given the pinned reader and judge snapshots. The reader is the only source of API-side non-determinism remaining; in practice, `claude-sonnet-4-5-20250929` at temperature 0 produces stable outputs across reruns on this prompt structure (informally observed; not formally measured here).

---

## 9. References and prior work

- Yohei Nakajima, [*Evidence Compilation Before Semantic Memory: ActiveGraph on LongMemEval-S*](https://activegraph.ai/blog/evidence-compilation-before-semantic-memory-longmemeval), May 2026.
- Yohei Nakajima, [*Semantic Memory Forgot the Assistant: A 500-Question LongMemEval-S Debugging Report*](https://activegraph.ai/blog/semantic-memory-forgot-the-assistant), May 2026.
- Yohei Nakajima, [*The Log is the Agent: Event-Sourced Reactive Graphs for Auditable, Forkable Agentic Systems*](https://arxiv.org/abs/2605.21997), 2026.
- Di Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025.
- Nelson F. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024.
- Cheng-Yu Hsieh et al., *Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization*, ACL Findings 2024.
- Jiaqi Wei et al., *AlignRAG: An Adaptable Framework for Resolving Misalignments in Retrieval-Aware Reasoning of RAG*, 2025.
- Preston Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*, 2025.
- Prateek Chhikara et al., *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*, 2025.
- Sourav Verma, *Contextual Compression in Retrieval-Augmented Generation for Large Language Models: A Survey*, 2024.

---

*Appendix A: per-question diffs of `sem-hybrid` v1 vs. `sem-extract`, in `paper/v1_per_question_audit.md` (TBD).*
*Appendix B: per-question audit of the 4 ss-assistant recoveries under role-aware v2, in `paper/v2_assistant_audit.md` (TBD).*
*Appendix C: offline `seed-A` vs `seed-B` fact-level variance breakdown, in `.scratch_build/seed_diff_A_B.md`.*
