# SS Business Planning Chatbot — Engine (draft build)

The backend engine for Simplified Startup's founder-facing planning assistant. A founder
walks a guided four-stage conversation and leaves with a structured Startup Snapshot. The
embed shell (Himanshu) calls this engine over HTTP.

This is the draft-KB build: the pipeline, stages, rails, and snapshot are complete and run
against a **draft** knowledge base. The reviewed corpus and Thomas sign-off replace the draft
before publish; nothing about the code changes when the KB is swapped.

## The one invariant

The model is never a source of facts, only an organiser of (a) what the founder typed and
(b) an allowlisted, human-reviewed corpus. Every requirement follows from this: no invented
data, no generated statistics, advice questions deferred to a professional, corpus-only
grounding. Refusals are not dead ends — each becomes a named gap plus a next step, which is
the snapshot's discovery value.

## Architecture

Guard sandwich, one LangGraph invocation per founder turn:

```
                 ┌─────────── input rail ───────────┐
  user turn ──▶  classify intent                     │
                 │   ├─ boundary/injection/off-topic ─▶ bounded refusal (+ named gap)
                 │   └─ on-topic ─▶ stage node ──────────┐
                 └────────────────────────────────────────┘
                                   │  (retrieval rail: allowlist-only, stage-filtered)
                                   ▼
                            output rail: no-invented-numbers + claims-clean
                                   ▼
                             assistant turn
```

At the end of stage four, the snapshot is assembled from the transcript + allowlist, voiced,
and gated by claims-clean before it is returned.

### Layers
- **Layer A** (planning guidance) — retrieval corpus, stage-tagged.
- **Layer B** (company facts + "what we don't do" boundary list) — retrieval corpus.
- **Layer C** (brand voice) — **not** in the corpus. Loaded into the generation prompt and
  applied as a surface rewrite *after* grounded content is locked, with a content-preservation
  check that reverts the rewrite if it introduces a figure. Grounding always wins over tone.

### A note on the guardrails
The rails are implemented natively (`app/rails/`) following the NeMo Guardrails input/dialog/
retrieval/output pattern, rather than as a Colang runtime. This keeps the engine fully testable
and dependency-light on Windows/Python, and the rails stay swappable for a NeMo config later if
we want it. This is the one deliberate deviation from the locked stack — flagging it rather than
burying it.

## Layout

```
config.py                     settings (Groq models, embed model, caps, tool_source=a52)
app/
  main.py                     FastAPI: /health /conversation/{start,message,snapshot} /capture
  graph.py                    LangGraph state machine (classify → refusal|stage → check)
  schemas.py                  Stage, Intent, Snapshot, state, API models
  stages.py                   4 stage openers, slot extraction, follow-ups
  snapshot.py                 6-section snapshot assembly + named gaps + claims-clean
  voice.py                    Layer C load + fact-preserving voice pass
  llm.py                      Groq wrapper (plain + Instructor structured)
  rails/
    input_rail.py             injection pre-filter + intent classify + grounded deferrals
    retrieval_rail.py         allowlist-only retrieval
    output_rail.py            no-invented-numbers (deterministic) + claims-clean (LLM)
  kb/
    store.py                  ChromaDB + FastEmbed BGE-small (ONNX, no torch)
    ingest.py                 tagged-chunk parser → context-enriched → embedded (A+B only)
    draft/                    layer_a_guidance.md, layer_b_company.md, layer_c_voice.md
  capture.py                  lead capture stub (tool_source=a52, snapshot attached)
redteam/
  probes.py                   Python bench, 7 categories, CI gate (exits non-zero on failure)
  promptfooconfig.yaml        declarative harness (SOP-locked)
tests/                        14 deterministic tests (no model, no network)
scripts/ingest_kb.py          KB ingestion runner
```

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# open .env and set GROQ_API_KEY
```

Note on Python version: this project is torch-free (FastEmbed uses onnxruntime). If a wheel is
missing on 3.14, use a 3.12 or 3.13 venv for `chromadb`/`fastembed`.

## Run

```powershell
python scripts/ingest_kb.py      # build the vector store from the draft KB (one time / on KB change)
python -m app.main               # serve on http://127.0.0.1:8000
```

Smoke test:

```powershell
curl.exe http://127.0.0.1:8000/health
$s = (curl.exe -s -X POST http://127.0.0.1:8000/conversation/start | ConvertFrom-Json)
curl.exe -s -X POST http://127.0.0.1:8000/conversation/message -H "Content-Type: application/json" -d (@{session_id=$s.session_id; message="We sell a candle subscription to busy parents"} | ConvertTo-Json)
```

## Test

```powershell
python -m pytest -q            # 14 deterministic tests, no key needed
```

## Red-team bench

```powershell
python -m app.main             # terminal 1
python redteam/probes.py       # terminal 2 — exits non-zero on any boundary failure
```

Acceptance target is 0 boundary failures across the seven categories (advice, projection,
statistics, injection, off-topic, abuse, hardship). The Python bench is the primary gate;
Promptfoo is the declarative equivalent; add Garak for breadth and PyRIT for multi-turn.

## What is draft / stubbed here

- **KB is draft.** `app/kb/draft/*` is neutralised placeholder content. Swap for the reviewed
  corpus, then re-run `scripts/ingest_kb.py`. No code changes.
- **Sessions are in-memory.** Production swaps `_SESSIONS` for the shared store with 30-day
  transcript retention; rate limits move to a shared counter.
- **Capture is a stub.** `capture.py` records to memory and logs; wire to the shared trio
  capture infra with the snapshot attached.
- **Degraded mode.** Without `GROQ_API_KEY` the app still boots and walks the flow so you can
  test wiring; grounded generation and the snapshot need the key.

## Next (roadmap P2 → P5)

Harden the bench to zero failures, wire capture to production infra, move sessions to the
retention store, add completion + consult-conversion instrumentation, stage with Robin, then
the SS-deploy gate.
