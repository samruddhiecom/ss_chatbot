# SS Business Planning Chatbot — Engine

The backend engine for Simplified Startup's founder-facing AI Advisor. A founder walks a guided four-stage conversation and leaves with a structured Startup Snapshot. The embed shell calls this engine over HTTP.

## What's built

- **4-stage guided flow** — idea → market → offer → ops, with slot extraction and follow-ups
- **Guard sandwich** — input rail (injection filter + intent classify) → retrieval (Notion KB, allowlist-only) → output rail (no-invented-numbers + claims-clean)
- **Notion RAG** — KB lives as a document in Notion; PMs edit it naturally, one sync command updates the bot
- **6-section Snapshot** — idea framing, customer hypothesis, offer sketch, channel shortlist, named gaps, next steps
- **Lead capture** — tagged tool_source=a52, snapshot attached
- **14/14 unit tests passing**, **5/5 boundary probes passing**

## The one invariant

The model is never a source of facts — only an organiser of (a) what the founder typed and (b) the Notion KB corpus. No invented data, no generated statistics, advice questions deferred to a professional.

## Architecture

```
user turn → input rail (classify intent)
              ├─ boundary/injection → bounded refusal + named gap
              └─ on-topic → stage node
                              ↓ (retrieval rail: Notion KB, stage-filtered)
                         output rail: no-invented-numbers + claims-clean
                              ↓
                         assistant turn
```

## Knowledge base — Notion RAG

The KB is a Notion page (`Simplified Startup Rag System`) containing the full SS AI Advisor document. PMs edit the page directly — no code, no files.

To sync changes into ChromaDB:
```powershell
python notion_page_sync.py
```

Then restart the server. The bot picks up new content immediately.

## Layout

```
config.py                     settings (Groq models, embed model, caps, tool_source=a52)
notion_page_sync.py           Notion page → ChromaDB sync (run after KB edits)
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
redteam/
  probes.py                   Python bench, 7 categories, CI gate (exits non-zero on failure)
  promptfooconfig.yaml        declarative harness (SOP-locked)
tests/                        14 deterministic tests (no model, no network)
```

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Fill in .env — see required keys below
```

### Required .env keys

```
GROQ_API_KEY=           # Groq API key
SMART_MODEL=openai/gpt-oss-120b
FAST_MODEL=openai/gpt-oss-20b
NOTION_TOKEN=           # ss chatbot sync integration token
NOTION_PAGE_ID=3d48ba072e19805a96d3f42b67c9a6e0
MAX_FOLLOWUPS_PER_STAGE=0
```

## Run

```powershell
python notion_page_sync.py   # sync KB from Notion (first time + after any KB edit)
python -m app.main           # serve on http://127.0.0.1:8000
```

## Test

```powershell
python -m pytest -q          # 14 deterministic tests, no key needed
python boundary_test.py      # 5 boundary probes against live server
```

## Red-team bench

```powershell
python -m app.main           # terminal 1
python redteam/probes.py     # terminal 2 — exits non-zero on any boundary failure
```

Acceptance target: 0 boundary failures across legal, tax, financial, projection, statistics, injection, off-topic, abuse, hardship.

## What is still stubbed

- **Sessions are in-memory** — production swaps to shared store with 30-day transcript retention
- **Capture is a stub** — wire `capture.py` to shared trio capture infra
- **Models** — currently using `openai/gpt-oss-120b` and `openai/gpt-oss-20b` as substitutes for Llama; confirm with Utkarsh before deploy

## Remaining roadmap (P2 → P5)

- P2 — Red-team hardening to zero failures (Garak + PyRIT)
- P3 — Wire capture to production infra, sessions to retention store
- P4 — Completion + consult-conversion instrumentation
- P5 — Staging with Robin → SS deploy gate
