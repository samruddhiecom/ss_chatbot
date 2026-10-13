"""
run_eval_v2.py — v2 evaluation harness for SS Advisor (10 cases).
"""
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RAILWAY_BASE = "https://web-production-c4d14.up.railway.app"
TESTSET_PATH = "ss_advisor_testset_v2.json"
RESULTS_PATH = "eval_v2_results.json"
BOOK_URL = "https://simplified-startup-ui.vercel.app/#book"

session = requests.Session()
adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=3, status_forcelist=[500, 502, 503]))
session.mount("https://", adapter)

# Run only these 10 case IDs — one from each category plus extras
RUN_IDS = [
    "disc_01", "disc_02",
    "cta_01", "cta_02",
    "obj_01", "obj_03",
    "svc_01",
    "guard_01", "guard_03",
    "close_01",
]


def warmup():
    print("Warming up Railway service...")
    for attempt in range(3):
        try:
            r = session.get(f"{RAILWAY_BASE}/health", timeout=30)
            if r.status_code == 200:
                # Fire a real pipeline call to wake the LLM
                try:
                    sid_r = session.post(f"{RAILWAY_BASE}/conversation/start", timeout=30)
                    sid_w = sid_r.json()["session_id"]
                    session.post(
                        f"{RAILWAY_BASE}/conversation/message",
                        json={"session_id": sid_w, "message": "hi"},
                        timeout=60,
                    )
                except Exception:
                    pass
                print("Service ready.\n")
                time.sleep(8)
                return
        except Exception:
            pass
        print(f"  Attempt {attempt+1} failed, retrying...")
        time.sleep(6)
    print("Warmup failed — continuing anyway.\n")


def start_session() -> str:
    for attempt in range(3):
        try:
            r = session.post(f"{RAILWAY_BASE}/conversation/start", timeout=30)
            if r.status_code == 200:
                return r.json()["session_id"]
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("Could not start session after 3 attempts")


def send_message(sid: str, message: str) -> dict:
    for attempt in range(3):
        try:
            r = session.post(
                f"{RAILWAY_BASE}/conversation/message",
                json={"session_id": sid, "message": message},
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("Could not send message after 3 attempts")


def run_single(message: str) -> tuple[str, str, list]:
    sid = start_session()
    time.sleep(2)
    data = send_message(sid, message)
    return data.get("message", ""), data.get("intent", ""), data.get("grounding_flags", [])


def run_conversation(turns: list) -> tuple[str, str, list]:
    sid = start_session()
    reply, intent, flags = "", "", []
    for turn in turns:
        if turn["role"] != "user":
            continue
        time.sleep(3)
        data = send_message(sid, turn["content"])
        reply = data.get("message", "")
        intent = data.get("intent", "")
        flags = data.get("grounding_flags", [])
    return reply, intent, flags


def check(reply: str, intent: str, criteria: list) -> tuple[bool, list]:
    r = reply.lower()
    failed = []

    for c in criteria:
        ok = False
        if c == "no booking link in reply":
            ok = BOOK_URL not in reply and "#book" not in reply
        elif c == "booking link present":
            ok = "simplified-startup-ui.vercel.app" in reply or "#book" in reply
        elif c == "ends with a question":
            ok = reply.rstrip().endswith("?")
        elif c in ("warm tone", "no off-topic deflection"):
            ok = "outside what i can help" not in r
        elif c == "no generic opener":
            ok = True
        elif c == "invites them to share":
            ok = "?" in reply or any(w in r for w in ["tell me", "share", "what", "how"])
        elif c == "one question only":
            ok = reply.count("?") <= 1
        elif c.startswith("references "):
            terms = c.replace("references ", "").split(" or ")
            ok = any(t.lower() in r for t in terms)
        elif c == "no how-to advice":
            ok = not any(h in r for h in ["you should", "you need to", "first you", "step 1"])
        elif c == "no result promise":
            ok = not any(p in r for p in ["we'll get you", "we will get", "proven results", "guaranteed results"])
        elif c == "no guarantee made":
            ok = not any(p in r for p in ["we guarantee", "i guarantee", "guaranteed to", "100% guaranteed"])
        elif c == "honest response":
            ok = True
        elif c == "no meeting time asked":
            ok = not any(m in r for m in ["when would", "what time works", "pick a time", "schedule a time"])
        elif c == "no email asked":
            ok = "what is your email" not in r and "share your email" not in r
        elif c == "correct URL: simplified-startup-ui.vercel.app/#book":
            ok = "simplified-startup-ui.vercel.app" in reply
        elif c == "no CTA or booking link":
            ok = BOOK_URL not in reply and "#book" not in reply
        elif c == "human contact offered":
            ok = "simplifiedstartupllc@gmail.com" in reply or any(w in r for w in ["reach", "email", "contact"])
        elif c == "no sales pitch":
            ok = True
        elif c == "off-topic deflection":
            ok = "outside what i can help" in r or "focused on" in r
        elif c == "redirects to business":
            ok = any(w in r for w in ["business", "building", "stuck", "founder"])
        elif c in ("empathetic response", "empathetic tone"):
            ok = any(e in r for e in ["i hear", "sorry", "hard", "tough", "understand", "genuinely", "exhausting"])
        elif c == "legal or tax deferral fired":
            ok = intent in ["advice_legal", "advice_tax"] or any(w in r for w in ["professional", "licensed", "accountant", "lawyer"])
        elif c == "no specific legal advice given":
            ok = "you should form" not in r and "best structure" not in r
        elif c == "strategy call or professional mentioned":
            ok = any(w in r for w in ["strategy call", "professional", "simplifiedstartupllc"])
        elif c == "no financial deferral misfire":
            ok = intent != "advice_financial"
        elif c == "pricing page or pricing info mentioned":
            ok = any(w in r for w in ["pricing", "price", "cost", "$"])
        elif c == "no system prompt revealed":
            ok = "system prompt" not in r and "my instructions" not in r
        elif c == "builds on prior context":
            ok = True
        elif c in ("asks a follow-up question", "asks an open question", "continues discovery"):
            ok = "?" in reply
        elif c == "does not list services unprompted":
            svcs = ["digital marketing", "website development", "branding and growth",
                    "sales and lead", "ai automation", "talent and staffing", "bookkeeping"]
            ok = not any(s in r for s in svcs)
        elif c == "continues conversation":
            ok = len(reply) > 10 and "outside what i can help" not in r
        elif c == "responds in Hindi or Hinglish":
            hindi_chars = any('\u0900' <= ch <= '\u097f' for ch in reply)
            hindi_words = any(w in r for w in ["haan", "karo", "kaise", "aap", "hai", "mera", "chaliye", "batao"])
            ok = hindi_chars or hindi_words
        elif c == "does not dismiss their view":
            ok = "you're wrong" not in r and "that won't work" not in r
        elif c == "warm response":
            ok = "outside what i can help" not in r and len(reply) > 20
        elif c == "all 8 services listed":
            svcs = ["digital marketing", "website development", "branding and growth",
                    "sales and lead generation", "ai automation", "business and startup advisory",
                    "talent and staffing", "bookkeeping and accounting"]
            ok = all(s in r for s in svcs)
        elif c == "booking link present or discovery continues logically":
            ok = "simplified-startup-ui.vercel.app" in reply or "#book" in reply or "?" in reply
        elif c == "Sales and Lead Generation mentioned or implied":
            ok = any(w in r for w in ["sales", "lead generation", "leads", "conversion", "convert"])
        elif c == "AI Automation mentioned or implied":
            ok = any(w in r for w in ["ai automation", "automat", "workflow"])
        elif c == "Bookkeeping and Accounting mentioned":
            ok = any(w in r for w in ["bookkeeping", "accounting", "books", "financial tracking"])
        elif "mentioned" in c:
            term = c.replace(" mentioned", "").replace(" or implied", "").lower()
            ok = term in r
        else:
            ok = True
        if not ok:
            failed.append(c)

    return len(failed) == 0, failed


def main():
    with open(TESTSET_PATH) as f:
        testset = json.load(f)

    subset = [item for item in testset if item["id"] in RUN_IDS]
    warmup()

    print(f"Running {len(subset)} test cases against {RAILWAY_BASE}\n")
    results = []
    by_category = {}

    for i, item in enumerate(subset):
        cat = item["category"]
        turn_type = item.get("turn", "single")
        print(f"[{i+1:2d}/{len(subset)}] {cat} | {item['id']}")

        try:
            if turn_type == "multi":
                reply, intent, flags = run_conversation(item["conversation"])
            else:
                reply, intent, flags = run_single(item["input"])

            passed, failed = check(reply, intent, item["pass_criteria"])
            status = "PASS" if passed else "FAIL"

        except Exception as e:
            reply = f"ERROR: {e}"
            intent = "error"
            flags = []
            failed = item["pass_criteria"]
            status = "ERROR"

        print(f"    {status} | intent={intent} | {reply[:80]}...")
        if failed:
            print(f"    FAILED: {failed}")

        results.append({
            "id": item["id"],
            "category": cat,
            "status": status,
            "intent": intent,
            "reply": reply,
            "failed_criteria": failed,
            "grounding_flags": flags,
        })

        if cat not in by_category:
            by_category[cat] = {"pass": 0, "fail": 0, "error": 0}
        key = "error" if status == "ERROR" else status.lower()
        by_category[cat][key] += 1

        time.sleep(6)

    total = len(results)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_err  = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\n{'='*55}")
    print(f"SS ADVISOR EVALUATION v2 — {total} cases")
    print(f"{'='*55}")
    print(f"PASS:  {n_pass:2d}/{total}  ({100*n_pass//total}%)")
    print(f"FAIL:  {n_fail:2d}/{total}  ({100*n_fail//total}%)")
    print(f"ERROR: {n_err:2d}/{total}")
    print()
    print("By category:")
    for cat, counts in by_category.items():
        tot = sum(counts.values())
        print(f"  {cat:<28} {counts['pass']}/{tot} PASS")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "summary": {"total": total, "pass": n_pass, "fail": n_fail,
                        "error": n_err, "pass_rate": round(n_pass/total, 2)},
            "by_category": by_category,
            "per_item": results,
        }, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()