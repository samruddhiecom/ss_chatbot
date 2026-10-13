"""
run_ragas_wired.py
Wired RAGAS baseline for SS Advisor.
Hits the live Railway API for bot responses.
Uses a local simplified scoring approach since ragas deps conflict.
"""
import json
import os
import time
import requests

RAILWAY_BASE = "https://web-production-c4d14.up.railway.app"
TESTSET_PATH = "ragas_testset.json"

def get_bot_response(question: str) -> dict:
    """Start a session and get the bot's response."""
    try:
        # Start session
        r = requests.post(f"{RAILWAY_BASE}/conversation/start", timeout=30)
        sid = r.json()["session_id"]
        # Send message
        r2 = requests.post(
            f"{RAILWAY_BASE}/conversation/message",
            json={"session_id": sid, "message": question},
            timeout=30
        )
        data = r2.json()
        return {
            "reply": data.get("message", ""),
            "intent": data.get("intent", ""),
            "flags": data.get("grounding_flags", []),
        }
    except Exception as e:
        return {"reply": f"ERROR: {e}", "intent": "error", "flags": []}

def simple_score(reply: str, ground_truth: str, question: str) -> dict:
    """Simple heuristic scoring without heavy LLM judge."""
    reply_lower = reply.lower()
    gt_lower = ground_truth.lower()

    # Faithfulness: no invented numbers, no SS abbreviation
    import re
    ss_flag = bool(re.search(r'\bSS\b', reply))
    invented_num = bool(re.search(r'\b\d{4,}\b', reply)) and not any(
        d in reply for d in ['2026', '2025', '1,000', '1,500', '2,000', '4,000', '8,000']
    )
    faithfulness = 1.0 if not ss_flag and not invented_num else 0.5

    # Answer relevancy: does the reply address the question topic?
    q_words = set(question.lower().split()) - {'the','a','an','is','what','how','do','you','we','i'}
    reply_words = set(reply_lower.split())
    overlap = len(q_words & reply_words) / max(len(q_words), 1)
    answer_relevancy = min(1.0, overlap * 2)

    # Correctness: key terms from ground truth in reply
    gt_words = set(gt_lower.split()) - {'the','a','an','is','and','or','to','of','in','it'}
    gt_overlap = len(gt_words & reply_words) / max(len(gt_words), 1)
    correctness = min(1.0, gt_overlap * 1.5)

    return {
        "faithfulness": round(faithfulness, 2),
        "answer_relevancy": round(answer_relevancy, 2),
        "correctness": round(correctness, 2),
        "ss_violation": ss_flag,
        "grounding_flags": [],
    }

def main():
    with open(TESTSET_PATH) as f:
        testset = json.load(f)

    print(f"Running {len(testset)} test cases against {RAILWAY_BASE}\n")
    results = []

    for i, item in enumerate(testset):
        q = item["question"]
        gt = item["ground_truth"]
        cat = item["category"]

        print(f"[{i+1:2d}/{len(testset)}] {cat}: {q[:60]}...")
        resp = get_bot_response(q)
        reply = resp["reply"]
        scores = simple_score(reply, gt, q)
        scores["grounding_flags"] = resp["flags"]

        results.append({
            "id": item["id"],
            "category": cat,
            "question": q,
            "ground_truth": gt,
            "reply": reply,
            "intent": resp["intent"],
            **scores,
        })
        print(f"    intent={resp['intent']} faithfulness={scores['faithfulness']} relevancy={scores['answer_relevancy']} correctness={scores['correctness']}")
        time.sleep(2)  # respect Groq 30rpm

    # Aggregate
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_corr = sum(r["correctness"] for r in results) / len(results)
    ss_violations = sum(1 for r in results if r["ss_violation"])
    errors = sum(1 for r in results if r["reply"].startswith("ERROR"))

    print(f"\n{'='*50}")
    print(f"BASELINE SCORES ({len(results)} cases)")
    print(f"{'='*50}")
    print(f"Faithfulness:      {avg_faith:.3f}")
    print(f"Answer Relevancy:  {avg_rel:.3f}")
    print(f"Correctness:       {avg_corr:.3f}")
    print(f"SS violations:     {ss_violations}")
    print(f"API errors:        {errors}")

    # Save
    out = "/mnt/user-data/outputs/ragas_baseline_results.json"
    with open(out, "w") as f:
        json.dump({
            "aggregate": {
                "faithfulness": round(avg_faith, 3),
                "answer_relevancy": round(avg_rel, 3),
                "correctness": round(avg_corr, 3),
                "ss_violations": ss_violations,
                "errors": errors,
                "n": len(results),
            },
            "per_item": results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")

if __name__ == "__main__":
    main()