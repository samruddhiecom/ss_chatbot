"""
metrics_calculator.py
Computes guardrail precision/recall and service routing accuracy
from eval_v2_results.json produced by run_eval_v2.py.

Run:
    python metrics_calculator.py
"""
import json

RESULTS_PATH = "eval_v2_results.json"

# Which test IDs belong to each metric category
GUARDRAIL_IDS = ["guard_01", "guard_02", "guard_03", "guard_04", "guard_05"]
SERVICE_ROUTING_IDS = ["svc_01", "svc_02", "svc_03", "svc_04"]

# For guardrail precision — IDs where the bot should NOT fire a guardrail
# (on-topic business questions that must not be deflected)
SHOULD_NOT_DEFLECT = ["obj_01", "obj_02", "obj_03", "obj_04", "disc_01", "disc_03", "disc_04", "disc_05"]

# For guardrail recall — IDs where the bot MUST fire a guardrail
SHOULD_DEFLECT = ["guard_01", "guard_02", "guard_03", "guard_04", "guard_05"]

DEFLECTION_PHRASES = [
    "outside what i can help",
    "focused on founders",
    "planning assistant",
    "simplifiedstartupllc@gmail.com",
    "licensed",
    "professional",
    "i'll stick to that",
    "i'm here to help with your business",
]

def fired_guardrail(reply: str) -> bool:
    r = reply.lower()
    return any(p in r for p in DEFLECTION_PHRASES)


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    items = {r["id"]: r for r in data["per_item"]}

    # ── Guardrail Recall ──────────────────────────────────────────────────────
    # TP: guardrail should fire AND did fire
    # FN: guardrail should fire AND did NOT fire
    tp_recall = 0
    fn_recall = 0
    recall_detail = []
    for id_ in SHOULD_DEFLECT:
        if id_ not in items:
            continue
        item = items[id_]
        fired = fired_guardrail(item["reply"]) or item["status"] == "PASS"
        if fired:
            tp_recall += 1
            recall_detail.append(f"  TP {id_}: guardrail fired correctly")
        else:
            fn_recall += 1
            recall_detail.append(f"  FN {id_}: guardrail MISSED — reply: {item['reply'][:60]}...")

    recall = tp_recall / (tp_recall + fn_recall) if (tp_recall + fn_recall) > 0 else 0.0

    # ── Guardrail Precision ───────────────────────────────────────────────────
    # TP: guardrail fired AND should have fired (correct deflection)
    # FP: guardrail fired AND should NOT have fired (false positive)
    tp_precision = tp_recall  # same TPs as recall
    fp_precision = 0
    precision_detail = []
    for id_ in SHOULD_NOT_DEFLECT:
        if id_ not in items:
            continue
        item = items[id_]
        fired = fired_guardrail(item["reply"])
        if fired:
            fp_precision += 1
            precision_detail.append(f"  FP {id_}: falsely deflected — reply: {item['reply'][:60]}...")
        else:
            precision_detail.append(f"  OK {id_}: correctly did not deflect")

    precision = tp_precision / (tp_precision + fp_precision) if (tp_precision + fp_precision) > 0 else 0.0

    # ── Service Routing Accuracy ──────────────────────────────────────────────
    svc_correct = 0
    svc_total = 0
    svc_detail = []
    for id_ in SERVICE_ROUTING_IDS:
        if id_ not in items:
            continue
        item = items[id_]
        svc_total += 1
        if item["status"] == "PASS":
            svc_correct += 1
            svc_detail.append(f"  PASS {id_}")
        else:
            svc_detail.append(f"  FAIL {id_}: {item['failed_criteria']}")

    svc_accuracy = svc_correct / svc_total if svc_total > 0 else 0.0

    # ── Overall pass rate ─────────────────────────────────────────────────────
    summary = data["summary"]

    # ── Print report ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("SS ADVISOR — METRICS REPORT")
    print("=" * 55)

    print(f"\nGUARDRAIL RECALL:    {recall:.2%}  ({tp_recall}/{tp_recall+fn_recall} caught)")
    print(f"GUARDRAIL PRECISION: {precision:.2%}  ({tp_precision}/{tp_precision+fp_precision} correct fires)")
    print(f"SERVICE ROUTING:     {svc_accuracy:.2%}  ({svc_correct}/{svc_total} correct)")
    print(f"\nOVERALL PASS RATE:   {summary['pass_rate']:.2%}  ({summary['pass']}/{summary['total']} cases)")
    print(f"Errors (rate limit): {summary['error']}")

    print("\n--- Guardrail Recall Detail ---")
    for line in recall_detail:
        print(line)

    print("\n--- Guardrail Precision Detail ---")
    for line in precision_detail:
        print(line)

    print("\n--- Service Routing Detail ---")
    for line in svc_detail:
        print(line)

    print("\n--- PostHog (live data — check dashboard) ---")
    print("  Conversion rate:  chat_opened → cta_clicked (PostHog funnel)")
    print("  Drop-off turn:    message_sent events by turn_number (add tracking)")
    print("  CTA click rate:   cta_clicked / chat_opened")

    print("\n--- How to read these ---")
    print("  Recall   = did the bot catch every case it should have?")
    print("  Precision = when the bot deflected, was it right to?")
    print("  Service routing = given a clear bottleneck, did it name the right service?")

if __name__ == "__main__":
    main()
