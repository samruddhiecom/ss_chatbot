"""Schema, stage transitions, and deterministic KB chunk parsing. No model, no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DRAFT_KB_DIR
from app.kb.ingest import parse_file
from app.schemas import Snapshot, Stage, next_stage


def test_stage_transitions_in_order():
    assert next_stage(Stage.IDEA) == Stage.MARKET_CUSTOMER
    assert next_stage(Stage.MARKET_CUSTOMER) == Stage.OFFER_CHANNELS
    assert next_stage(Stage.OFFER_CHANNELS) == Stage.OPERATIONS
    assert next_stage(Stage.OPERATIONS) == Stage.DONE
    assert next_stage(Stage.DONE) == Stage.DONE


def test_snapshot_schema_defaults():
    snap = Snapshot(
        idea_framing="x", customer_hypothesis="y", offer_sketch="z",
    )
    assert snap.channel_shortlist == []
    assert snap.named_gaps == []
    assert snap.next_steps == []


def test_parse_layer_b_chunks_have_metadata():
    chunks = parse_file(DRAFT_KB_DIR / "layer_b_company.md")
    assert len(chunks) >= 5
    for meta, text in chunks:
        assert meta.get("layer") == "B"
        assert "type" in meta
        assert text.strip()


def test_parse_layer_a_has_deferral_language():
    chunks = parse_file(DRAFT_KB_DIR / "layer_a_guidance.md")
    types = {meta.get("type") for meta, _ in chunks}
    assert "deferral_language" in types
    assert "guidance" in types
