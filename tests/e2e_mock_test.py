"""
e2e_mock_test.py
End-to-end test that injects fake torch and sentence_transformers modules into
sys.modules before any import — no packages need to be installed on disk.
"""
import sys
import csv
import types
import logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger("e2e_test")

# ── Inject a minimal fake 'torch' module ─────────────────────────────────────
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = fake_torch

# ── Inject a minimal fake 'sentence_transformers' module ─────────────────────
def fake_predict(pairs, batch_size=32):
    """Lightweight word-overlap scorer that replaces the CrossEncoder."""
    from config import JD_TEXT
    jd_words = set(JD_TEXT.lower().split())
    scores = []
    for jd, cand in pairs:
        overlap = len(jd_words & set(cand.lower().split()))
        scores.append(float(overlap))
    return scores

class FakeCrossEncoder:
    def __init__(self, model_name, device="cpu"):
        pass
    def predict(self, pairs, batch_size=32):
        return fake_predict(pairs, batch_size)

fake_st = types.ModuleType("sentence_transformers")
fake_st.CrossEncoder = FakeCrossEncoder
sys.modules["sentence_transformers"] = fake_st

# ── Now safe to import pipeline modules ──────────────────────────────────────


# ── Validate the CSV ──────────────────────────────────────────────────────────
print("\n=== CSV Validation ===")
PASS, FAIL = [], []

def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} — {detail}")

with open(out_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames

check("Headers correct",        headers == ["candidate_id", "rank", "score", "reasoning"])
check("Exactly 100 rows",       len(rows) == 100, f"got {len(rows)}")
check("All IDs start CAND_",    all(r["candidate_id"].startswith("CAND_") for r in rows))
check("IDs are unique",         len(set(r["candidate_id"] for r in rows)) == 100)

ranks = [int(r["rank"]) for r in rows]
check("Ranks 1-100 each used once", sorted(ranks) == list(range(1, 101)))

scores = [float(r["score"]) for r in rows]
monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
check("Scores monotonically non-increasing", monotonic)

check("All reasoning non-empty", all(r["reasoning"].strip() for r in rows))
check("No Jinja2 tags in any reasoning", all("{{" not in r["reasoning"] for r in rows))

# Confirm traps/honeypots are NOT in top 100
top_ids = {r["candidate_id"] for r in rows}
# Honeypot IDs should be CAND_0000301 to CAND_0000310 (indices 300-309)
honeypot_ids = {f"CAND_{i:07d}" for i in range(301, 311)}
honeypots_in_top100 = honeypot_ids & top_ids
check(
    "No honeypots in top 100",
    len(honeypots_in_top100) == 0,
    f"found: {honeypots_in_top100}"
)

print(f"\n{'='*45}")
print(f"  E2E TEST: {len(PASS)} passed, {len(FAIL)} failed")
print(f"{'='*45}")
if FAIL:
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("  Full pipeline validated successfully!")
    # Show a sample row
    print(f"\n  Sample top-1 reasoning:\n  \"{rows[0]['reasoning'][:200]}\"")
