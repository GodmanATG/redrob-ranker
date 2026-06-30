"""
smoke_test.py
Tests every stage individually without requiring the cross-encoder model.
Validates that all our changes (synonyms, logging, type hints, Jinja2) work correctly.
"""
import sys
import logging
from datetime import date, timedelta

# Ensure we run from the project root
sys.path.insert(0, ".")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger("smoke_test")

PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} — {detail}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config: SYNONYMS exists and has expected keys
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 1. Config ===")
from config import SYNONYMS, CORE_SKILLS, JD_TEXT, WEIGHT_AI, WEIGHT_TFIDF, WEIGHT_EXPERIENCE, WEIGHT_SKILLS, WEIGHT_BEHAVIORAL

check("SYNONYMS dict exists",        isinstance(SYNONYMS, dict) and len(SYNONYMS) > 0)
check("SYNONYMS has 'llm' key",      "llm" in SYNONYMS)
check("SYNONYMS has 'ltr' key",      "ltr" in SYNONYMS)
check("Weights sum to 1.0",          abs(WEIGHT_AI + WEIGHT_TFIDF + WEIGHT_EXPERIENCE + WEIGHT_SKILLS + WEIGHT_BEHAVIORAL - 1.0) < 0.001)
check("CORE_SKILLS non-empty",       len(CORE_SKILLS) > 0)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF: synonym expansion
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 2. TF-IDF Stage: Synonym Expansion ===")
from stages.tfidf_search import expand_synonyms, build_candidate_document, run_tfidf_search

expanded = expand_synonyms("we use llm for generation tasks")
check("'llm' expands to include 'large language model'",
      "large language model" in expanded,
      f"got: {expanded}")

expanded2 = expand_synonyms("learning to rank models trained with xgboost")
check("'learning to rank' expands to include 'ltr'",
      "ltr" in expanded2,
      f"got: {expanded2}")

no_change = expand_synonyms("ordinary java backend developer")
check("no expansion for unrelated text",
      "llm" not in no_change and "ltr" not in no_change)

# Build a candidate document
mock_cand = {
    "profile": {"headline": "ML Engineer", "summary": "NLP expert", "current_title": "ML Engineer"},
    "career_history": [{"title": "ML Engineer", "description": "Built LLM pipelines"}],
    "skills": [{"name": "FAISS"}, {"name": "Pinecone"}],
}
doc = build_candidate_document(mock_cand)
check("build_candidate_document returns non-empty string", len(doc) > 10)
check("document is lowercased", doc == doc.lower())

# ─────────────────────────────────────────────────────────────────────────────
# 3. Honeypot filter
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 3. Honeypot Filter ===")
from stages.honeypot import is_honeypot_or_trap, filter_candidates

clean_cand = {
    "candidate_id": "CAND_0000001",
    "profile": {"years_of_experience": 6.0},
    "skills": [{"name": "FAISS", "proficiency": "expert", "duration_months": 36}],
    "career_history": [{"duration_months": 36}, {"duration_months": 36}],
}
trap_skill = {
    "candidate_id": "CAND_0000002",
    "profile": {"years_of_experience": 6.0},
    "skills": [{"name": "FAISS", "proficiency": "expert", "duration_months": 0}],  # TRAP
    "career_history": [{"duration_months": 12}],
}
trap_yoe = {
    "candidate_id": "CAND_0000003",
    "profile": {"years_of_experience": 2.0},   # TRAP: stated 2, history = 14
    "skills": [],
    "career_history": [{"duration_months": 84}, {"duration_months": 84}],
}

is_trap_clean, _ = is_honeypot_or_trap(clean_cand)
is_trap_skill, reason_skill = is_honeypot_or_trap(trap_skill)
is_trap_yoe, reason_yoe = is_honeypot_or_trap(trap_yoe)

check("Clean candidate passes honeypot",         not is_trap_clean)
check("Expert+0months skill is flagged as trap", is_trap_skill, reason_skill)
check("Impossible YOE is flagged as trap",       is_trap_yoe, reason_yoe)

# filter_candidates takes (cand, score) tuples
pairs = [(clean_cand, 0.9), (trap_skill, 0.8), (trap_yoe, 0.7)]
filtered = filter_candidates(pairs)
check("filter_candidates removes 2 traps leaving 1", len(filtered) == 1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Scorer
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 4. Scorer ===")
from stages.scorer import score_experience, score_skills, score_behavioral, score_location

check("YOE=7 scores 1.0",   score_experience(7) == 1.0)
check("YOE=2 scores 0.0",   score_experience(2) == 0.0)
check("YOE=16 scores 0.3",  score_experience(16) == 0.3)
check("YOE=4 scores < 1.0", score_experience(4) < 1.0)

skills_good = [{"name": "FAISS", "proficiency": "expert", "duration_months": 60}]
skills_none = []
check("Good skills score > 0",  score_skills(skills_good) > 0)
check("No skills scores 0.0",   score_skills(skills_none) == 0.0)

ghost_signals = {"last_active_date": (date.today() - timedelta(days=200)).isoformat()}
active_signals = {
    "last_active_date": (date.today() - timedelta(days=5)).isoformat(),
    "recruiter_response_rate": 0.9,
    "interview_completion_rate": 0.9,
    "offer_acceptance_rate": 0.8,
    "github_activity_score": 95,
    "notice_period_days": 15,
    "open_to_work_flag": True,
}
check("Ghost candidate scores 0.0", score_behavioral(ghost_signals) == 0.0)
check("Active candidate scores > 0.5", score_behavioral(active_signals) > 0.5)

profile_pune   = {"location": "Pune, Maharashtra"}
profile_remote = {"location": "Rajasthan"}
check("Pune location scores 1.0",         score_location(profile_pune, {}) == 1.0)
check("Unknown location scores 0.3",      score_location(profile_remote, {}) == 0.3)
check("Willing to relocate scores 0.8",   score_location(profile_remote, {"willing_to_relocate": True}) == 0.8)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Reasoning generator (Jinja2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 5. Reasoning Generator (Jinja2) ===")
from stages.reasoning import generate_reasoning

def make_full_candidate(cid, title, yoe, skills_data, github=80, rr=0.85, np=30):
    return {
        "candidate_id": cid,
        "profile": {
            "current_title": title,
            "years_of_experience": yoe,
            "location": "Pune",
        },
        "skills": skills_data,
        "career_history": [
            {"company": "DeepMind", "title": title,
             "description": "Built production retrieval systems."}
        ],
        "redrob_signals": {
            "recruiter_response_rate": rr,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "github_activity_score": github,
            "notice_period_days": np,
            "last_active_date": date.today().isoformat(),
            "open_to_work_flag": True,
        },
    }

tier1_cand = make_full_candidate(
    "CAND_0000010", "Senior AI Engineer", 7.0,
    [{"name": "FAISS", "proficiency": "expert", "duration_months": 48},
     {"name": "Pinecone", "proficiency": "advanced", "duration_months": 36}],
)
tier2_cand = make_full_candidate("CAND_0000020", "ML Engineer", 5.5, [])
tier3_cand = make_full_candidate("CAND_0000030", "Data Scientist", 4.0, [], rr=0.3)

r1 = generate_reasoning(tier1_cand, rank=1, score=0.95)
r2 = generate_reasoning(tier2_cand, rank=50, score=0.60)
r3 = generate_reasoning(tier3_cand, rank=90, score=0.40)

check("Tier 1 reasoning non-empty",           len(r1) > 30, f"got: '{r1}'")
check("Tier 2 reasoning non-empty",           len(r2) > 30, f"got: '{r2}'")
check("Tier 3 reasoning non-empty",           len(r3) > 30, f"got: '{r3}'")
check("Tier 1 reasoning is a plain string",   isinstance(r1, str))
check("No Jinja2 tags leaked into output",    "{{" not in r1 and "{%" not in r1, r1)
check("No Jinja2 tags in tier 3 output",      "{{" not in r3 and "{%" not in r3, r3)
# Whitespace cleanup: no consecutive spaces
check("No double-spaces in tier 1 reasoning", "  " not in r1, r1)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  SMOKE TEST COMPLETE: {len(PASS)} passed, {len(FAIL)} failed")
print(f"{'='*55}")
if FAIL:
    print("\nFailed tests:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("  All tests passed!")
