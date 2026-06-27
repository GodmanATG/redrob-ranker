"""
Audit & Extract Script
1. Validates submission.csv against hackathon spec
2. Checks how many honeypot/trap candidates are in the top 100
3. Extracts full JSON for all 100 ranked candidates from candidates.jsonl
"""
import csv
import json
import gzip
import os
import sys

CSV_PATH = os.path.join(os.path.dirname(__file__), "submission.csv")
JSONL_PATH = r"D:\H2S RedrobAI\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
JSONL_GZ_PATH = r"D:\H2S RedrobAI\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl.gz"
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "top100_full_data.json")

# ------ Consulting firms list (same as config.py) ------
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "tech mahindra",
    "hcl", "ibm", "mindtree"
}

RED_FLAG_TITLES = [
    "marketing", "sales", "hr", "human resources", "recruiter",
    "finance", "accountant", "operations", "mechanical", "electrical",
    "civil", "doctor", "nurse", "teacher", "designer", "content",
    "writer", "seo", "support"
]


def is_honeypot(candidate):
    """Same logic as stages/honeypot.py"""
    # 1. Expert skill with 0 duration
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0:
            return True, "Expert skill with 0 months duration"

    # 2. Career history checks
    career_history = candidate.get("career_history", [])
    total_yoe_from_jobs = 0
    all_consulting = True

    for job in career_history:
        total_yoe_from_jobs += job.get("duration_months", 0) / 12.0
        company = job.get("company", "").lower()
        if company not in CONSULTING_FIRMS:
            all_consulting = False

    # YOE mismatch
    profile = candidate.get("profile", {})
    stated_yoe = profile.get("years_of_experience", 0)
    if stated_yoe > 0 and total_yoe_from_jobs > stated_yoe * 1.5:
        return True, "Stated YOE drastically lower than career history"

    # Consulting-only
    if all_consulting and len(career_history) > 0:
        return True, "Only consulting experience"

    # Red flag titles
    title = profile.get("current_title", "").lower()
    for flag in RED_FLAG_TITLES:
        if flag in title:
            return True, f"Irrelevant title: {title}"

    return False, ""


def main():
    print("=" * 70)
    print("  SUBMISSION AUDIT REPORT")
    print("=" * 70)

    # ---------------------------------------------------------
    # PART 1: CSV Audit
    # ---------------------------------------------------------
    print("\n📋 PART 1: CSV Format Validation Against Spec\n")
    
    submission_file = "submission.csv"
    if not os.path.exists(submission_file):
        print(f"❌ Error: {submission_file} not found.")
        return

    with open(submission_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    # Check headers
    expected_headers = ["candidate_id", "rank", "score", "reasoning"]
    issues = []
    if headers != expected_headers:
        issues.append(f"❌ Headers mismatch. Expected {expected_headers}, got {headers}")
    else:
        print(f"  ✅ Headers correct: {headers}")

    # Check row count
    if len(rows) != 100:
        issues.append(f"❌ Expected 100 rows, got {len(rows)}")
    else:
        print(f"  ✅ Row count: {len(rows)}")

    # Check candidate_id format
    bad_ids = []
    ids_set = set()
    for r in rows:
        cid = r["candidate_id"]
        if not cid.startswith("CAND_"):
            bad_ids.append(cid)
        if cid in ids_set:
            issues.append(f"❌ Duplicate candidate_id: {cid}")
        ids_set.add(cid)
    if bad_ids:
        issues.append(f"❌ IDs not matching CAND_XXXXXXX format: {bad_ids[:5]}")
    else:
        print(f"  ✅ All candidate IDs match CAND_XXXXXXX format")
    print(f"  ✅ All candidate IDs are unique ({len(ids_set)} unique)")

    # Check ranks: 1-100, each used exactly once
    ranks = [int(r["rank"]) for r in rows]
    expected_ranks = list(range(1, 101))
    if sorted(ranks) != expected_ranks:
        missing = set(expected_ranks) - set(ranks)
        extra = set(ranks) - set(expected_ranks)
        issues.append(f"❌ Ranks not 1-100. Missing: {missing}, Extra: {extra}")
    else:
        print(f"  ✅ Ranks 1-100 each used exactly once")

    # Check scores are monotonically non-increasing
    scores = [float(r["score"]) for r in rows]
    monotonic = True
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            monotonic = False
            issues.append(f"❌ Score NOT monotonically non-increasing at rank {i+1}: {scores[i]} > {scores[i-1]}")
            break
    if monotonic:
        print(f"  ✅ Scores are monotonically non-increasing")

    # Check tie-break rule: same score → candidate_id ascending
    tie_violations = 0
    for i in range(1, len(rows)):
        if float(rows[i]["score"]) == float(rows[i - 1]["score"]):
            if rows[i]["candidate_id"] < rows[i - 1]["candidate_id"]:
                tie_violations += 1
                issues.append(f"  ⚠️  Tie-break violation at ranks {i}/{i+1}: "
                              f"{rows[i-1]['candidate_id']} ({rows[i-1]['score']}) vs "
                              f"{rows[i]['candidate_id']} ({rows[i]['score']})")
    if tie_violations == 0:
        print(f"  ✅ Tie-break rule satisfied (score desc, then candidate_id asc)")

    # Check reasoning exists
    reasoning_count = sum(1 for r in rows if r.get("reasoning", "").strip())
    print(f"  ✅ Reasoning provided for {reasoning_count}/100 candidates")

    # Score range
    print(f"\n  📊 Score range: {min(scores):.4f} → {max(scores):.4f}")
    print(f"  📊 Mean score: {sum(scores)/len(scores):.4f}")

    if issues:
        print(f"\n  ❌ {len(issues)} ISSUE(S) FOUND:")
        for iss in issues:
            print(f"    {iss}")
    else:
        print(f"\n  ✅✅ ALL CHECKS PASSED — CSV is fully compliant with the spec!")

    # ---- PART 2: Honeypot Check ----
    print("\n" + "=" * 70)
    print("🍯 PART 2: Honeypot / Trap Check on Top 100")
    print("=" * 70 + "\n")

    # Load the 100 candidate IDs from the CSV
    top100_ids = set(r["candidate_id"] for r in rows)

    # We need to find these candidates in the JSONL and check them
    # Determine which JSONL file to use
    jsonl_file = JSONL_PATH if os.path.exists(JSONL_PATH) else JSONL_GZ_PATH
    if not os.path.exists(jsonl_file):
        print(f"  ❌ Could not find candidates file at {JSONL_PATH} or {JSONL_GZ_PATH}")
        return

    print(f"  Loading candidates from: {jsonl_file}")

    opener = gzip.open if jsonl_file.endswith('.gz') else open
    found_candidates = {}
    total_scanned = 0

    with opener(jsonl_file, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_scanned += 1
            cand = json.loads(line)
            cid = cand.get("candidate_id", "")
            if cid in top100_ids:
                found_candidates[cid] = cand
            if total_scanned % 20000 == 0:
                print(f"    Scanned {total_scanned} candidates... found {len(found_candidates)}/100 so far")

    print(f"  Scanned {total_scanned} total candidates")
    print(f"  Found {len(found_candidates)}/100 of the ranked candidates in JSONL")

    # Check each for honeypot
    honeypots_in_top100 = []
    for cid, cand in found_candidates.items():
        is_trap, reason = is_honeypot(cand)
        if is_trap:
            rank_row = next((r for r in rows if r["candidate_id"] == cid), None)
            rank_num = int(rank_row["rank"]) if rank_row else "?"
            honeypots_in_top100.append((cid, rank_num, reason))

    if honeypots_in_top100:
        print(f"\n  ⚠️  {len(honeypots_in_top100)} HONEYPOT/TRAP CANDIDATES FOUND IN TOP 100:")
        for cid, rank, reason in sorted(honeypots_in_top100, key=lambda x: x[1]):
            print(f"    Rank {rank:3d} | {cid} | Reason: {reason}")
    else:
        print(f"\n  ✅ ZERO honeypot/trap candidates in the top 100! Clean submission.")

    # ---- PART 3: Extract full JSON data ----
    print("\n" + "=" * 70)
    print("📦 PART 3: Extracting Full JSON Data for Top 100")
    print("=" * 70 + "\n")

    # Build ordered list matching CSV rank order
    ordered_data = []
    for r in rows:
        cid = r["candidate_id"]
        cand_data = found_candidates.get(cid, {})
        ordered_data.append({
            "rank": int(r["rank"]),
            "score": float(r["score"]),
            "reasoning": r.get("reasoning", ""),
            "candidate_data": cand_data
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ordered_data, f, indent=2, ensure_ascii=False)

    file_size_mb = os.path.getsize(OUTPUT_JSON) / (1024 * 1024)
    print(f"  ✅ Wrote full JSON data to: {OUTPUT_JSON}")
    print(f"  📊 File size: {file_size_mb:.2f} MB")
    print(f"  📊 Contains {len(ordered_data)} candidates with full profile, skills, career_history, redrob_signals, etc.")

    # Quick summary of what's in each candidate record
    if ordered_data and ordered_data[0].get("candidate_data"):
        sample = ordered_data[0]["candidate_data"]
        print(f"\n  📋 Fields in each candidate record:")
        for key in sample.keys():
            val = sample[key]
            if isinstance(val, list):
                print(f"    • {key}: list ({len(val)} items)")
            elif isinstance(val, dict):
                print(f"    • {key}: dict ({len(val)} keys)")
            else:
                print(f"    • {key}: {type(val).__name__}")

    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
