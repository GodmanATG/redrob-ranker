import argparse
import json
import csv
import gzip
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple

from stages.tfidf_search import run_tfidf_search
from stages.honeypot import filter_candidates
from stages.ai_reranker import run_ai_reranker
from stages.reasoning import generate_reasoning
from config import AI_RERANK_TOP_K

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("rank")


# ── Data loading ──────────────────────────────────────────────────────────────
def load_candidates(filepath: str) -> List[Dict]:
    """Load candidates from a plain or gzip-compressed JSONL file."""
    logger.info(f"Loading candidates from: {filepath}")
    candidates: List[Dict] = []

    opener = gzip.open if filepath.endswith(".gz") else open
    try:
        with opener(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    except FileNotFoundError:
        logger.error(f"Candidates file not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSONL: {exc}")
        sys.exit(1)

    logger.info(f"Loaded {len(candidates):,} candidates.")
    return candidates


# ── CSV export ────────────────────────────────────────────────────────────────
def export_csv(top_100: List[Tuple[Dict, float]], output_path: str) -> None:
    """Write the ranked top-100 with reasoning to a CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, (cand, score) in enumerate(top_100):
            rank = idx + 1
            reasoning = generate_reasoning(cand, rank, score)
            writer.writerow([cand.get("candidate_id"), rank, score, reasoning])
    logger.info(f"Results written to: {output_path}")


# ── Pipeline ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob Hackathon Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz]")
    parser.add_argument("--out", required=True, help="Path for output submission.csv")
    args = parser.parse_args()

    pipeline_start = time.time()

    # 1. Load data ─────────────────────────────────────────────────────────────
    candidates = load_candidates(args.candidates)
    if not candidates:
        logger.error("No candidates loaded. Exiting.")
        sys.exit(1)

    # 2. Stage 1: TF-IDF fast filter ──────────────────────────────────────────
    t0 = time.time()
    logger.info("Stage 1 | N-Gram TF-IDF + synonym-expanded fast filter...")
    top_semantic = run_tfidf_search(candidates)
    logger.info(f"Stage 1 | Done in {time.time() - t0:.1f}s — {len(top_semantic):,} candidates retained.")

    # 3. Stage 2: Honeypot / trap filter ──────────────────────────────────────
    t0 = time.time()
    logger.info("Stage 2 | Honeypot & trap detection...")
    valid_candidates = filter_candidates(top_semantic)
    logger.info(
        f"Stage 2 | Done in {time.time() - t0:.1f}s — "
        f"{len(top_semantic) - len(valid_candidates):,} traps removed, "
        f"{len(valid_candidates):,} candidates remaining."
    )

    # Keep only the top K for AI reranker to stay within the time budget
    top_k_for_ai = valid_candidates[:AI_RERANK_TOP_K]

    # 4. Stage 3: AI cross-encoder re-ranking + heuristic blend ───────────────
    t0 = time.time()
    logger.info(f"Stage 3 | Cross-Encoder AI reranking on top {len(top_k_for_ai):,} candidates...")
    scored = run_ai_reranker(top_k_for_ai)
    logger.info(f"Stage 3 | Done in {time.time() - t0:.1f}s.")

    # Round scores for deterministic tie-breaking
    scored = [(cand, round(score, 4)) for cand, score in scored]

    # Sort: score descending, then candidate_id ascending to break ties deterministically
    scored.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))

    top_100 = scored[:100]

    # 5. Stage 4: Reasoning generation + CSV export ───────────────────────────
    t0 = time.time()
    logger.info(f"Stage 4 | Generating reasoning and exporting to {args.out}...")
    export_csv(top_100, args.out)
    logger.info(f"Stage 4 | Done in {time.time() - t0:.1f}s.")

    total = time.time() - pipeline_start
    logger.info(
        f"Success! Ranked {len(top_100)} candidates in {total:.1f}s total. "
        f"Output: {args.out}"
    )


if __name__ == "__main__":
    main()

