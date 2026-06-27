import argparse
import json
import csv
import sys
import gzip

from stages.tfidf_search import run_tfidf_search
from stages.honeypot import filter_candidates
from stages.ai_reranker import run_ai_reranker
from stages.reasoning import generate_reasoning
from config import AI_RERANK_TOP_K

def load_candidates(filepath):
    print("Loading candidates...")
    candidates = []
    
    # Handle both gzipped and plain jsonl
    opener = gzip.open if filepath.endswith('.gz') else open
    
    with opener(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
                
    print(f"Loaded {len(candidates)} candidates.")
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    args = parser.parse_args()

    # 1. Load Data
    candidates = load_candidates(args.candidates)
    if not candidates:
        print("No candidates loaded.")
        sys.exit(1)

    # 2. Stage 1: Fast Semantic Filter (TF-IDF)
    print("Running Stage 1: TF-IDF text matching...")
    top_semantic = run_tfidf_search(candidates)

    # 3. Stage 2: Honeypot & Trap Detection
    print("Running Stage 2: Trap filtering...")
    valid_candidates = filter_candidates(top_semantic)
    
    # Keep only the top K for the AI reranker to stay within time budget
    top_k_for_ai = valid_candidates[:AI_RERANK_TOP_K]

    # 4. Stage 3: AI Re-Ranking & Blended Scoring
    print(f"Running Stage 3: AI Reranking (top {len(top_k_for_ai)})...")
    scored = run_ai_reranker(top_k_for_ai)
    
    # Round the score to exactly what will be in the CSV so sorting matches validator
    scored = [(cand, round(score, 4)) for cand, score in scored]
    
    # Sort by final score descending, then candidate_id ascending (to break ties deterministically)
    scored.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))
    
    # Take top 100
    top_100 = scored[:100]

    # 5. Stage 4: Reasoning Generation & CSV Export
    print(f"Running Stage 4: Generating reasoning and exporting to {args.out}...")
    
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for idx, (cand, score) in enumerate(top_100):
            rank = idx + 1
            reasoning = generate_reasoning(cand, rank, score)
            writer.writerow([cand.get("candidate_id"), rank, score, reasoning])
            
    print("Success! Ranked 100 candidates.")

if __name__ == "__main__":
    main()
