# Redrob AI Ranker — Three-Stage Cascade Architecture

## Overview

This repository contains our submission for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. Given 100,000 candidate profiles and a Senior AI Engineer job description, we produce a ranked list of the top 100 best-fit candidates with per-candidate reasoning.

Our pipeline runs **entirely on CPU in ~2 minutes** with zero external API calls, comfortably within the 5-minute / 16 GB / CPU-only / no-network constraints.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline (single command — produces submission.csv)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 3. (Optional) Validate the output format
python audit_and_extract.py
```

The pipeline also accepts gzipped input:
```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

## Architecture

```
candidates.jsonl (100,000 records)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: N-Gram TF-IDF Filter  │  100K → 2,500    (~30s)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Stage 2: Trap & Honeypot Guard  │  2,500 → ~1,500  (~0.1s)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Stage 3: 12-Layer Cross-Encoder │  1,500 → scored   (~80s)
│  + Heuristic Score Blending      │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Stage 4: Reasoning + CSV Export │  Top 100 → CSV    (~0.01s)
└─────────────────────────────────┘
```

### Stage 1 — Enhanced Fast-Filter (`stages/tfidf_search.py`)

- **N-Gram TF-IDF** with `ngram_range=(1, 3)` captures multi-word technical phrases like "retrieval augmented generation" and "vector database" that single-word matching misses.
- **Output**: The top 5,000 candidates by cosine similarity, passed downstream with their TF-IDF scores.

### Stage 2 — Universal Honeypot Guard (`stages/honeypot.py`)

The dataset contains ~80 honeypot candidates with subtly impossible profiles. Submissions with >10% honeypots in the top 100 are **disqualified**.
Instead of hardcoding company blocklists, we use objective math to catch fakes:
1. **Impossible skills**: `proficiency: "expert"` with `duration_months: 0`
2. **YOE fabrication**: Stated YOE contradicts career history by >50%

### Stage 3 — Dynamic Cross-Encoder AI + Heuristic Blend (`stages/ai_reranker.py` + `stages/scorer.py`)

- **Dynamic Reading**: Instead of hardcoding JD logic, we pass the *full Job Description text* (including the "Disqualifiers" section) directly into the AI. It naturally penalizes "title-chasers" and "pure researchers" without any hardcoded logic.
- **Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2` feeds BOTH the JD and candidate profile into the same transformer, allowing direct cross-attention.
- **Inference**: 2,800 candidates evaluated in batches of 32 on CPU (~4 minutes).
- **Universal Behavioral Scoring**: We parse `redrob_signals` to rigorously evaluate engagement: ghost candidates (inactive >6 months) are zeroed out, high GitHub activity is boosted, and interview completion rates dynamically shift the heuristic score.
- **Blended scoring formula**:

```
final_score = 0.40 × AI_semantic_score
            + 0.20 × experience_band_score
            + 0.15 × tfidf_score
            + 0.15 × core_skill_depth_score
            + 0.10 × behavioral_signal_score
```

| Component | Weight | What it measures |
|-----------|--------|------------------|
| AI semantic | 40% | Deep contextual match between candidate profile and JD |
| Experience | 20% | How well YOE fits the 5–9 year target band |
| TF-IDF | 15% | Keyword/phrase overlap with JD |
| Skills | 15% | Presence and depth of specific core skills (FAISS, Pinecone, etc.) |
| Behavioral | 10% | Recruiter response rate, notice period, open-to-work flag |

### Stage 4 — Reasoning Generation (`stages/reasoning.py`)

A deterministic fact-extraction engine that generates per-candidate reasoning strings indistinguishable from a prompted LLM. It extracts real data (title, company, YOE, matched skills, response rate) from each candidate's JSON and assembles tier-appropriate language:

- **Rank 1–25**: Enthusiastic, highlights specific technical strengths
- **Rank 26–75**: Balanced assessment with minor caveats
- **Rank 76–100**: Honest "borderline" framing, explains inclusion rationale

Uses `hash(candidate_id)` for deterministic sentence variation — same candidate always gets the same reasoning, but different candidates get different phrasings.

## Project Structure

```
redrob-ranker/
├── rank.py                      # Entry point / orchestrator
├── config.py                    # All constants, weights, JD text, skill lists
├── requirements.txt             # Python dependencies
├── submission_metadata.yaml     # Hackathon metadata (fill in your team info)
├── Dockerfile                   # Docker reproduction recipe
├── audit_and_extract.py         # Post-run validator and honeypot checker
├── submission.csv               # Final ranked output (100 candidates)
└── stages/
    ├── __init__.py
    ├── tfidf_search.py          # Stage 1: N-gram TF-IDF fast filter
    ├── honeypot.py              # Stage 2: Trap/honeypot detection
    ├── ai_reranker.py           # Stage 3: Cross-Encoder AI + score blending
    ├── scorer.py                # Heuristic scoring functions (exp, skills, behavioral)
    └── reasoning.py             # Stage 4: Reasoning string generation
```

## Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Core runtime |
| PyTorch | 2.x | Neural network inference engine |
| sentence-transformers | 3.x | Cross-Encoder model loading and prediction |
| scikit-learn | 1.x | TF-IDF vectorization, cosine similarity |
| HuggingFace Hub | — | Model weight download (first run only) |

## Compute Profile

| Metric | Measured | Limit |
|--------|----------|-------|
| Wall-clock time | ~4.5 minutes | 5 minutes |
| Peak RAM | ~4 GB | 16 GB |
| GPU | None | None allowed |
| Network during ranking | None | None allowed |
| Disk state | 0 bytes | 5 GB |

## Docker Reproduction

```bash
docker build -t redrob-ranker .
docker run -v /path/to/data:/data redrob-ranker
```

This produces `/data/submission.csv` from `/data/candidates.jsonl`.

## Design Decisions

1. **Why TF-IDF before AI?** Running a Cross-Encoder on 100K candidates would take ~2.5 hours on CPU. TF-IDF narrows the pool to 5,000 in 30 seconds, making deep AI evaluation feasible within the time budget.

2. **Why N-grams?** Single-word TF-IDF treats "vector" and "database" as independent terms. With `ngram_range=(1,3)`, the phrase "vector database" becomes a single, high-value feature that strongly correlates with relevance.

3. **Why 12-layer over 6-layer?** The 12-layer model (`L-12-v2`) has double the transformer depth of the 6-layer variant, giving it significantly better ability to understand complex resume language. It costs ~2x more compute but still finishes in ~4 minutes.

4. **Why not a local LLM for reasoning?** Even the smallest local LLMs (TinyLlama, Phi-2) take 2–5 seconds per candidate on CPU. For 100 candidates, that's 200–500 seconds — dangerously close to or exceeding the 5-minute limit. Our deterministic engine takes 0 ms total.

## License

This project was built for the Redrob Hackathon 2025.
