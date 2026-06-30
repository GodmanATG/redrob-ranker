"""
generate_mock_candidates.py
Generates a mock candidates.jsonl file with 120 records for smoke-testing.
Includes:
 - Valid strong-match AI/ML engineers (should rank high)
 - Honeypot traps (expert skill + 0 months, impossible YOE)
 - Consulting-only candidates (should score low / be filtered)
 - Ghost candidates (inactive > 180 days)
 - Junior candidates (YOE < 3, penalised by scorer)
"""
import json
import random

random.seed(42)

def make_candidate(cid, title, yoe, skills, career_history, signals, location="Pune, India"):
    return {
        "candidate_id": f"CAND_{cid:07d}",
        "profile": {
            "current_title": title,
            "years_of_experience": yoe,
            "location": location,
            "headline": f"{title} with {yoe} years of experience",
            "summary": f"Experienced {title} specialising in {', '.join(s['name'] for s in skills[:3])}.",
        },
        "skills": skills,
        "career_history": career_history,
        "redrob_signals": signals,
    }

def skill(name, prof="advanced", months=36):
    return {"name": name, "proficiency": prof, "duration_months": months}

def job(title, company, months=24, desc=""):
    return {"title": title, "company": company, "duration_months": months, "description": desc}

def signals(rr=0.8, icr=0.85, oar=0.7, github=90, np=30, otw=True, active_days_ago=10):
    from datetime import date, timedelta
    last_active = (date.today() - timedelta(days=active_days_ago)).isoformat()
    return {
        "recruiter_response_rate": rr,
        "interview_completion_rate": icr,
        "offer_acceptance_rate": oar,
        "github_activity_score": github,
        "notice_period_days": np,
        "open_to_work_flag": otw,
        "last_active_date": last_active,
        "willing_to_relocate": True,
    }

candidates = []

# ── 60 strong AI/ML engineers (expected top rankers) ──────────────────────────
strong_skills_pool = [
    "FAISS", "Pinecone", "Qdrant", "Weaviate", "Elasticsearch", "OpenSearch",
    "sentence-transformers", "Python", "PyTorch", "NLP", "LLM Fine-Tuning",
    "Hybrid Search", "Embeddings", "Retrieval Augmented Generation",
    "NDCG", "MRR", "Learning to Rank", "Vector Database",
]

for i in range(60):
    yoe = round(random.uniform(5.0, 9.0), 1)
    chosen_skills = random.sample(strong_skills_pool, k=random.randint(4, 8))
    sk = [skill(s, prof=random.choice(["advanced", "expert"]), months=random.randint(18, 60)) for s in chosen_skills]
    ch = [
        job("Senior AI Engineer", "Qdrant Labs", 30,
            "Built vector search retrieval systems using FAISS and Pinecone. Improved NDCG@10 by 18%."),
        job("ML Engineer", "Zepto", 24,
            "Deployed sentence-transformer embeddings pipeline for product search. Reduced latency by 40%."),
        job("Data Scientist", "Flipkart", 18,
            "Designed hybrid BM25+dense retrieval ranking system. Implemented LTR using XGBoost."),
    ]
    s = signals(rr=round(random.uniform(0.7, 1.0), 2),
                github=random.randint(60, 100),
                np=random.randint(15, 45),
                active_days_ago=random.randint(1, 30))
    candidates.append(make_candidate(i + 1, "Senior AI Engineer", yoe, sk, ch, s))

# ── 20 mediocre candidates (expected mid-tier) ────────────────────────────────
for i in range(20):
    yoe = round(random.uniform(3.0, 5.0), 1)
    sk = [skill("Python", "intermediate", 36), skill("TensorFlow", "intermediate", 24)]
    ch = [job("ML Engineer", "InfoEdge", 24, "Worked on recommendation system using collaborative filtering.")]
    s = signals(rr=0.5, github=40, np=60, active_days_ago=45)
    candidates.append(make_candidate(100 + i, "ML Engineer", yoe, sk, ch, s))

# ── 15 ghost candidates (inactive > 180 days, should score 0 behavioural) ────
for i in range(15):
    yoe = round(random.uniform(5.0, 8.0), 1)
    sk = [skill("FAISS", "expert", 48), skill("Python", "expert", 60)]
    ch = [job("ML Researcher", "IIT Research Lab", 48, "Pure research on dense retrieval without production deployment.")]
    s = signals(rr=0.3, github=20, np=90, active_days_ago=200)  # ghost
    candidates.append(make_candidate(200 + i, "ML Researcher", yoe, sk, ch, s))

# ── 10 honeypot traps (expert skill + 0 months duration, impossible YOE) ─────
for i in range(10):
    if i < 5:
        # Trap type 1: expert with 0 months duration
        sk = [{"name": "FAISS", "proficiency": "expert", "duration_months": 0}]
        ch = [job("AI Lead", "FakeStartup", 12, "Built AGI.")]
        s = signals(rr=0.9, github=95, np=10)
    else:
        # Trap type 2: stated YOE wildly lower than career history
        sk = [skill("Python", "advanced", 24)]
        ch = [
            job("Software Engineer", "TCS", 60, "Worked on Java backend."),
            job("Senior Engineer", "Wipro", 72, "Led team of 10."),
            job("Architect", "Infosys", 60, "Enterprise architecture."),
        ]
        s = signals(rr=0.6, github=30, np=90)
    trap = make_candidate(300 + i, "AI Lead", 2.0, sk, ch, s)  # stated YOE=2 but history=16+
    candidates.append(trap)

# ── 15 consulting-only candidates ─────────────────────────────────────────────
consulting_firms = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant"]
for i in range(15):
    yoe = round(random.uniform(5.0, 10.0), 1)
    sk = [skill("Java", "advanced", 60), skill("SQL", "expert", 48)]
    ch = [
        job("Senior Consultant", random.choice(consulting_firms), 48,
            "Business process automation and SAP ERP configuration."),
        job("Consultant", random.choice(consulting_firms), 36, "Delivered client Java solutions."),
    ]
    s = signals(rr=0.5, github=10, np=90, active_days_ago=60)
    candidates.append(make_candidate(400 + i, "Senior Consultant", yoe, sk, ch, s))

# Shuffle so honeypots aren't always at the end
random.shuffle(candidates)

output_path = "mock_candidates.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for c in candidates:
        f.write(json.dumps(c) + "\n")

print(f"Written {len(candidates)} mock candidates to {output_path}")
print(f"  Strong AI/ML engineers: 60")
print(f"  Mediocre candidates:    20")
print(f"  Ghost candidates:       15")
print(f"  Honeypot traps:         10")
print(f"  Consulting-only:        15")
