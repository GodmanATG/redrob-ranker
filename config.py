# config.py

JD_TEXT = """
Job Description: Senior AI Engineer — Founding Team
Location: Pune/Noida, India (Hybrid) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years

What you'd actually be doing:
- Own the intelligence layer of Redrob's product (ranking, retrieval, matching systems).
- Audit BM25 + rule-based scoring.
- Ship v2 ranking system (embeddings, hybrid retrieval, LLM-based re-ranking).
- Set up evaluation infrastructure (offline benchmarks, A/B testing).

Things you absolutely need:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, etc.) deployed to real users.
- Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
- Strong Python.
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).

Things we'd like you to have:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting tech, or marketplace products

Disqualifiers:
- Pure research environments (academic labs, research-only roles) without production deployment.
- "AI experience" only consisting of recent (under 12 months) projects using LangChain to call OpenAI, without pre-LLM-era ML production experience.
- Senior engineers who haven't written production code in the last 18 months ("architecture" or "tech lead" roles).
- Title-chasers switching companies every 1.5 years.
- Framework enthusiasts (only LangChain tutorials, etc.).
- People who have ONLY worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) without product-company experience.
- People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure.
"""

# Text matching settings
TFIDF_MAX_FEATURES = 10000
TOP_K_SEMANTIC = 3000
AI_RERANK_TOP_K = 800

# Scoring Weights
WEIGHT_AI = 0.40
WEIGHT_TFIDF = 0.15
WEIGHT_EXPERIENCE = 0.20
WEIGHT_SKILLS = 0.15
WEIGHT_BEHAVIORAL = 0.10
WEIGHT_LOCATION = 0.0

# Experience Bands
EXP_MIN_TARGET = 5.0
EXP_MAX_TARGET = 9.0

# Core Required Skills (for reasoning and targeted matching)
CORE_SKILLS = [
    "embedding", "retrieval", "vector", "faiss", "pinecone", 
    "qdrant", "milvus", "weaviate", "elasticsearch", "opensearch",
    "python", "ndcg", "mrr", "map", "ranking", "search"
]
