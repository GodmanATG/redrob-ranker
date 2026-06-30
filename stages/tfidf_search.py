import logging
import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import JD_TEXT, TFIDF_MAX_FEATURES, TOP_K_SEMANTIC, SYNONYMS

logger = logging.getLogger(__name__)

def expand_synonyms(text: str) -> str:
    """
    Finds acronyms/terms in text that exist in SYNONYMS and appends their expansions
    to address vocabulary staticness in TF-IDF.
    """
    text_lower = text.lower()
    expansions = []
    for key, val in SYNONYMS.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            if val not in expansions and val not in text_lower:
                expansions.append(val)
    if expansions:
        return text + " " + " ".join(expansions)
    return text

def build_candidate_document(candidate: Dict) -> str:
    """
    Constructs a rich text document for a candidate to be used in TF-IDF,
    expanded with common acronym/synonym mappings.
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")
    
    # Career history descriptions
    career_history = candidate.get("career_history", [])
    career_text = " ".join([c.get("title", "") + " " + c.get("description", "") for c in career_history])
    
    # Skills
    skills = candidate.get("skills", [])
    skills_text = " ".join([s.get("name", "") for s in skills])
    
    raw_doc = f"{headline} {current_title} {summary} {career_text} {skills_text}".lower()
    return expand_synonyms(raw_doc)

def run_tfidf_search(candidates: List[Dict]) -> List[Tuple[Dict, float]]:
    """
    Runs TF-IDF vectorization and cosine similarity against the JD.
    Returns the top K candidates with their semantic match scores.
    """
    logger.info("Building candidate documents for TF-IDF...")
    documents = [build_candidate_document(c) for c in candidates]
    
    logger.info(f"Initializing TF-IDF Vectorizer with max_features={TFIDF_MAX_FEATURES}...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english', ngram_range=(1, 3))
    
    # Synonym Injection: Create a dense gravity well of terms we care about
    # so anyone touching these words gets pulled into the top AI evaluation pool
    synonym_magnet = (
        "pinecone weaviate milvus qdrant pgvector chromadb "
        "vector database vector search semantic search hybrid search "
        "embeddings retrieval augmented generation llm nlp "
        "elasticsearch opensearch faiss ndcg mrr ranking ltr"
    ) * 3
    
    enhanced_jd = expand_synonyms((JD_TEXT + " " + synonym_magnet).lower())
    
    # Fit on candidates + enhanced JD
    all_texts = documents + [enhanced_jd]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    candidate_vectors = tfidf_matrix[:-1]
    jd_vector = tfidf_matrix[-1]
    
    logger.info("Computing cosine similarities between candidates and JD...")
    similarities = cosine_similarity(candidate_vectors, jd_vector).flatten()
    
    # Get top K indices
    top_indices = similarities.argsort()[-TOP_K_SEMANTIC:][::-1]
    
    top_candidates = []
    for idx in top_indices:
        candidate = candidates[idx]
        score = similarities[idx]
        top_candidates.append((candidate, float(score)))
        
    logger.info(f"TF-IDF search complete. Retained top {len(top_candidates)} candidates.")
    return top_candidates

