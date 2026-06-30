import logging
import re
import torch
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from config import JD_TEXT, WEIGHT_AI, WEIGHT_TFIDF, WEIGHT_EXPERIENCE, WEIGHT_SKILLS, WEIGHT_BEHAVIORAL, WEIGHT_LOCATION
from stages.scorer import score_experience, score_skills, score_behavioral, score_location

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_model = None

def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading Cross-Encoder model ({MODEL_NAME})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        _model = CrossEncoder(MODEL_NAME, device=device)
    return _model

# Pre-compile vocabulary from JD (excluding common stop words) for quick relevance scoring
stop_words = {'the', 'and', 'a', 'of', 'to', 'in', 'is', 'that', 'it', 'for', 'on', 'with', 'as', 'this', 'our', 'you', 'your', 'we'}
jd_words = set(re.findall(r'\b[a-z]{3,}\b', JD_TEXT.lower())) - stop_words

def get_job_relevance(job: Dict, jd_words: set) -> int:
    """
    Scores the relevance of a career history job based on overlap with JD terms.
    """
    job_text = (job.get('title', '') + " " + job.get('description', '')).lower()
    words = set(re.findall(r'\b[a-z]{3,}\b', job_text))
    return len(words.intersection(jd_words))

def format_candidate_for_ai(candidate: Dict) -> str:
    """
    Format the candidate's profile into a dense string for the CrossEncoder.
    We prioritize title, YOE, companies, and skills.
    To prevent token truncation of important details, we rank career history
    by JD relevance and keep the current job + top 2 most relevant past jobs.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    summary = profile.get("summary", "")
    
    # Smart Career History Selection: Lock in the current/most recent job,
    # and fill the remaining slots with the most relevant past roles.
    selected_jobs = []
    if history:
        # Lock in current job (always the first entry in career history)
        selected_jobs.append(history[0])
        
        # Sort remaining jobs by relevance and select top 2
        if len(history) > 1:
            other_jobs = history[1:]
            other_jobs_sorted = sorted(
                other_jobs,
                key=lambda j: get_job_relevance(j, jd_words),
                reverse=True
            )
            selected_jobs.extend(other_jobs_sorted[:2])
            
    # Format selected jobs (allow 250 characters each since we have selected the most relevant)
    jobs_text = " | ".join([
        f"{j.get('title', '')} at {j.get('company', '')}: {j.get('description', '')[:250]}"
        for j in selected_jobs
    ])
    
    # High-proficiency skills
    top_skills = [s.get('name', '') for s in skills if s.get('proficiency') in ['advanced', 'expert']]
    skills_text = ", ".join(top_skills)
    
    text = f"Candidate Profile:\nTitle: {title}\nYears of Experience: {yoe}\nSummary: {summary}\nKey Jobs: {jobs_text}\nTop Skills: {skills_text}"
    return text

def run_ai_reranker(scored_candidates: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
    """
    Takes the top K candidates from the TF-IDF stage, formats them,
    and runs a real local Cross-Encoder AI over them to get a deep semantic score.
    Returns: list of (candidate, final_blended_score)
    """
    if not scored_candidates:
        return []
        
    model = get_model()
    
    # Build pairs of (JD, Candidate Profile)
    pairs = []
    max_tfidf = max(s for _, s in scored_candidates) if scored_candidates else 1.0
    if max_tfidf == 0: max_tfidf = 1.0
    
    for cand, tfidf in scored_candidates:
        cand_text = format_candidate_for_ai(cand)
        pairs.append((JD_TEXT, cand_text))
        
    logger.info(f"Running Cross-Encoder AI inference on {len(pairs)} candidates...")
    ai_scores_raw = model.predict(pairs, batch_size=32)
    
    # Normalize AI scores (typically -10 to +10 for MS-MARCO)
    min_ai = min(ai_scores_raw)
    max_ai = max(ai_scores_raw)
    range_ai = max_ai - min_ai if max_ai > min_ai else 1.0
    
    final_scored = []
    
    for i, (cand, tfidf) in enumerate(scored_candidates):
        s_ai = (float(ai_scores_raw[i]) - float(min_ai)) / float(range_ai)
        
        # Heuristics
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        skills = cand.get("skills", [])
        yoe = profile.get("years_of_experience", 0)
        
        s_tfidf = tfidf / max_tfidf
        s_exp = score_experience(yoe)
        s_skills = score_skills(skills)
        s_behav = score_behavioral(signals)
        s_loc = score_location(profile, signals)
        
        final_score = (
            s_ai * WEIGHT_AI +
            s_tfidf * WEIGHT_TFIDF +
            s_exp * WEIGHT_EXPERIENCE +
            s_skills * WEIGHT_SKILLS +
            s_behav * WEIGHT_BEHAVIORAL +
            s_loc * WEIGHT_LOCATION
        )
        
        final_scored.append((cand, final_score))
        
    return final_scored

