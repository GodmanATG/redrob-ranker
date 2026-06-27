import torch
from sentence_transformers import CrossEncoder
from config import JD_TEXT, WEIGHT_AI, WEIGHT_TFIDF, WEIGHT_EXPERIENCE, WEIGHT_SKILLS, WEIGHT_BEHAVIORAL, WEIGHT_LOCATION
from stages.scorer import score_experience, score_skills, score_behavioral, score_location

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading compact local Cross-Encoder ({MODEL_NAME})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CrossEncoder(MODEL_NAME, device=device)
    return _model

def format_candidate_for_ai(candidate):
    """
    Format the candidate's profile into a dense string for the CrossEncoder.
    We prioritize title, YOE, companies, and skills.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    summary = profile.get("summary", "")
    
    # Companies & Jobs
    jobs_text = " | ".join([f"{j.get('title', '')} at {j.get('company', '')}: {j.get('description', '')[:150]}" for j in history[:3]])
    
    # High-proficiency skills
    top_skills = [s.get('name', '') for s in skills if s.get('proficiency') in ['advanced', 'expert']]
    skills_text = ", ".join(top_skills)
    
    text = f"Candidate Profile:\nTitle: {title}\nYears of Experience: {yoe}\nSummary: {summary}\nKey Jobs: {jobs_text}\nTop Skills: {skills_text}"
    return text

def run_ai_reranker(scored_candidates):
    """
    Takes the top K candidates from the TF-IDF stage, formats them,
    and runs a real local Cross-Encoder AI over them to get a deep semantic score.
    Returns: list of (candidate, final_blended_score)
    """
    if not scored_candidates:
        return []
        
    model = get_model()
    
    # Build pairs of (JD, Candidate Profile)
    # Pass the full JD text so the AI naturally learns the disqualifiers (e.g. "no consulting", "no researchers")
    pairs = []
    max_tfidf = max(s for _, s in scored_candidates) if scored_candidates else 1.0
    if max_tfidf == 0: max_tfidf = 1.0
    
    for cand, tfidf in scored_candidates:
        cand_text = format_candidate_for_ai(cand)
        pairs.append((JD_TEXT, cand_text))
        
    print("Running AI inference locally on top candidates...")
    # Batch predict
    ai_scores_raw = model.predict(pairs, batch_size=32)
    
    # Normalize AI scores (they can be negative or > 10 depending on the model, MS-MARCO typically -10 to +10)
    min_ai = min(ai_scores_raw)
    max_ai = max(ai_scores_raw)
    range_ai = max_ai - min_ai if max_ai > min_ai else 1.0
    
    final_scored = []
    
    for i, (cand, tfidf) in enumerate(scored_candidates):
        # Normalize AI score to 0-1
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
