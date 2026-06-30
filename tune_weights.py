import logging
import itertools
from typing import List, Tuple, Dict
from stages.scorer import score_experience, score_skills, score_behavioral

# Set up simple logging for the tuning script
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tune")

def evaluate_weights(candidates_with_ai_scores: List[Tuple[Dict, float, float]], weights: Tuple[float, float, float, float, float]) -> float:
    """
    Evaluates a specific set of weights and returns a metric (e.g., number of high-quality candidates in top 10).
    For demonstration, we score all candidates with the given weights.
    """
    w_ai, w_tfidf, w_exp, w_skills, w_behav = weights
    scored = []
    
    for cand, ai_score, tfidf_score in candidates_with_ai_scores:
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        s_exp = score_experience(profile.get("years_of_experience", 0))
        s_skills = score_skills(profile.get("skills", []))
        s_behav = score_behavioral(signals)
        
        final_score = (
            w_ai * ai_score +
            w_tfidf * tfidf_score +
            w_exp * s_exp +
            w_skills * s_skills +
            w_behav * s_behav
        )
        scored.append((cand, final_score))
        
    # Sort descending
    scored.sort(key=lambda x: -x[1])
    
    # Simple metric: sum of AI scores of the top 10 candidates (we want the top 10 to have high semantic relevance)
    metric = sum(ai_score for cand, ai_score, _ in candidates_with_ai_scores if cand in [c[0] for c in scored[:10]])
    return metric

def main():
    logger.info("Starting Hyperparameter Grid Search...")
    
    # We would normally load real candidates and their pre-computed AI and TF-IDF scores here.
    # For this script, we mock some pre-computed data to demonstrate the grid search algorithm.
    mock_data = [
        ({"profile": {"years_of_experience": 7.0, "skills": []}, "redrob_signals": {"github_activity_score": 90}}, 0.9, 0.8),
        ({"profile": {"years_of_experience": 2.0, "skills": []}, "redrob_signals": {"github_activity_score": 10}}, 0.4, 0.3),
        ({"profile": {"years_of_experience": 8.0, "skills": []}, "redrob_signals": {"github_activity_score": 85}}, 0.95, 0.75),
        ({"profile": {"years_of_experience": 5.5, "skills": []}, "redrob_signals": {"github_activity_score": 70}}, 0.8, 0.85),
        ({"profile": {"years_of_experience": 15.0, "skills": []}, "redrob_signals": {"github_activity_score": 50}}, 0.5, 0.6),
    ]
    
    # Grid of potential weights to test
    ai_weights = [0.3, 0.4, 0.5]
    tfidf_weights = [0.1, 0.15, 0.2]
    exp_weights = [0.1, 0.2, 0.3]
    skills_weights = [0.1, 0.15, 0.2]
    behav_weights = [0.05, 0.1, 0.15]
    
    best_weights = None
    best_metric = -1
    
    combinations = list(itertools.product(ai_weights, tfidf_weights, exp_weights, skills_weights, behav_weights))
    valid_combinations = [w for w in combinations if abs(sum(w) - 1.0) < 0.01]
    
    logger.info(f"Testing {len(valid_combinations)} valid weight combinations (sum = 1.0)...")
    
    for w in valid_combinations:
        metric = evaluate_weights(mock_data, w)
        if metric > best_metric:
            best_metric = metric
            best_weights = w
            
    logger.info("\n=== Optimization Complete ===")
    logger.info(f"Best Metric Score: {best_metric:.4f}")
    logger.info("Optimal Weights Found:")
    logger.info(f"  WEIGHT_AI         = {best_weights[0]:.2f}")
    logger.info(f"  WEIGHT_TFIDF      = {best_weights[1]:.2f}")
    logger.info(f"  WEIGHT_EXPERIENCE = {best_weights[2]:.2f}")
    logger.info(f"  WEIGHT_SKILLS     = {best_weights[3]:.2f}")
    logger.info(f"  WEIGHT_BEHAVIORAL = {best_weights[4]:.2f}")
    logger.info("\nUpdate config.py with these values for maximum precision.")

if __name__ == "__main__":
    main()
