import datetime

def parse_date(date_str):
    if not date_str:
        return datetime.date.today()
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return datetime.date.today()

def is_honeypot_or_trap(candidate):
    """
    Checks if a candidate is a honeypot (logically impossible profile).
    Returns (True, "reason") if it's a trap, (False, "") otherwise.
    """
    
    # 1. Check for impossible skill duration (expert with 0 duration)
    skills = candidate.get("skills", [])
    for skill in skills:
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0:
            return True, "Honeypot: Expert skill with 0 months duration"
            
    # 2. Career history checks (YOE verification)
    career_history = candidate.get("career_history", [])
    total_yoe_from_jobs = 0
    
    for job in career_history:
        duration = job.get("duration_months", 0)
        total_yoe_from_jobs += (duration / 12.0)
        
    # Check if YOE stated heavily contradicts YOE from jobs
    profile = candidate.get("profile", {})
    stated_yoe = profile.get("years_of_experience", 0)
    if stated_yoe > 0 and total_yoe_from_jobs > stated_yoe * 1.5:
        return True, "Honeypot: Stated YOE drastically lower than career history"

    return False, ""

def filter_candidates(scored_candidates):
    """
    Filters out honeypots.
    scored_candidates: list of (candidate_dict, semantic_score)
    Returns: filtered list
    """
    valid_candidates = []
    for cand, score in scored_candidates:
        is_trap, reason = is_honeypot_or_trap(cand)
        if not is_trap:
            valid_candidates.append((cand, score))
            
    return valid_candidates
