import datetime
from typing import List, Dict, Union
from config import EXP_MIN_TARGET, EXP_MAX_TARGET, CORE_SKILLS

def score_experience(yoe: Union[int, float]) -> float:
    """
    Scores years of experience against target band (5-9 years).
    """
    if yoe < 3: return 0.0
    if yoe > 15: return 0.3
    if EXP_MIN_TARGET <= yoe <= EXP_MAX_TARGET: return 1.0
    
    if yoe < EXP_MIN_TARGET:
        return max(0.0, 1.0 - 0.3 * (EXP_MIN_TARGET - yoe))
    else:
        return max(0.0, 1.0 - 0.1 * (yoe - EXP_MAX_TARGET))

def score_skills(skills: List[Dict]) -> float:
    """
    Scores skills based on presence of core JD skills and proficiency/duration.
    """
    score = 0.0
    skill_names = [s.get("name", "").lower() for s in skills]
    
    for core_skill in CORE_SKILLS:
        for sk in skills:
            if core_skill in sk.get("name", "").lower():
                prof = sk.get("proficiency", "beginner")
                duration = sk.get("duration_months", 0)
                
                multiplier = 1.0
                if prof == "advanced": multiplier = 1.5
                if prof == "expert": multiplier = 2.0
                
                score += (multiplier * min(duration, 60)) / 60.0
                break # count each core skill once
                
    # Normalize (arbitrary cap at 5 strong core skills = 1.0)
    return min(1.0, score / 5.0)

def parse_date(date_str: str) -> datetime.date:
    if not date_str:
        return datetime.date.today() - datetime.timedelta(days=365)
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return datetime.date.today() - datetime.timedelta(days=365)

def score_behavioral(signals: Dict) -> float:
    """
    Scores based on platform engagement, recency, and responsiveness.
    """
    if not signals:
        return 0.5
        
    # Last Active Date check (critical drop-off if > 6 months)
    last_active = signals.get("last_active_date", "")
    last_active_d = parse_date(last_active)
    days_since_active = (datetime.date.today() - last_active_d).days
    
    if days_since_active > 180:
        return 0.0 # Ghost candidate, penalize entirely
        
    score = 0.5 # base score
    
    # Recruiter response rate (huge signal)
    rr = signals.get("recruiter_response_rate", 0.5)
    score += (rr - 0.5) * 0.4
    
    # Interview Completion Rate
    icr = signals.get("interview_completion_rate", 0.5)
    score += (icr - 0.5) * 0.2
    
    # Offer Acceptance Rate
    oar = signals.get("offer_acceptance_rate", 0.5)
    if oar != -1: # -1 means no prior offers
        score += (oar - 0.5) * 0.1
        
    # GitHub Activity
    github = signals.get("github_activity_score", 0.0)
    if github > 80:
        score += 0.15
    elif github > 50:
        score += 0.05
    
    # Open to work
    if signals.get("open_to_work_flag", False):
        score += 0.1
        
    # Notice period (JD prefers sub 30 days)
    np_days = signals.get("notice_period_days", 60)
    if np_days <= 30: score += 0.1
    elif np_days > 90: score -= 0.2
    
    return max(0.0, min(1.0, score))
    
def score_location(profile: Dict, signals: Dict) -> float:
    """
    Scores based on JD location preference (Pune/Noida, Tier 1, or willing to relocate).
    """
    loc = profile.get("location", "").lower()
    
    target_cities = ["pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore"]
    
    if any(city in loc for city in target_cities):
        return 1.0
        
    if signals and signals.get("willing_to_relocate", False):
        return 0.8
        
    return 0.3


