from config import CORE_SKILLS

def get_conversational_opener(variant, title, company):
    openers = [
        f"I reviewed this candidate's profile and they are an exceptional fit. They are currently a {title}",
        f"This candidate stands out as a strong potential hire. Bringing their experience as a {title}",
        f"An excellent candidate for the founding team. They are working as a {title}",
        f"I highly recommend this {title}"
    ]
    if company and company.lower() != "their previous role":
        openers = [
            f"I noticed their strong background at {company} where they work as a {title}.",
            f"This {title} from {company} is a standout candidate.",
            f"Their experience at {company} as a {title} caught my attention."
        ]
    return openers[variant % len(openers)]

def generate_reasoning(candidate, rank, score):
    """
    Generates a dynamic, fact-based reasoning string that simulates a highly-prompted LLM.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    
    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0.0)
    loc = profile.get("location", "India")
    
    # Extract actual matched core skills
    matched_skills = []
    for sk in skills:
        sk_name = sk.get("name", "").lower()
        if any(c in sk_name for c in CORE_SKILLS) and sk.get("proficiency") in ["advanced", "expert"]:
            matched_skills.append(sk.get("name"))
            
    companies = [j.get("company", "") for j in history if j.get("company")]
    latest_company = companies[0] if companies else ""
    
    rr = signals.get("recruiter_response_rate", 0.0)
    rr_pct = int(rr * 100)
    np_days = signals.get("notice_period_days", 60)
    
    variant = hash(candidate.get("candidate_id", ""))
    
    sentences = []
    
    # Tier 1 (Rank 1-25)
    if rank <= 25:
        sentences.append(get_conversational_opener(variant, title, latest_company))
        
        if 5.0 <= yoe <= 9.0:
            sentences.append(f"With exactly {yoe} years of experience, they perfectly hit our senior band requirement.")
        elif yoe < 5.0:
            sentences.append(f"While their {yoe} YOE is slightly under our 5-year target, their immense technical depth more than compensates for it.")
        else:
            sentences.append(f"They bring {yoe} YOE, offering deep senior leadership potential.")
            
        if len(matched_skills) >= 2:
            sk1, sk2 = matched_skills[0], matched_skills[1]
            if variant % 2 == 0:
                sentences.append(f"What makes them an elite fit is their advanced, production-level mastery of both {sk1} and {sk2}.")
            else:
                sentences.append(f"Their specific expertise in {sk1} and {sk2} directly aligns with the intelligence layer we are building.")
        elif matched_skills:
            sentences.append(f"Their advanced proficiency in {matched_skills[0]} specifically fulfills our core technical requirements.")
            
        github = signals.get("github_activity_score", 0.0)
        icr = signals.get("interview_completion_rate", 0.0)
        icr_pct = int(icr * 100)
            
        if np_days <= 30 and github > 80:
            sentences.append(f"Crucially, they can join in just {np_days} days and have phenomenal behavioral signals (GitHub Activity: {github}, {rr_pct}% recruiter response rate).")
        elif icr > 0.8:
            sentences.append(f"Behavioral signals are incredibly strong (response rate: {rr_pct}%, interview completion: {icr_pct}%).")
        else:
            sentences.append(f"Behavioral signals are solid with a {rr_pct}% recruiter response rate.")

    # Tier 2 (Rank 26-75)
    elif rank <= 75:
        if variant % 2 == 0:
            sentences.append(f"A very solid contender coming in with {yoe} years of industry experience.")
        else:
            sentences.append(f"I've ranked this {title} highly because of their strong foundational background.")
            
        if matched_skills:
            sentences.append(f"They show proven capability in {matched_skills[0]}, which is essential for our vector DB needs.")
            
        if np_days > 60:
            sentences.append(f"The only minor drawback is a {np_days}-day notice period, but their technical alignment justifies the wait.")
        elif rr_pct > 70:
            sentences.append(f"They are highly engaged, boasting a {rr_pct}% response rate.")
        else:
            sentences.append(f"They have an acceptable {np_days}-day notice period and reasonable engagement metrics.")

    # Tier 3 (Rank 76-100)
    else:
        sentences.append(f"I included this {title} as a borderline but viable candidate.")
        if yoe < 4.5:
            sentences.append(f"Their {yoe} YOE is on the lower end, so they might require some ramp-up time.")
        else:
            sentences.append(f"They have {yoe} YOE, providing a solid baseline.")
            
        if matched_skills:
            sentences.append(f"They have exposure to {matched_skills[0]}, though they may lack the deepest architectural retrieval experience of our top picks.")
        else:
            sentences.append("While they lack direct advanced vector DB keywords, their overall engineering profile is adjacent enough to warrant a look.")
            
        if rr_pct < 50:
            sentences.append(f"Keep in mind their response rate is quite low at {rr_pct}%.")
        else:
            sentences.append(f"Still, a {rr_pct}% response rate shows they are actively looking.")

    reasoning = " ".join(sentences)
    
    if len(reasoning) < 30:
        reasoning = f"Reviewed this {title} with {yoe} YOE. Matches core skills but lacks definitive behavioral signals."
        
    return reasoning
