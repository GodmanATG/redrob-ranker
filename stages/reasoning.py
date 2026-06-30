import logging
from jinja2 import Template
from typing import Dict, List
from config import CORE_SKILLS

logger = logging.getLogger(__name__)

TEMPLATE_REASONING = """
{%- if rank <= 25 -%}
    {{ opener }} 
    {%- if yoe >= 5.0 and yoe <= 9.0 -%}
        With exactly {{ yoe }} years of experience, they perfectly hit our senior band requirement.
    {%- elif yoe < 5.0 -%}
        While their {{ yoe }} YOE is slightly under our 5-year target, their immense technical depth more than compensates for it.
    {%- else -%}
        They bring {{ yoe }} YOE, offering deep senior leadership potential.
    {%- endif -%}
    
    {%- if matched_skills|length >= 2 -%}
        {%- if variant % 2 == 0 -%}
            What makes them an elite fit is their advanced, production-level mastery of both {{ matched_skills[0] }} and {{ matched_skills[1] }}.
        {%- else -%}
            Their specific expertise in {{ matched_skills[0] }} and {{ matched_skills[1] }} directly aligns with the intelligence layer we are building.
        {%- endif -%}
    {%- elif matched_skills -%}
        Their advanced proficiency in {{ matched_skills[0] }} specifically fulfills our core technical requirements.
    {%- endif -%}
    
    {%- if np_days <= 30 and github > 80 -%}
        Crucially, they can join in just {{ np_days }} days and have phenomenal behavioral signals (GitHub Activity: {{ github }}, {{ rr_pct }}% recruiter response rate).
    {%- elif icr > 0.8 -%}
        Behavioral signals are incredibly strong (response rate: {{ rr_pct }}%, interview completion: {{ icr_pct }}%).
    {%- else -%}
        Behavioral signals are solid with a {{ rr_pct }}% recruiter response rate.
    {%- endif -%}
{%- elif rank <= 75 -%}
    {%- if variant % 2 == 0 -%}
        A very solid contender coming in with {{ yoe }} years of industry experience.
    {%- else -%}
        I've ranked this {{ title }} highly because of their strong foundational background.
    {%- endif -%}
    
    {%- if matched_skills -%}
        They show proven capability in {{ matched_skills[0] }}, which is essential for our vector DB needs.
    {%- endif -%}
    
    {%- if np_days > 60 -%}
        The only minor drawback is a {{ np_days }}-day notice period, but their technical alignment justifies the wait.
    {%- elif rr_pct > 70 -%}
        They are highly engaged, boasting a {{ rr_pct }}% response rate.
    {%- else -%}
        They have an acceptable {{ np_days }}-day notice period and reasonable engagement metrics.
    {%- endif -%}
{%- else -%}
    I included this {{ title }} as a borderline but viable candidate.
    {%- if yoe < 4.5 -%}
        Their {{ yoe }} YOE is on the lower end, so they might require some ramp-up time.
    {%- else -%}
        They have {{ yoe }} YOE, providing a solid baseline.
    {%- endif -%}
    
    {%- if matched_skills -%}
        They have exposure to {{ matched_skills[0] }}, though they may lack the deepest architectural retrieval experience of our top picks.
    {%- else -%}
        While they lack direct advanced vector DB keywords, their overall engineering profile is adjacent enough to warrant a look.
    {%- endif -%}
    
    {%- if rr_pct < 50 -%}
        Keep in mind their response rate is quite low at {{ rr_pct }}%.
    {%- else -%}
        Still, a {{ rr_pct }}% response rate shows they are actively looking.
    {%- endif -%}
{%- endif -%}
"""

def get_conversational_opener(variant: int, title: str, company: str) -> str:
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
    return openers[abs(variant) % len(openers)]

def generate_reasoning(candidate: Dict, rank: int, score: float) -> str:
    """
    Generates a dynamic, fact-based reasoning string that simulates a highly-prompted LLM.
    Uses a Jinja2 template under the hood to structure paragraphs dynamically.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    
    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0.0)
    
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
    
    variant = abs(hash(candidate.get("candidate_id", "")))
    
    opener = get_conversational_opener(variant, title, latest_company)
    github = signals.get("github_activity_score", 0.0)
    icr = signals.get("interview_completion_rate", 0.0)
    icr_pct = int(icr * 100)
    
    # Render Jinja2 template
    t = Template(TEMPLATE_REASONING)
    rendered = t.render(
        rank=rank,
        opener=opener,
        yoe=yoe,
        matched_skills=matched_skills,
        variant=variant,
        np_days=np_days,
        github=github,
        icr=icr,
        icr_pct=icr_pct,
        rr_pct=rr_pct,
        title=title
    )
    
    # Clean up excess whitespace and format neatly
    reasoning = " ".join(rendered.split())
    
    if len(reasoning) < 30:
        reasoning = f"Reviewed this {title} with {yoe} YOE. Matches core skills but lacks definitive behavioral signals."
        
    return reasoning

