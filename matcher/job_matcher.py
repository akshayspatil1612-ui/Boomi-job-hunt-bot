"""
job_matcher.py - Fixed: uses gemini-2.0-flash (current free model)
"""

import google.generativeai as genai
import json
import os
import sys
import time
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")
PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "resume_profile.json")

GEMINI_MODEL = "gemini-2.0-flash"  # Current free tier model

BOOMI_TITLE_KEYWORDS = [
    "boomi developer", "boomi integration", "integration developer",
    "boomi engineer", "boomi consultant", "api integration",
    "dell boomi", "integration architect", "boomi cloud",
    "edi developer", "data transformation", "boomi atomsphere",
    "boomi specialist", "integration engineer", "mulesoft", "talend"
]


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_profile():
    with open(PROFILE_PATH) as f:
        return json.load(f)


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    if "```" in raw:
        lines = raw.split("\n")
        cleaned = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(cleaned)
    return raw.strip()


def title_based_score(job: dict, profile: dict) -> dict:
    title = job.get("title", "").lower()
    score = 0
    reasons = []
    missing = []

    # Ensure experience_years is an integer
    exp = profile.get("experience_years", 0)
    if isinstance(exp, str):
        try:
            exp = profile.get("experience_years", 0)
            exp = int(exp) if isinstance(exp, str) else exp    # Convert to int
            
            if exp >= 4 and any(w in title for w in ["senior", "lead", "principal"]):
                score += 20
                reasons.append("Seniority match matches your experience")
        except (ValueError, TypeError):
            exp = 0

    if any(kw in title for kw in BOOMI_TITLE_KEYWORDS):
        score += 50
        reasons.append("Job title matches Boomi/Integration domain")

    if exp >= 3 and any(w in title for w in ["senior", "lead", "principal", "architect"]):
        score += 20
        reasons.append("Seniority level matches your experience")
    elif not any(w in title for w in ["senior", "lead", "junior", "principal"]):
        score += 15
        reasons.append("Mid-level position matches your profile")

    profile_skills = (
        profile.get("tech_skills", {}).get("boomi_specific", []) +
        profile.get("tech_skills", {}).get("programming_languages", []) +
        profile.get("tech_skills", {}).get("integration_tools", [])
    )
    for skill in profile_skills:
        if skill.lower() in title:
            score += 10
            reasons.append(f"{skill} mentioned in title")
            break

    skill_names = [s.lower() for s in profile_skills]
    if "mulesoft" not in skill_names: missing.append("MuleSoft")
    if "talend" not in skill_names: missing.append("Talend")
    if "api management" not in skill_names: missing.append("API Management")

    return {
        "match_score": min(score, 85),
        "match_reasons": reasons or ["Boomi/Integration role matching your profile"],
        "missing_skills": missing[:3],
        "nice_to_have_present": [],
        "recommendation": "APPLY" if score >= 50 else "MAYBE",
        "recommendation_reason": "Title-based match",
        "seniority_match": True,
        "remote_type": "not_specified",
        "scored_by": "title_fallback"
    }


def match_job_to_profile(job: dict, profile: dict, model) -> dict:
    profile_summary = {
        "experience_years": profile.get("experience_years", 0),
        "current_level": profile.get("current_level", "mid"),
        "job_titles": profile.get("job_titles", []),
        "boomi_specific": profile["tech_skills"].get("boomi_specific", []),
        "programming_languages": profile["tech_skills"].get("programming_languages", []),
        "integration_tools": profile["tech_skills"].get("integration_tools", []),
        "ci_cd": profile["tech_skills"].get("ci_cd", []),
        "methodologies": profile.get("methodologies", []),
    }

    description = job.get("description", "").strip()
    if len(description) < 100:
        print(f"      ℹ️  Short description — title fallback")
        job.update(title_based_score(job, profile))
        return job

    prompt = f"""You are a Boomi Integration recruiter. Score this candidate vs job.

CANDIDATE:
{json.dumps(profile_summary, indent=2)}

JOB:
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
Description: {description[:2000]}

Rules: Boomi/Integration role + Boomi/Integration candidate = minimum score 50.

Return ONLY JSON, no markdown:
{{
  "match_score": 75,
  "match_reasons": ["Has Boomi experience", "Java matches"],
  "missing_skills": ["MuleSoft", "Talend"],
  "nice_to_have_present": ["JIRA"],
  "recommendation": "APPLY",
  "recommendation_reason": "Strong core match",
  "seniority_match": true,
  "remote_type": "fully_remote"
}}"""

    try:
        response = model.generate_content(prompt)
        raw = clean_json_response(response.text)
        try:
            match_data = json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            match_data = json.loads(m.group()) if m else {}

        job.update({
            "match_score": int(match_data.get("match_score", 0)),
            "match_reasons": match_data.get("match_reasons", []),
            "missing_skills": match_data.get("missing_skills", []),
            "nice_to_have_present": match_data.get("nice_to_have_present", []),
            "recommendation": match_data.get("recommendation", "MAYBE"),
            "recommendation_reason": match_data.get("recommendation_reason", ""),
            "seniority_match": match_data.get("seniority_match", False),
            "remote_type": match_data.get("remote_type", "not_specified"),
            "scored_by": "gemini",
        })
        print(f"      ✅ Score: {job['match_score']}% ({job['recommendation']})")

    except Exception as e:
        print(f"      ⚠️  Gemini error: {str(e)[:80]} — title fallback")
        job.update(title_based_score(job, profile))

    return job


def batch_match_jobs(jobs: list, min_score: int = 0) -> list:
    config = load_config()
    profile = load_profile()
    api_key = config["api_keys"]["gemini_api_key"]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    matched = []
    print(f"🤖 Matching {len(jobs)} jobs with Gemini AI ({GEMINI_MODEL})...")

    for i, job in enumerate(jobs, 1):
        print(f"   [{i}/{len(jobs)}] {job.get('title','?')[:40]} @ {job.get('company','?')[:20]}...")
        matched_job = match_job_to_profile(job, profile, model)
        if matched_job.get("match_score", 0) >= min_score:
            matched.append(matched_job)
        if i % 10 == 0:
            print(f"   ⏳ Rate limit pause 3s...")
            time.sleep(3)

    matched.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    print(f"\n✅ Matched: {len(matched)} jobs (score >= {min_score}%)")
    return matched


def generate_skill_gap_analysis(all_jobs: list, profile: dict, api_key: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)  # Use same model, not pro

    all_missing = []
    for job in all_jobs[:20]:
        all_missing.extend(job.get("missing_skills", []))
    missing_counts = Counter(all_missing).most_common(15)
    if not missing_counts:
        missing_counts = [("MuleSoft", 5), ("Talend", 4), ("Kubernetes", 3), ("Docker", 3)]

    prompt = f"""Boomi Integration career coach. {profile.get('experience_years', 3)} years exp candidate.
Skills: {', '.join(profile['tech_skills'].get('boomi_specific', []) + profile['tech_skills'].get('programming_languages', []))}
Frequently missing: {json.dumps(missing_counts)}

Return ONLY JSON, no markdown:
{{
  "critical_skills_to_learn": [
    {{"skill": "MuleSoft", "reason": "Enterprise integration platform", "learning_time": "4-6 weeks", "resources": ["mulesoft.com/training"]}}
  ],
  "trending_in_integration": ["iPaaS", "Hybrid integrations", "API-first architecture"],
  "certifications_recommended": [
    {{"cert": "Dell Boomi Certified Advanced", "reason": "Specialist credentials", "url": "dell.com/boomi"}}
  ],
  "quick_wins": ["Learn Talend - 2 weeks", "Master API Gateway - 1 week"],
  "career_advice": "Personalized advice."
}}"""

    try:
        raw = clean_json_response(model.generate_content(prompt).text)
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️ Skill gap error: {e}")
        return {
            "critical_skills_to_learn": [
                {"skill": "MuleSoft", "reason": "Enterprise iPaaS platform, high demand", "learning_time": "4-6 weeks", "resources": ["mulesoft.com/training"]},
                {"skill": "Talend", "reason": "Data integration and ETL tool", "learning_time": "3-4 weeks", "resources": ["talend.com/learn"]},
                {"skill": "Kubernetes", "reason": "Container orchestration for cloud integrations", "learning_time": "3-4 weeks", "resources": ["kubernetes.io/docs"]},
            ],
            "trending_in_integration": ["iPaaS (Integration Platform as a Service)", "Hybrid integration", "API-first architecture", "Composable applications"],
            "certifications_recommended": [{"cert": "Dell Boomi Certified Advanced Developer", "reason": "Valued for specialist roles", "url": "dell.com/boomi"}],
            "quick_wins": ["Master Boomi EDI module", "Learn API Gateway patterns"],
            "career_advice": f"With {profile.get('experience_years',3)} years Boomi experience, focus on MuleSoft or Talend to expand marketability in enterprise integration."
        }
