"""One-shot Gemini call for customized breakfast / lunch / dinner."""

from __future__ import annotations

import json
import re
from typing import Any

from google.generativeai.types import GenerationConfig

from gemini_call import generate_content_with_timeout
from meal_plan import build_profile_day_meal_plan
from meal_preferences import normalize_meal_preferences, preferences_summary


def _parse_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _gemini_text(response) -> str:
    try:
        return (response.text or "").strip()
    except Exception:
        parts = []
        for c in getattr(response, "candidates", []) or []:
            content = getattr(c, "content", None)
            for p in getattr(content, "parts", []) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()


def _plan_from_gemini_json(data: dict, daily_kcal: int, prefs: dict) -> dict | None:
    meals_out = []
    labels = (
        ("breakfast", "Breakfast"),
        ("lunch", "Lunch"),
        ("dinner", "Dinner"),
    )
    splits = (0.26, 0.34, 0.32)
    for i, (key, label) in enumerate(labels):
        block = data.get(key)
        if not isinstance(block, dict):
            return None
        desc = (block.get("description") or block.get("meal") or "").strip()
        if not desc:
            return None
        kcal = block.get("kcal")
        if kcal is None:
            kcal = int(round(daily_kcal * splits[i]))
        meals_out.append(
            {"key": key, "label": label, "kcal": int(kcal), "description": desc}
        )
    return {
        "source": "gemini",
        "intro": (
            f"AI-customized day near {daily_kcal} kcal based on your profile and preferences."
        ),
        "disclaimer": (
            "AI-generated suggestions — verify portions and ingredients. "
            "Not medical advice. For intake vs what you already logged today, use Food log day suggestions."
        ),
        "preferences_summary": preferences_summary(prefs),
        "total_kcal": daily_kcal,
        "meals": meals_out,
    }


async def generate_profile_meal_plan_gemini(
    model,
    *,
    daily_kcal: int,
    gender: str,
    age: int,
    height_cm: float,
    weight_kg: float,
    activity_level: str,
    preferences: dict | None,
) -> dict:
    """Call Gemini once; on failure return rule-based plan with ai_note."""
    prefs = normalize_meal_preferences(preferences)
    daily_kcal = max(800, min(6000, int(daily_kcal)))
    summary = preferences_summary(prefs)

    prompt = f"""Create one day of eating for this user. Reply with JSON only (no markdown).

Daily calorie target: {daily_kcal} kcal
Gender: {gender}
Age: {age}
Height cm: {height_cm}
Weight kg: {weight_kg}
Activity: {activity_level}
Preferences: {summary}

Schema:
{{
  "breakfast": {{"description": "specific foods and portions", "kcal": number}},
  "lunch": {{"description": "...", "kcal": number}},
  "dinner": {{"description": "...", "kcal": number}}
}}

Rules:
- Respect diet_style (omnivore / vegetarian / vegan), body_goal, spicy preference, favorites, and restrictions.
- Each description must name concrete foods (not generic advice only).
- kcal values should sum close to {daily_kcal} (within about 10%).
- English only in JSON values."""

    try:
        response = await generate_content_with_timeout(
            model,
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )
        raw = _gemini_text(response)
        parsed = _parse_json_object(raw)
        if parsed:
            plan = _plan_from_gemini_json(parsed, daily_kcal, prefs)
            if plan:
                return plan
        response2 = await generate_content_with_timeout(model, prompt)
        parsed2 = _parse_json_object(_gemini_text(response2))
        if parsed2:
            plan2 = _plan_from_gemini_json(parsed2, daily_kcal, prefs)
            if plan2:
                return plan2
    except Exception as exc:
        fallback = build_profile_day_meal_plan(
            daily_kcal,
            gender=gender,
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            activity_level=activity_level,
            preferences=prefs,
        )
        fallback["source"] = "rules"
        fallback["ai_note"] = f"AI unavailable ({exc}). Showing rule-based plan."
        return fallback

    fallback = build_profile_day_meal_plan(
        daily_kcal,
        gender=gender,
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        activity_level=activity_level,
        preferences=prefs,
    )
    fallback["ai_note"] = "AI response could not be parsed. Showing rule-based plan."
    return fallback
