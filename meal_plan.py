"""Rule-based meal recommendations for profile (full day) and food log (remaining intake)."""

from __future__ import annotations

from meal_preferences import (
    activity_phrase,
    enrich_profile_blocks,
    normalize_meal_preferences,
    preferences_summary,
)

MEAL_DISCLAIMER = (
    "General wellness suggestions only — not medical advice. "
    "Adjust portions for health conditions with a qualified professional."
)

_PROFILE_SPLITS = {
    "sedentary": (0.26, 0.34, 0.32, 0.08),
    "light": (0.26, 0.34, 0.31, 0.09),
    "moderate": (0.25, 0.33, 0.32, 0.10),
    "active": (0.24, 0.32, 0.32, 0.12),
    "very_active": (0.23, 0.31, 0.31, 0.15),
}

_DEFAULT_SPLIT = _PROFILE_SPLITS["moderate"]

_MACRO_PROTEIN = 0.30
_MACRO_CARBS = 0.45
_MACRO_FAT = 0.25
_KCAL_PER_G_P = 4
_KCAL_PER_G_C = 4
_KCAL_PER_G_F = 9


def _split_daily_kcal(total: int, activity_level: str) -> tuple[int, int, int, int]:
    b, l, d, s = _PROFILE_SPLITS.get(activity_level, _DEFAULT_SPLIT)
    breakfast = int(round(total * b))
    lunch = int(round(total * l))
    dinner = int(round(total * d))
    snack = int(round(total * s))
    used = breakfast + lunch + dinner + snack
    remainder = total - used
    if remainder:
        dinner += remainder
    if snack <= 0 and remainder > 50:
        snack = min(remainder, int(total * 0.08))
        dinner -= snack
    return breakfast, lunch, dinner, max(0, snack)


def build_profile_day_meal_plan(
    daily_goal_kcal: float,
    *,
    gender: str = "male",
    age: int = 30,
    height_cm: float = 170.0,
    weight_kg: float = 70.0,
    activity_level: str = "moderate",
    preferences: dict | None = None,
) -> dict:
    """Full-day sample meals from profile stats (does not use food log)."""
    prefs = normalize_meal_preferences(preferences)
    goal = max(800, min(6000, int(round(float(daily_goal_kcal)))))
    bmi = weight_kg / ((height_cm / 100.0) ** 2) if height_cm > 0 else 22.0

    bk, ln, dn, sn = _split_daily_kcal(goal, activity_level)
    blocks = [
        {
            "key": "breakfast",
            "label": "Breakfast",
            "kcal": bk,
            "description": f"Aim for about {bk} kcal to start the day steadily.",
        },
        {
            "key": "lunch",
            "label": "Lunch",
            "kcal": ln,
            "description": f"Target roughly {ln} kcal with balanced protein and carbs.",
        },
        {
            "key": "dinner",
            "label": "Dinner",
            "kcal": dn,
            "description": f"Plan around {dn} kcal; include vegetables and lean protein.",
        },
    ]
    if sn > 0:
        blocks.append(
            {
                "key": "snack",
                "label": "Snack",
                "kcal": sn,
                "description": f"Optional snack near {sn} kcal if hungry between meals.",
            }
        )

    blocks = enrich_profile_blocks(
        blocks,
        prefs,
        bmi=bmi,
        age=age,
        activity_level=activity_level,
    )

    intro = (
        f"Sample day for a {age}-year-old, {activity_phrase(activity_level)} profile "
        f"at about {goal} kcal/day (from your estimate, not from food log)."
    )
    if gender == "female" and age < 51:
        intro += " Include iron-rich plant or lean animal sources as fits your diet."

    return {
        "source": "rules",
        "intro": intro,
        "disclaimer": MEAL_DISCLAIMER
        + " For what you already ate today, use Meal suggestions on a Food log day page.",
        "preferences_summary": preferences_summary(prefs),
        "total_kcal": goal,
        "meals": blocks,
    }


def _macro_summary_text(gap_p: float, gap_c: float, gap_f: float) -> str:
    """One-line macros insight for the food-log meal card."""
    parts: list[str] = []
    if gap_p > 8:
        parts.append(
            "protein is still below a typical split (consider lean protein or legumes next)"
        )
    elif gap_p < -15:
        parts.append("protein is already above a usual share for today")
    else:
        parts.append("protein is relatively on track")

    if gap_c > 12:
        parts.append("carbohydrate has room to catch up (whole grains, fruit, or starchy veg)")
    elif gap_c < -20:
        parts.append("carbohydrate is already fairly high for this target")
    else:
        parts.append("carbohydrate is relatively on track")

    if gap_f > 6:
        parts.append("fat still has headroom (nuts, olive oil, or avocado in moderation)")
    elif gap_f < -8:
        parts.append("fat is already on the higher side for today")
    else:
        parts.append("fat is close to a usual balance")

    return (
        "For this calorie target, "
        + parts[0]
        + " while "
        + parts[1]
        + " and "
        + parts[2]
        + "."
    )


def _status_meta(remaining: float) -> tuple[str, str, str]:
    if remaining <= 0:
        return "at_or_over_goal", "At or over target", "danger"
    if remaining < 150:
        return "almost_done", "Almost at target", "warning"
    return "room_remaining", "Below target", "success"


def _food_log_suggestions(
    remaining: float,
    gap_p: float,
    gap_c: float,
    gap_f: float,
    prefs: dict,
) -> list[dict]:
    """Up to three numbered meal ideas with kcal estimates."""
    if remaining <= 0:
        return [
            {
                "title": "Wrap up for today",
                "kcal_estimate": None,
                "detail": (
                    "You are at or above your calorie target. If still hungry, "
                    "choose water-rich vegetables, broth-based soup, or a very small portion."
                ),
            }
        ]

    if remaining < 150:
        kcal = int(round(remaining))
        detail = enrich_profile_blocks(
            [
                {
                    "key": "snack",
                    "label": "Snack",
                    "kcal": kcal,
                    "description": f"A light top-up around {kcal} kcal should close the gap.",
                }
            ],
            prefs,
        )[0]["description"]
        return [
            {
                "title": "Small top-up",
                "kcal_estimate": kcal,
                "detail": detail,
            }
        ]

    ideas: list[dict] = []
    budget = float(remaining)

    if gap_p > 8:
        k1 = int(min(max(200, budget * 0.42), budget))
        detail = enrich_profile_blocks(
            [
                {
                    "key": "lunch",
                    "label": "Protein-forward meal",
                    "kcal": k1,
                    "description": (
                        f"Build your next plate around lean protein and vegetables (~{k1} kcal)."
                    ),
                }
            ],
            prefs,
        )[0]["description"]
        ideas.append(
            {
                "title": "Protein-forward meal",
                "kcal_estimate": k1,
                "detail": detail,
            }
        )
        budget -= k1

    if budget >= 180:
        k2 = int(min(max(180, budget * 0.55), budget))
        meal_key = "dinner" if gap_c > 10 else "lunch"
        detail = enrich_profile_blocks(
            [
                {
                    "key": meal_key,
                    "label": "Balanced plate",
                    "kcal": k2,
                    "description": (
                        f"Balanced carbs, protein, and produce for roughly {k2} kcal."
                    ),
                }
            ],
            prefs,
        )[0]["description"]
        ideas.append(
            {
                "title": "Balanced plate",
                "kcal_estimate": k2,
                "detail": detail,
            }
        )
        budget -= k2

    if budget >= 120:
        k3 = int(min(350, max(120, budget)))
        detail = enrich_profile_blocks(
            [
                {
                    "key": "snack",
                    "label": "Snack",
                    "kcal": k3,
                    "description": "A modest snack if you are hungry later in the day.",
                }
            ],
            prefs,
        )[0]["description"]
        ideas.append(
            {
                "title": "Light finish if you snack later",
                "kcal_estimate": k3,
                "detail": detail,
            }
        )

    if not ideas:
        kcal = int(round(remaining * 0.4))
        detail = enrich_profile_blocks(
            [
                {
                    "key": "lunch",
                    "label": "Next meal",
                    "kcal": kcal,
                    "description": f"Plan about {kcal} kcal for your next eating occasion.",
                }
            ],
            prefs,
        )[0]["description"]
        ideas.append(
            {
                "title": "Next meal idea",
                "kcal_estimate": kcal,
                "detail": detail,
            }
        )

    return ideas[:3]


def build_meal_recommendations(
    daily_goal_kcal: float,
    consumed_calories: float,
    consumed_protein: float,
    consumed_carbs: float,
    consumed_fat: float,
    *,
    preferences: dict | None = None,
) -> dict:
    """Suggest next eating based on goal vs logged intake for one day."""
    prefs = normalize_meal_preferences(preferences)
    goal = max(800, min(6000, float(daily_goal_kcal)))
    eaten_c = max(0.0, float(consumed_calories))
    eaten_p = max(0.0, float(consumed_protein))
    eaten_carbs = max(0.0, float(consumed_carbs))
    eaten_f = max(0.0, float(consumed_fat))

    remaining = goal - eaten_c
    target_p = goal * _MACRO_PROTEIN / _KCAL_PER_G_P
    target_carbs = goal * _MACRO_CARBS / _KCAL_PER_G_C
    target_f = goal * _MACRO_FAT / _KCAL_PER_G_F

    gap_p = target_p - eaten_p
    gap_c = target_carbs - eaten_carbs
    gap_f = target_f - eaten_f

    status, status_label, status_tone = _status_meta(remaining)
    suggestions = _food_log_suggestions(remaining, gap_p, gap_c, gap_f, prefs)

    return {
        "status": status,
        "status_label": status_label,
        "status_tone": status_tone,
        "daily_goal_kcal": round(goal, 1),
        "consumed_calories": round(eaten_c, 1),
        "remaining_calories": round(remaining, 1),
        "macro_gaps": {
            "protein": round(gap_p, 1),
            "carbs": round(gap_c, 1),
            "fat": round(gap_f, 1),
        },
        "macro_summary": _macro_summary_text(gap_p, gap_c, gap_f),
        "preferences_summary": preferences_summary(prefs),
        "suggestions": suggestions,
        "disclaimer": MEAL_DISCLAIMER,
    }
