"""Normalize diet preferences and enrich rule-based meal descriptions."""

from __future__ import annotations

import re
from typing import Any

VALID_DIET = frozenset({"omnivore", "vegetarian", "vegan"})
VALID_GOAL = frozenset({"lose", "gain", "maintain"})
VALID_SPICY = frozenset({"yes", "no", "either"})

_DISH_POOLS: dict[str, dict[str, list[str]]] = {
    "omnivore": {
        "breakfast": [
            "Greek yogurt with berries and honey",
            "Scrambled eggs on whole-grain toast",
            "Oatmeal with banana and peanut butter",
            "Smoked salmon bagel with cream cheese",
        ],
        "lunch": [
            "Grilled chicken quinoa bowl with vegetables",
            "Turkey and avocado whole-wheat wrap",
            "Brown rice with stir-fried beef and broccoli",
            "Tuna salad sandwich on multigrain bread",
        ],
        "dinner": [
            "Baked salmon with sweet potato and greens",
            "Lean pork tenderloin with roasted vegetables",
            "Chicken curry with basmati rice (moderate oil)",
            "Beef and vegetable stew with barley",
        ],
        "snack": [
            "Apple with almond butter",
            "Cottage cheese and fruit",
            "Hard-boiled eggs",
            "Trail mix (nuts and dried fruit)",
        ],
    },
    "vegetarian": {
        "breakfast": [
            "Greek yogurt parfait with granola",
            "Vegetable omelet with whole-grain toast",
            "Overnight oats with chia and berries",
            "Avocado toast with poached egg",
        ],
        "lunch": [
            "Chickpea and feta grain bowl",
            "Caprese panini with side salad",
            "Lentil soup with whole-grain bread",
            "Tofu buddha bowl with peanut sauce",
        ],
        "dinner": [
            "Eggplant parmesan with side greens",
            "Paneer tikka with brown rice",
            "Mushroom risotto (portion-controlled)",
            "Black bean tacos with salsa and avocado",
        ],
        "snack": [
            "Hummus with carrot sticks",
            "Cheese and whole-grain crackers",
            "Edamame",
            "Fruit and nuts",
        ],
    },
    "vegan": {
        "breakfast": [
            "Soy yogurt with granola and berries",
            "Tofu scramble with spinach and toast",
            "Oatmeal with flax, walnuts, and banana",
            "Smoothie bowl with plant protein",
        ],
        "lunch": [
            "Chickpea salad wrap",
            "Tempeh grain bowl with tahini dressing",
            "Red lentil dal with quinoa",
            "Vegan burrito bowl (beans, rice, salsa)",
        ],
        "dinner": [
            "Tofu and vegetable stir-fry with brown rice",
            "Lentil bolognese with whole-wheat pasta",
            "Stuffed bell peppers with quinoa and beans",
            "Thai green curry with tofu and jasmine rice",
        ],
        "snack": [
            "Roasted chickpeas",
            "Apple with peanut butter",
            "Mixed nuts and dried fruit",
            "Rice cakes with avocado",
        ],
    },
}

_GOAL_PHRASE = {
    "lose": "fat loss",
    "gain": "muscle gain / higher energy intake",
    "maintain": "weight maintenance",
}

_SPICY_PHRASE = {
    "yes": "spicy-friendly",
    "no": "mild",
    "either": "any spice level",
}

_ACTIVITY_PHRASE = {
    "sedentary": "mostly sedentary",
    "light": "lightly active",
    "moderate": "moderately active",
    "active": "active",
    "very_active": "very active",
}


def normalize_meal_preferences(raw: dict | None) -> dict[str, str]:
    raw = raw or {}
    diet = (raw.get("diet_style") or raw.get("diet") or "omnivore").strip().lower()
    goal = (raw.get("body_goal") or raw.get("goal") or "maintain").strip().lower()
    spicy = (raw.get("spicy") or "either").strip().lower()
    favorites = (raw.get("favorites") or raw.get("favorite_foods") or "").strip()
    restrictions = (raw.get("restrictions") or raw.get("avoid") or "").strip()

    if diet not in VALID_DIET:
        diet = "omnivore"
    if goal not in VALID_GOAL:
        goal = "maintain"
    if spicy not in VALID_SPICY:
        spicy = "either"

    return {
        "diet_style": diet,
        "body_goal": goal,
        "spicy": spicy,
        "favorites": favorites[:500],
        "restrictions": restrictions[:500],
    }


def preferences_summary(prefs: dict) -> str:
    p = normalize_meal_preferences(prefs)
    parts = [
        p["diet_style"],
        _GOAL_PHRASE[p["body_goal"]],
        _SPICY_PHRASE[p["spicy"]],
    ]
    if p["favorites"]:
        parts.append(f"favorites: {p['favorites'][:80]}")
    if p["restrictions"]:
        parts.append(f"avoid: {p['restrictions'][:80]}")
    return "; ".join(parts)


def _pick_dish(meal_key: str, prefs: dict, index: int = 0) -> str:
    pool = _DISH_POOLS.get(prefs["diet_style"], _DISH_POOLS["omnivore"])
    options = pool.get(meal_key, pool.get("lunch", ["Balanced plate"]))
    return options[index % len(options)]


def _spicy_hint(prefs: dict) -> str:
    if prefs["spicy"] == "yes":
        return " Include chili, pepper, or bold spices if you like heat."
    if prefs["spicy"] == "no":
        return " Keep flavors mild; skip heavy chili and hot sauce."
    return ""


def _goal_hint(prefs: dict) -> str:
    if prefs["body_goal"] == "lose":
        return " Favor lean protein, vegetables, and controlled portions."
    if prefs["body_goal"] == "gain":
        return " Include calorie-dense sides (grains, nuts, healthy fats) as needed."
    return ""


def enrich_meal_description(
    base: str,
    meal_key: str,
    prefs: dict,
    *,
    bmi: float | None = None,
    age: int | None = None,
    activity_level: str | None = None,
) -> str:
    """Turn a template line into a concrete dish suggestion."""
    dish = _pick_dish(meal_key, prefs)
    text = f"{dish}. {base}".strip()
    text += _spicy_hint(prefs) + _goal_hint(prefs)
    if prefs.get("favorites"):
        text += f" Work in favorites when sensible: {prefs['favorites'][:120]}."
    if prefs.get("restrictions"):
        text += f" Avoid: {prefs['restrictions'][:120]}."
    if bmi is not None:
        if bmi >= 27:
            text += " Emphasize volume from vegetables and lean protein."
        elif bmi < 18.5:
            text += " Add an energy-dense side if hunger is high."
    if age is not None and age >= 55 and meal_key == "breakfast":
        text += " Spread protein across the day starting at breakfast."
    if activity_level in ("active", "very_active") and meal_key in ("lunch", "snack"):
        text += " Training days: include quality carbs around activity."
    return re.sub(r"\s+", " ", text).strip()


def enrich_profile_blocks(
    blocks: list[dict],
    prefs: dict,
    *,
    bmi: float | None = None,
    age: int | None = None,
    activity_level: str | None = None,
) -> list[dict]:
    out = []
    for i, block in enumerate(blocks):
        key = block.get("key", "lunch")
        desc = block.get("description", "")
        enriched = enrich_meal_description(
            desc,
            key,
            prefs,
            bmi=bmi,
            age=age,
            activity_level=activity_level,
        )
        item = dict(block)
        item["description"] = enriched
        out.append(item)
    return out


def activity_phrase(activity_level: str) -> str:
    return _ACTIVITY_PHRASE.get(activity_level, activity_level.replace("_", " "))
