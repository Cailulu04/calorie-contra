import os
import re
import json
import asyncio
import quart_flask_patch
from quart import (
    Quart,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    session,
    jsonify,
)
import requests

# from flask_caching import Cache
# 删掉原来的 try-except 嵌套，只留这一行
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timedelta, time, date
from helpers import (
    apology,
    login_required,  # Ensure this is the updated decorator
    is_float,
    daily_values,
    validate_registration_form,
    get_nutritional_info,
)
from meal_plan import build_meal_recommendations, build_profile_day_meal_plan
from meal_preferences import normalize_meal_preferences
from profile_meal_gemini import generate_profile_meal_plan_gemini
from gemini_call import generate_content_with_timeout, run_sync_with_timeout
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import pandas as pd
import joblib
from sqlalchemy.orm import scoped_session
from sqlalchemy import select, insert, func, update, delete
from sqlalchemy.exc import NoResultFound, IntegrityError
#from database import setup_database, SessionLocal, User, FoodCount
###############################################################
from database import (
    setup_database,
    SessionLocal,
    User,
    FoodCount,
    ChatConversation,
    ChatMessage,
)
###############################################################

load_dotenv()

# 强制使用 REST 协议，这是解决 404 最稳定的方案
genai.configure(
    api_key=os.environ["gemini_api_key"],
    transport='rest'
)

# 使用 gemini-1.5-flash-latest，这是一个更明确的标识符
#model = genai.GenerativeModel(

######################################################
gemini_food_image_model = genai.GenerativeModel(
######################################################
    model_name="gemini-2.5-flash",
    system_instruction="""
You are a knowledgeable nutritionist. 
Analyze the image and provide the estimation strictly following these aggregation rules:

1. AGGREGATE SAME FOODS: If there are multiple items of the SAME food (e.g., 5 apples), DO NOT list them separately. Combine them into one single entry.
2. ESTIMATE TOTAL WEIGHT: Calculate the TOTAL weight of all items of that food type combined.
3. OUTPUT FORMAT: For each UNIQUE food type, provide exactly 7 lines in English (numbers only, no extra sentences):

[Food Name]
Calorie per 100g: [Value]kcal
Weight: [Total Weight of all items]g
Calorie: [Total Calorie]kcal
Protein: [Total Protein]g
Fat: [Total Fat]g
Carbs: [Total Carbs]g

Use TOTAL grams of protein, fat, and carbs for the combined weight of that food (same scope as Weight and Calorie lines).

IMPORTANT:
- Separate different food types with a blank line.
- NO introductory text, NO markdown bolding (**), NO bullet points.
- If the image displays multiple similar foods, then one food will only be output once.
""", #',' is new added
)

##################################################################
gemini_chat_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="""
You are a knowledgeable, friendly nutrition assistant for a calorie-tracking app.
- Answer questions about nutrition, calories, macros, diet patterns, food comparisons, and general wellness. Text-only questions do not require a photo.
- You may give reasonable estimates when the user describes food in words.
- Only ask the user to upload or share a food photo when they explicitly want vision-based analysis from a picture and that cannot be done from text alone.
- Do not default every reply to asking for an image.
- Match the user's language for your main answer: English questions → English prose; Chinese questions → Chinese prose. Follow the latest user message when the language switches.
- Be concise unless the user asks for more detail.
- When the user message includes a block labeled [USER_FOOD_LOG_DATA], those lines are authoritative totals from the user's saved food log in this app. Answer using those numbers for the dates shown, in the user's language. Do not ask the user to paste their food log for dates that appear in that block. If a date they ask about is not in the block, say there is no saved log for that day (or that the question is outside the data provided).

STRUCTURED FOOD LOG (enables the in-app "Add to Food log" button — required whenever you give concrete calorie/macro estimates for specific food(s) from the user's description):
- If you answer solely from [USER_FOOD_LOG_DATA], or you only give general definitions with no portion estimates, or you refuse to estimate: do NOT append this block.
- When you DO give estimates: you MUST append the block every time. Do NOT skip it because your explanation is in Chinese; the appendix is separate from the language of your main reply.
- Format rules:
  1) After your markdown answer, add a blank line, then a line containing exactly: <<<FOOD_LOG_BLOCKS>>>
  2) Next line must be the first food's name (plain text, not in brackets, not a label like "For each").
  3) Then exactly six more lines for that food, in English with these prefixes (numbers only where shown):
     Calorie per 100g: Xkcal
     Weight: Xg
     Calorie: Xkcal
     Protein: Xg
     Fat: Xg
     Carbs: Xg
  4) For additional foods, insert one blank line, then repeat steps 2–3.
  5) End with a line containing exactly: <<<END_FOOD_LOG_BLOCKS>>>
- Do not insert any other sentences or headings between <<<FOOD_LOG_BLOCKS>>> and the first food name line.
""",
)

# Second pass: JSON extract when the main chat model omits <<<FOOD_LOG_BLOCKS>>> markup.
gemini_chat_food_extract = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=(
        "You reply with JSON only (no markdown). "
        'Schema: {"items":[{"name":string,"calories":number,"protein":number,"carbs":number,"fat":number}]} '
        "Each item is one food or one combined meal the assistant estimated with numbers. "
        "Copy numeric totals from the assistant text; use 0 for a macro not stated. "
        "If there are no concrete per-food or per-meal estimates, return {\"items\":[]}."
    ),
)

gemini_profile_meal_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=(
        "You are a registered-dietitian-style meal planner for a calorie-tracking app. "
        "Suggest practical, culturally neutral meals with portion hints. "
        "Respect vegetarian/vegan rules and user restrictions. "
        "Reply in JSON when asked."
    ),
)

# Configure application
app = Quart(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

db_session = scoped_session(SessionLocal)

##########################################################################
# Max stored messages to send as Gemini chat history (FIFO by recency).
CHAT_GEMINI_MESSAGE_CAP = 40
##########################################################################

# USDA FoodData Central: prefer USDA_API_KEY (see .env), then legacy names
api_key = (
    os.getenv("USDA_API_KEY")
    or os.getenv("api_key")
    or os.getenv("API_KEY")
    or os.getenv("FDC_API_KEY")
)

_calorie_model = None


def _load_calorie_expenditure_model():
    global _calorie_model
    if _calorie_model is not None:
        return _calorie_model

    model_path = os.path.join(
        os.path.dirname(__file__), "models", "calorie-expenditure-predictor"
    )
    _calorie_model = joblib.load(model_path)
    return _calorie_model


def _calculate_bmi(height_cm: float, weight_kg: float) -> float:
    return weight_kg / ((height_cm / 100.0) ** 2)


def _calculate_bmr_mifflin_st_jeor(
    gender: str, age: int, height_cm: float, weight_kg: float
) -> float:
    # Mifflin–St Jeor (kcal/day)
    base = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age
    if gender.lower() == "male":
        return base + 5.0
    return base - 161.0


_ACTIVITY_FACTORS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}


@app.before_serving
async def startup():
    await setup_database()


@app.after_request
async def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

############################################
@app.route("/", methods=["GET"])
async def root():
    return redirect("/home")
############################################

@app.route("/home", methods=["GET", "POST"])
async def index():
    """displays search form"""
    if request.method == "POST":
        query = (await request.form).get("q")

        # long string or words not accepted
        if len(query) > 30:
            return await apology("String length Error", 404)
        # Redirect to the results page with the search query
        return redirect(url_for("search"))
    else:
        return await render_template("home.html")


@app.route("/search", methods=["GET"])
async def search():
    """returns lists of food for matched query"""
    query = request.args.get("q")

    return await render_template("search_foods.html", query=query, api_key=api_key)


@app.route("/api/search_foods", methods=["GET"])
async def search_foods():
    query = request.args.get("query")
    page = request.args.get("page", 1)
    data_type = request.args.get("dataType", "")

    if not query:
        return jsonify({"error": "Query parameter is required", "foods": [], "totalPages": 0}), 400

    if not api_key:
        return jsonify(
            {
                "error": "USDA API key is not configured. Set USDA_API_KEY (or api_key) in your environment.",
                "foods": [],
                "totalPages": 0,
            }
        ),
        503

    try:
        params = {
            "api_key": api_key,
            "query": query,
            "pageNumber": page,
            "pageSize": 50,
        }
        if data_type:
            params["dataType"] = data_type
        response = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return jsonify(data)
    except requests.RequestException as e:
        return jsonify({"error": str(e), "foods": [], "totalPages": 0}), 500


@app.route("/login", methods=["GET", "POST"])
async def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        form = await request.form
        # Ensure username was submitted
        if not form.get("username"):
            return await apology("must provide username", 403)

        # Ensure password was submitted
        elif not form.get("password"):
            return await apology("must provide password", 403)

        # Query database for username
        username = form.get("username")
        password = form.get("password")

        try:
            stmt = select(User).where(User.username == username)
            result = await db_session.execute(stmt)
            user = result.scalar_one()

            # Ensure username exists and password is correct
            if not check_password_hash(user.hash, password):
                return await apology("invalid username and/or password", 403)

            # Remember which user has logged in
            session["user_id"] = user.id

            # Redirect user to home page
            return redirect("/home")

        except NoResultFound:
            return await apology("invalid username and/or password", 403)

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return await render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
async def register():
    """register user"""

    # user reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        form = await request.form
        username = form.get("username")
        email = form.get("email")
        password = form.get("password")
        confirm_pass = form.get("confirm_password")

        error_message = validate_registration_form(
            username, email, password, confirm_pass
        )
        if error_message:
            apology(error_message, 400)

        try:
            stmt = select(User).where(User.username == username)
            result = await db_session.execute(stmt)
            user_check = result.scalar_one_or_none()

            # check if the username is taken using user_check
            if user_check:
                return await apology("Username already taken", 400)

            # hashes the plain-text password
            hashed_password = generate_password_hash(password)

            # Add the newly registered user to the database
            new_user = User(username=username, email=email, hash=hashed_password)
            db_session.add(new_user)
            await db_session.commit()

            # log in our newly registered user into the website
            session["user_id"] = new_user.id

            await flash("Registered!", "success")
            return redirect("/home")

        except IntegrityError:
            await db_session.rollback()
            return await apology("Username already taken", 400)

    # User reached route via GET (as by clicking register or via redirect)
    else:
        return await render_template("register.html")


@app.route("/logout")
async def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/home")


@app.route("/food/<int:id>", methods=["GET"])
@login_required
async def food(id):
    """display's the selected food's nutrition facts"""

    food = await get_nutritional_info(id, api_key)

    return await render_template("food.html", food=food)


@app.route("/food-log", methods=["GET", "POST"])
@login_required
async def food_log():
    """User's food log or diet history"""

    # if user reached POST (as by submitting a form via POST)
    if request.method == "POST":
        data = await request.get_json()
        ok, err = await _add_food_log_from_payload(data, session["user_id"])
        if ok:
            return {"message": "success"}
        return await apology(err[0], err[1])

    # if the user reached GET (as by clicking food_log)
    else:
        # Get the selected date from the form or use the current date as default
        selected_date_str = request.args.get("selected_date")
        if selected_date_str:
            selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d")
        else:
            selected_date = datetime.now()

        prev_sunday = selected_date - timedelta(days=selected_date.weekday() + 1)
        week_dates = []
        # Initialize a list to store the dates for the entire week
        for i in range(7):
            day_date = prev_sunday + timedelta(days=i)
            week_dates.append(
                {"month": day_date.month, "day": day_date.day, "year": day_date.year}
            )

        # store our queries in a list
        food_log_query = []
        # get the user's food intake for the last 7 days
        for date in week_dates:
            stmt = select(
                func.sum(FoodCount.calories).label("total_calories"),
                func.sum(FoodCount.protein).label("total_protein"),
                func.sum(FoodCount.carbs).label("total_carbs"),
                func.sum(FoodCount.fat).label("total_fat"),
            ).where(
                FoodCount.user_id == session["user_id"],
                FoodCount.month == date["month"],
                FoodCount.day == date["day"],
                FoodCount.year == date["year"],
            )
            result = await db_session.execute(stmt)
            food_log_query.append(result.fetchone())

        # Initialize variables to handle no data case
        food_log = None

        week_names = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]

        week_days = None
        cal_bar_max = 1.0
        week_trend = None

        if food_log_query:
            total_calories = []
            total_protein = []
            total_carbs = []
            total_fat = []

            for result in food_log_query:
                total_calories.append(result.total_calories or 0)
                total_protein.append(result.total_protein or 0)
                total_carbs.append(result.total_carbs or 0)
                total_fat.append(result.total_fat or 0)

            week_days = []
            for i in range(7):
                dd = prev_sunday + timedelta(days=i)
                tc = float(total_calories[i] or 0)
                p = float(total_protein[i] or 0)
                c = float(total_carbs[i] or 0)
                f = float(total_fat[i] or 0)
                kp, kc, kf = 4.0 * p, 4.0 * c, 9.0 * f
                macro_kcal = kp + kc + kf
                if tc > 0 and macro_kcal > tc:
                    scale = tc / macro_kcal
                    kp, kc, kf = kp * scale, kc * scale, kf * scale
                other_kcal = max(0.0, tc - kp - kc - kf)

                week_days.append(
                    {
                        "date_iso": dd.strftime("%Y-%m-%d"),
                        "weekday": week_names[i],
                        "total_calories": tc,
                        "stack": {
                            "fat": kf,
                            "carb": kc,
                            "prot": kp,
                            "other": other_kcal,
                        },
                    }
                )
            cal_bar_max = max(float(x or 0) for x in total_calories)
            if cal_bar_max < 1:
                cal_bar_max = 1.0

            n = len(week_days)
            trend_pts = []
            trend_dots = []
            for i, wd in enumerate(week_days):
                tc = float(wd["total_calories"] or 0)
                x = (i + 0.5) * (100.0 / n)
                y = 90.0 - min(76.0, (tc / cal_bar_max) * 76.0)
                trend_pts.append(f"{x:.3f},{y:.3f}")
                trend_dots.append({"x": round(x, 3), "y": round(y, 3)})
            grid_fracs = (0.25, 0.5, 0.75)
            week_trend = {
                "polyline_points": " ".join(trend_pts),
                "dots": trend_dots,
                "grid_ys": [90.0 - 76.0 * g for g in grid_fracs],
                "grid_labels": [int(round(cal_bar_max * (1.0 - g))) for g in grid_fracs],
            }
        else:
            week_days = None
            cal_bar_max = 1.0
            week_trend = None

        date_input_value = selected_date.strftime("%Y-%m-%d")
        week_anchor = date_input_value

        return await render_template(
            "food-log.html",
            food_log=food_log,
            selected_date=date_input_value,
            week_days=week_days,
            cal_bar_max=cal_bar_max,
            week_anchor=week_anchor,
            week_trend=week_trend,
        )


@app.route("/api/food-log", methods=["POST"])
async def api_food_log():
    """JSON API to append a food log entry (same rules as POST /food-log)."""
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Please log in to save to your food log."}), 401
    data = await request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "Invalid JSON body"}), 400
    ok, err = await _add_food_log_from_payload(data, session["user_id"])
    if ok:
        return jsonify({"ok": True, "message": "success"})
    return jsonify({"ok": False, "error": err[0]}), err[1]


@app.route("/food-log/day", methods=["GET"])
@login_required
async def food_log_day():
    """Single-day breakdown: foods that make up that day's log."""
    date_str = (request.args.get("date") or "").strip()
    week_anchor = (request.args.get("week_anchor") or date_str).strip()
    if not date_str:
        return redirect(url_for("food_log"))
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return redirect(url_for("food_log"))

    stmt = (
        select(FoodCount)
        .where(
            FoodCount.user_id == session["user_id"],
            FoodCount.year == day.year,
            FoodCount.month == day.month,
            FoodCount.day == day.day,
        )
        .order_by(FoodCount.id)
    )
    result = await db_session.execute(stmt)
    entries = result.scalars().all()

    rows = []
    total_cal = total_p = total_c = total_f = 0.0
    for e in entries:
        rows.append(
            {
                "id": e.id,
                "food_name": e.food_name,
                "calories": e.calories,
                "protein": e.protein,
                "carbs": e.carbs,
                "fat": e.fat,
                "logged_time": f"{e.hour:02d}:{e.minute:02d}",
            }
        )
        total_cal += float(e.calories or 0)
        total_p += float(e.protein or 0)
        total_c += float(e.carbs or 0)
        total_f += float(e.fat or 0)

    meal_recommendations = None
    daily_goal_raw = (request.args.get("daily_goal") or "").strip()
    if daily_goal_raw:
        try:
            daily_goal = float(daily_goal_raw)
            if 800 <= daily_goal <= 6000:
                prefs = None
                pref_json = (request.args.get("preferences") or "").strip()
                if pref_json:
                    try:
                        prefs = json.loads(pref_json)
                    except json.JSONDecodeError:
                        prefs = None
                meal_recommendations = build_meal_recommendations(
                    daily_goal,
                    total_cal,
                    total_p,
                    total_c,
                    total_f,
                    preferences=prefs,
                )
        except ValueError:
            meal_recommendations = None

    return await render_template(
        "food-log-day.html",
        day_iso=date_str,
        day_label=day.strftime("%A, %B %d, %Y"),
        week_anchor=week_anchor or date_str,
        entries=rows,
        totals={
            "calories": total_cal,
            "protein": total_p,
            "carbs": total_c,
            "fat": total_f,
        },
        meal_recommendations=meal_recommendations,
        daily_goal_value=daily_goal_raw,
    )


@app.route("/api/food-log/<int:entry_id>", methods=["PATCH", "DELETE"])
async def api_food_log_entry(entry_id):
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Please log in."}), 401
    uid = session["user_id"]

    if request.method == "DELETE":
        ok, err = await _delete_food_log_entry(entry_id, uid)
        if ok:
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": err[0]}), err[1]

    data = await request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "Invalid JSON body"}), 400
    ok, err = await _update_food_log_from_payload(entry_id, uid, data)
    if ok:
        return jsonify({"ok": True, "message": "updated"})
    return jsonify({"ok": False, "error": err[0]}), err[1]


_CONTACT_TEAM = (
    ("Hanhui ZHAN", "s230034060"),
    ("Weilin LI", "s230034027"),
    ("Tianyue ZHAO", "s230034064"),
    ("Xinlu CAI", "s230034002"),
    ("Gengchen Jiang", "t330034021"),
)
_CONTACT_EMAIL_DOMAIN = "mail.bnbu.edu.cn"


@app.route("/contact", methods=["GET"])
async def contact():
    members = [
        {
            "name": name,
            "student_id": sid,
            "email": f"{sid}@{_CONTACT_EMAIL_DOMAIN}",
        }
        for name, sid in _CONTACT_TEAM
    ]
    return await render_template(
        "contact.html",
        group_name="Group 3",
        members=members,
    )


@app.route('/profile')
@login_required  # 要求登录后才能访问
async def profile():
    return await render_template('profile.html')


@app.route("/api/calorie-needs", methods=["POST"])
@login_required
async def calorie_needs():
    data = await request.get_json()
    if not data:
        return jsonify({"message": "Invalid input"}), 400

    try:
        gender = (data.get("gender") or "").strip().lower()
        age = int(data.get("age"))
        height = float(data.get("height"))
        weight = float(data.get("weight"))
        duration = float(data.get("duration"))
        heart_rate = float(data.get("heart_rate"))
        body_temp = float(data.get("body_temp"))
        activity_level = (data.get("activity_level") or "sedentary").strip().lower()
    except (TypeError, ValueError):
        return jsonify({"message": "Invalid numeric values"}), 400

    if gender not in {"male", "female"}:
        return jsonify({"message": "Invalid gender"}), 400

    if activity_level not in _ACTIVITY_FACTORS:
        return jsonify({"message": "Invalid activity level"}), 400

    if not (1 <= age <= 160):
        return jsonify({"message": "Invalid age"}), 400
    if height <= 0:
        return jsonify({"message": "Height must be positive"}), 400
    if not (50 <= height <= 300):
        return jsonify({"message": "Invalid height"}), 400
    if weight <= 0:
        return jsonify({"message": "Weight must be positive"}), 400
    if not (10 <= weight <= 500):
        return jsonify({"message": "Invalid weight"}), 400
    if duration <= 0:
        return jsonify({"message": "Duration must be positive"}), 400
    if not (1 <= duration <= 600):
        return jsonify({"message": "Invalid duration"}), 400
    if heart_rate <= 0:
        return jsonify({"message": "Heart rate must be positive"}), 400
    if not (30 <= heart_rate <= 250):
        return jsonify({"message": "Invalid heart rate"}), 400
    if body_temp <= 0:
        return jsonify({"message": "Body temperature must be positive"}), 400
    # Resting oral / typical adult range (°C); matches profile form guidance
    if not (36.0 <= body_temp <= 37.2):
        return jsonify(
            {
                "message": "Body temperature should be a resting value within the normal human body temperature range (36.0–37.2 °C).",
            }
        ), 400

    bmi = _calculate_bmi(height, weight)
    model = _load_calorie_expenditure_model()

    # Trained model uses Age, Gender, Duration, Heart_Rate, Body_Temp, bmi.
    # Body temperature is usually a weaker driver than heart rate and duration,
    # so small changes within a normal resting band barely affect the prediction
    # (and outputs are rounded to two decimal places).
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": "male" if gender == "male" else "female",
                "Duration": duration,
                "Heart_Rate": heart_rate,
                "Body_Temp": body_temp,
                "bmi": bmi,
            }
        ]
    )

    exercise_calories = float(model.predict(input_df)[0])
    bmr = float(_calculate_bmr_mifflin_st_jeor(gender, age, height, weight))
    tdee = bmr * _ACTIVITY_FACTORS[activity_level]
    daily_need = tdee + exercise_calories

    return jsonify(
        {
            "bmr": round(bmr, 2),
            "activity_factor": _ACTIVITY_FACTORS[activity_level],
            "tdee": round(tdee, 2),
            "exercise_calories": round(exercise_calories, 2),
            "daily_calories": round(daily_need, 2),
        }
    )


@app.route("/api/meal-recommend", methods=["POST"])
@login_required
async def api_meal_recommend():
    data = await request.get_json()
    if not data:
        return jsonify({"message": "Invalid input"}), 400
    try:
        daily_calories = float(data.get("daily_calories"))
    except (TypeError, ValueError):
        return jsonify({"message": "daily_calories is required"}), 400
    if not (800 <= daily_calories <= 6000):
        return jsonify({"message": "daily_calories must be between 800 and 6000"}), 400

    date_str = (data.get("date") or "").strip()
    if date_str:
        try:
            day = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({"message": "Invalid date"}), 400
    else:
        day = datetime.now()

    uid = session["user_id"]
    stmt = select(FoodCount).where(
        FoodCount.user_id == uid,
        FoodCount.year == day.year,
        FoodCount.month == day.month,
        FoodCount.day == day.day,
    )
    result = await db_session.execute(stmt)
    entries = result.scalars().all()
    total_cal = total_p = total_c = total_f = 0.0
    for e in entries:
        total_cal += float(e.calories or 0)
        total_p += float(e.protein or 0)
        total_c += float(e.carbs or 0)
        total_f += float(e.fat or 0)

    prefs = data.get("preferences")
    plan = build_meal_recommendations(
        daily_calories,
        total_cal,
        total_p,
        total_c,
        total_f,
        preferences=prefs,
    )
    return jsonify(plan)


@app.route("/api/profile-meal-plan", methods=["POST"])
@login_required
async def api_profile_meal_plan():
    data = await request.get_json()
    if not data:
        return jsonify({"message": "Invalid input"}), 400

    try:
        daily_calories = float(data.get("daily_calories"))
        gender = (data.get("gender") or "").strip().lower()
        age = int(data.get("age"))
        height_cm = float(data.get("height"))
        weight_kg = float(data.get("weight"))
        activity_level = (data.get("activity_level") or "moderate").strip().lower()
    except (TypeError, ValueError):
        return jsonify({"message": "Invalid profile fields"}), 400

    if not (800 <= daily_calories <= 6000):
        return jsonify({"message": "daily_calories must be between 800 and 6000"}), 400
    if gender not in {"male", "female"}:
        return jsonify({"message": "Invalid gender"}), 400
    if activity_level not in _ACTIVITY_FACTORS:
        return jsonify({"message": "Invalid activity level"}), 400

    prefs = normalize_meal_preferences(data.get("preferences"))
    daily_int = int(round(daily_calories))

    try:
        plan = await generate_profile_meal_plan_gemini(
            gemini_profile_meal_model,
            daily_kcal=daily_int,
            gender=gender,
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            activity_level=activity_level,
            preferences=prefs,
        )
    except Exception as exc:
        print(f"profile-meal-plan error: {exc}")
        plan = build_profile_day_meal_plan(
            daily_int,
            gender=gender,
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            activity_level=activity_level,
            preferences=prefs,
        )
        plan["ai_note"] = f"AI unavailable ({exc}). Showing rule-based plan."

    return jsonify({"profile_meal_plan": plan})

# new added code


def parse_gemini_food_blocks(text: str):
    """
    Parse Gemini food vision output: 7-line blocks per food (English).
    Returns list of dicts: name, calories, protein, carbs, fat, optional per_100g, weight_g.
    """
    text = (text or "").strip()
    if not text:
        return []

    blocks = re.split(r"\n\s*\n", text)
    items = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        name = lines[0]
        per_100g = None
        weight_g = None
        calories = None
        protein = None
        fat = None
        carbs = None
        for ln in lines[1:]:
            m = re.search(r"Calorie per 100g:\s*([\d.]+)", ln, re.I)
            if m:
                per_100g = float(m.group(1))
            m = re.search(r"Weight:\s*([\d.]+)\s*g", ln, re.I)
            if m:
                weight_g = float(m.group(1))
            m = re.search(r"Calorie:\s*([\d.]+)\s*kcal", ln, re.I)
            if m:
                calories = float(m.group(1))
            m = re.search(r"Protein:\s*([\d.]+)\s*g", ln, re.I)
            if m:
                protein = float(m.group(1))
            m = re.search(r"Fat:\s*([\d.]+)\s*g", ln, re.I)
            if m:
                fat = float(m.group(1))
            m = re.search(r"Carbs:\s*([\d.]+)\s*g", ln, re.I)
            if m:
                carbs = float(m.group(1))
        if carbs is None:
            block_text = "\n".join(lines)
            m = re.search(r"Carbohydrate(?:s)?:\s*([\d.]+)\s*g", block_text, re.I)
            if m:
                carbs = float(m.group(1))
        if per_100g is not None and weight_g is not None and calories is None:
            calories = round(per_100g * weight_g / 100.0, 1)
        if calories is None:
            calories = 0.0
        protein = float(protein or 0)
        fat = float(fat or 0)
        carbs = float(carbs or 0)
        items.append(
            {
                "name": name,
                "calories": float(calories),
                "protein": protein,
                "fat": fat,
                "carbs": carbs,
                "calories_per_100g": per_100g,
                "weight_g": weight_g,
            }
        )
    return items


_CHAT_FOOD_START_RE = re.compile(r"<<<\s*FOOD_LOG_BLOCKS\s*>>>", re.I)
_CHAT_FOOD_END_RE = re.compile(r"<<<\s*END_FOOD_LOG_BLOCKS\s*>>>", re.I)


def _gemini_response_text(response) -> str:
    """Best-effort plain text from a generate_content / chat response."""
    t = getattr(response, "text", None)
    if t:
        return str(t)
    try:
        for cand in response.candidates or []:
            for p in getattr(cand.content, "parts", []) or []:
                if getattr(p, "text", None):
                    return str(p.text)
    except (TypeError, AttributeError):
        pass
    return ""


def _wants_chat_food_estimate(prompt: str) -> bool:
    """Heuristic: user is asking for food nutrition numbers (not e.g. pure thanks)."""
    if not (prompt and str(prompt).strip()):
        return False
    p = str(prompt).strip()
    if len(p) < 3:
        return False
    if re.match(r"^(hi|hello|hey|thanks|thank you|谢谢|谢了|好的|ok|okay)\b", p, re.I):
        return False
    return bool(
        re.search(
            r"calor|kcal|大卡|热量|千?卡|\bkj\b|macro|protein|carb|fat|蛋白|碳水|脂肪|"
            r"estimate|portion|serving|一份|一碗|一个|一根|一片|营养|多少克|几克|"
            r"grams?|\d\s*克|how\s+much|多少|营养价值|成分|含糖|含多少|几卡|"
            r"香蕉|苹果|鸡蛋|米饭|面包|牛奶|咖啡|肉|菜|饭|面|汤",
            p,
            re.I,
        )
    )


def _assistant_has_loggable_nutrition_signal(text: str) -> bool:
    """Assistant reply looks like it stated calories/macros for food (even if user wording was vague)."""
    if not (text and str(text).strip()):
        return False
    t = str(text)
    return bool(
        re.search(
            r"kcal|大卡|千卡|千焦|热量|卡路里|\bcalories?\b|蛋白(质)?|碳水|脂肪|"
            r"macro|nutrient|约\s*\d+|\d+\s*卡|\d+\s*kcal|P\s*[：:]\s*\d|C\s*[：:]\s*\d|F\s*[：:]\s*\d",
            t,
            re.I,
        )
    )


def _strip_json_fence(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, re.I)
    if m:
        return m.group(1).strip()
    return s


def _json_loads_loose(raw: str) -> dict | None:
    """Parse model JSON; tolerate extra prose by slicing first {...} span."""
    s = _strip_json_fence((raw or "").strip())
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except json.JSONDecodeError:
            return None
    return None


def _extract_loggable_items_chat_fallback_sync(user_prompt: str, assistant_text: str) -> list[dict]:
    """One-shot JSON extraction when the main model omits FOOD_LOG_BLOCKS."""
    if not (assistant_text and str(assistant_text).strip()):
        return []
    if not re.search(r"\d", assistant_text):
        return []
    payload = (
        "Extract structured rows from this exchange.\n\nUSER:\n"
        + str(user_prompt)[:3500]
        + "\n\nASSISTANT:\n"
        + str(assistant_text)[:14000]
    )
    tail = (
        '\n\nReturn ONLY one JSON object: {"items":[{"name":string,"calories":number,'
        '"protein":number,"carbs":number,"fat":number}]} — no markdown, no code fences.'
    )

    def _parse_items_from_data(data) -> list[dict]:
        if not isinstance(data, dict):
            return []
        items_raw = data.get("items")
        if not isinstance(items_raw, list):
            return []
        out: list[dict] = []
        for it in items_raw:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or it.get("food") or "").strip()
            if len(name) < 1:
                continue
            try:
                cal = float(it.get("calories", 0))
                pr = float(it.get("protein", 0))
                cr = float(it.get("carbs", 0))
                fa = float(it.get("fat", 0))
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "name": name[:220],
                    "calories": max(0.0, cal),
                    "protein": max(0.0, pr),
                    "carbs": max(0.0, cr),
                    "fat": max(0.0, fa),
                    "calories_per_100g": None,
                    "weight_g": None,
                }
            )
        return out

    for use_json_mime in (True, False):
        try:
            if use_json_mime:
                resp = gemini_chat_food_extract.generate_content(
                    payload + tail,
                    generation_config=GenerationConfig(
                        temperature=0.05,
                        response_mime_type="application/json",
                    ),
                )
            else:
                resp = gemini_chat_food_extract.generate_content(
                    payload + tail,
                    generation_config=GenerationConfig(temperature=0.05),
                )
            raw = _gemini_response_text(resp)
            data = _json_loads_loose(raw)
            if data is None:
                continue
            out = _parse_items_from_data(data)
            if out:
                return out
        except Exception as e:
            print(f"_extract_loggable_items_chat_fallback_sync ({use_json_mime=}): {e}")
    return []


def _sanitize_chat_food_log_inner(inner: str) -> str:
    """
    Drop prose lines the model sometimes pastes after <<<FOOD_LOG_BLOCKS>>> so
    parse_gemini_food_blocks sees a real food name on the first line.
    """
    out_lines: list[str] = []
    for ln in (inner or "").split("\n"):
        s = ln.strip()
        if not s:
            out_lines.append("")
            continue
        if re.match(
            r"^(For EACH|ONLY when|After your|Append|Example|Note:|"
            r"\[Food Name\]|\[Value\]|\[Total|\[Weight\])",
            s,
            re.I,
        ):
            continue
        if "same pattern as photo" in s.lower():
            continue
        out_lines.append(ln.rstrip())
    # Collapse repeated blank lines at start
    text = "\n".join(out_lines).strip()
    return text


def strip_chat_food_log_blocks(text: str) -> tuple[str, list[dict]]:
    """
    Remove optional machine-readable food blocks from chat model output.
    Same inner format as image analysis (parse_gemini_food_blocks).
    Markers are matched case-insensitively with optional inner spaces.
    """
    raw = (text or "").strip()
    sm = _CHAT_FOOD_START_RE.search(raw)
    if not sm:
        return raw, []
    em = _CHAT_FOOD_END_RE.search(raw, sm.end())
    if not em:
        return raw, []
    inner = raw[sm.end() : em.start()].strip()
    inner = re.sub(r"^\s*```(?:\w*)?\s*", "", inner)
    inner = re.sub(r"\s*```\s*$", "", inner)
    inner = _sanitize_chat_food_log_inner(inner)
    before = raw[: sm.start()].rstrip()
    after = raw[em.end() :].lstrip()
    sep = "\n\n" if before and after else ""
    visible = (before + sep + after).strip()
    items = parse_gemini_food_blocks(inner)
    return visible, items


def _entry_datetime_from_payload(data: dict):
    """
    Timestamp stored on FoodCount (year/month/day/hour/minute).
    Optional logged_date 'YYYY-MM-DD' and logged_time 'HH:MM' (24h).
    Defaults to now when logged_date is omitted. When logged_date is set but logged_time
    is omitted, uses the current local clock on that calendar day (not a fixed noon).
    """
    raw = data.get("logged_date")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return datetime.now(), None
    if not isinstance(raw, str):
        return None, ("logged_date must be a string YYYY-MM-DD", 400)
    try:
        d = datetime.strptime(raw.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None, ("logged_date must be YYYY-MM-DD", 400)

    t_raw = data.get("logged_time")
    if t_raw and isinstance(t_raw, str) and t_raw.strip():
        try:
            parts = t_raw.strip().split(":")
            h = max(0, min(23, int(parts[0])))
            m = max(0, min(59, int(parts[1]) if len(parts) > 1 else 0))
            return datetime.combine(d, time(h, m)), None
        except (ValueError, IndexError):
            return None, ("logged_time must be HH:MM (24-hour)", 400)

    # No explicit time: use the actual clock now. Calendar day stays `d` from logged_date.
    # (Previously non-"today" dates used noon, which made many entries show 12:00; UTC vs local
    # date mismatch also made "today" from the browser look like a different day on the server.)
    now = datetime.now()
    if d == now.date():
        return now, None
    return datetime.combine(d, now.time()), None


async def _add_food_log_from_payload(data, user_id):
    """
    Validate payload and insert FoodCount. Returns (True, None) or (False, (message, status_code)).
    """
    stamp, stamp_err = _entry_datetime_from_payload(data)
    if stamp_err:
        return False, stamp_err

    food = data.get("food")
    calorie = data.get("calories")
    protein = data.get("protein")
    carbs = data.get("carbs")
    fat = data.get("fat")

    if not food:
        return False, ("Missing food name", 400)

    if calorie is None or protein is None or carbs is None or fat is None:
        return False, ("calories, protein, carbs, and fat are required", 400)

    try:
        calorie = float(calorie)
        protein = float(protein)
        carbs = float(carbs)
        fat = float(fat)
    except (TypeError, ValueError):
        return False, ("Invalid numeric values", 400)

    if (
        not is_float(calorie)
        or not is_float(protein)
        or not is_float(carbs)
        or not is_float(fat)
    ):
        return False, ("Invalid values", 400)

    if int(calorie) < 0 or float(protein) < 0 or float(carbs) < 0 or float(fat) < 0:
        return False, ("Negative values not allowed", 400)

    stmt = insert(FoodCount).values(
        user_id=user_id,
        food_name=food,
        calories=calorie,
        protein=protein,
        carbs=carbs,
        fat=fat,
        month=stamp.month,
        day=stamp.day,
        year=stamp.year,
        hour=stamp.hour,
        minute=stamp.minute,
    )
    await db_session.execute(stmt)
    await db_session.commit()
    return True, None


async def _food_count_owned(entry_id: int, user_id: int):
    stmt = select(FoodCount).where(
        FoodCount.id == entry_id,
        FoodCount.user_id == user_id,
    )
    result = await db_session.execute(stmt)
    return result.scalar_one_or_none()


async def _update_food_log_from_payload(entry_id: int, user_id: int, data: dict):
    row = await _food_count_owned(entry_id, user_id)
    if row is None:
        return False, ("Entry not found", 404)

    food = data.get("food")
    calorie = data.get("calories")
    protein = data.get("protein")
    carbs = data.get("carbs")
    fat = data.get("fat")

    if not food:
        return False, ("Missing food name", 400)
    if calorie is None or protein is None or carbs is None or fat is None:
        return False, ("calories, protein, carbs, and fat are required", 400)
    try:
        calorie = float(calorie)
        protein = float(protein)
        carbs = float(carbs)
        fat = float(fat)
    except (TypeError, ValueError):
        return False, ("Invalid numeric values", 400)
    if (
        not is_float(calorie)
        or not is_float(protein)
        or not is_float(carbs)
        or not is_float(fat)
    ):
        return False, ("Invalid values", 400)
    if int(calorie) < 0 or float(protein) < 0 or float(carbs) < 0 or float(fat) < 0:
        return False, ("Negative values not allowed", 400)

    if data.get("logged_date") and str(data.get("logged_date")).strip():
        stamp, stamp_err = _entry_datetime_from_payload(data)
        if stamp_err:
            return False, stamp_err
    else:
        stamp = datetime(row.year, row.month, row.day, row.hour, row.minute)

    stmt = (
        update(FoodCount)
        .where(FoodCount.id == entry_id, FoodCount.user_id == user_id)
        .values(
            food_name=food,
            calories=calorie,
            protein=protein,
            carbs=carbs,
            fat=fat,
            year=stamp.year,
            month=stamp.month,
            day=stamp.day,
            hour=stamp.hour,
            minute=stamp.minute,
        )
    )
    await db_session.execute(stmt)
    await db_session.commit()
    return True, None


async def _delete_food_log_entry(entry_id: int, user_id: int):
    row = await _food_count_owned(entry_id, user_id)
    if row is None:
        return False, ("Entry not found", 404)
    await db_session.execute(
        delete(FoodCount).where(
            FoodCount.id == entry_id,
            FoodCount.user_id == user_id,
        )
    )
    await db_session.commit()
    return True, None

#############################################################
# English + Chinese cues that the user is asking about their logged intake (not general nutrition trivia).
_FOOD_LOG_QUERY_HINT = re.compile(
    r"food\s*log|食物记录|食物\s*记录|饮食记录|摄入|卡路里|热量|\bkcal\b|吃了多少|吃了\s*多少|记录里|根据.*记录|"
    r"my\s+(?:food\s*)?log|from\s+my\s+log|logged\s+(?:food|meals|intake)|what\s+did\s+i\s+(?:eat|have)|"
    r"how\s+much\s+did\s+i\s+eat|calorie\s+intake|calories\s+(?:did\s+i|i\s+(?:ate|had|logged))|"
    r"\bmy\s+intake\b|tracking\s+app|diary|journal|"
    r"\bcalories?\b",
    re.I,
)

# English month names for natural-language dates ("May 13, 2026").
_ENGLISH_MONTH_ALT = (
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sept?\.?|oct\.?|nov\.?|dec\.?"
)

_ENGLISH_MONTH_NUM = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _english_month_to_num(token: str) -> int | None:
    if not token:
        return None
    key = token.lower().strip().rstrip(".")
    return _ENGLISH_MONTH_NUM.get(key)


def _text_hints_food_log_calories(text: str) -> bool:
    if not (text and text.strip()):
        return False
    if _FOOD_LOG_QUERY_HINT.search(text):
        return True
    return bool(_extract_explicit_dates_from_text(text))

def _extract_explicit_dates_from_text(text: str) -> list[date]:
    """Parse calendar dates from free text. Unique dates, first occurrence order."""
    if not text:
        return []
    found: list[date] = []
    seen: set[date] = set()
    def _add(y: int, mo: int, d: int):
        try:
            dt = date(y, mo, d)
        except ValueError:
            return
        if dt not in seen:
            seen.add(dt)
            found.append(dt)
    # Use (?!\d) / (?<!\d) instead of \\b: after digits, Chinese letters are
    # "word" chars in Python, so \\b does not match before e.g. "2026.5.13这一天".
    for m in re.finditer(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", text):
        _add(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in re.finditer(r"(?<!\d)(\d{4})[\s./](\d{1,2})[\s./](\d{1,2})(?!\d)", text):
        _add(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?", text):
        _add(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    y0 = datetime.now().year
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日", text):
        _add(y0, int(m.group(1)), int(m.group(2)))

    # English: "May 13, 2026" / "May 13 2026" / "13 May 2026" / optional ordinals on day
    mo_alt = _ENGLISH_MONTH_ALT
    for m in re.finditer(
        rf"(?i)(?<![A-Za-z])(?P<mon>{mo_alt})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})(?!\d)",
        text,
    ):
        mo = _english_month_to_num(m.group("mon"))
        if mo:
            _add(int(m.group("y")), mo, int(m.group("d")))
    for m in re.finditer(
        rf"(?i)(?<!\d)(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mon>{mo_alt})\s*,?\s+(?P<y>\d{{4}})(?!\d)",
        text,
    ):
        mo = _english_month_to_num(m.group("mon"))
        if mo:
            _add(int(m.group("y")), mo, int(m.group("d")))

    # Numeric M/D/YYYY or D/M/YYYY when year is last (US default if ambiguous)
    for m in re.finditer(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)", text):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12:
            day, month = a, b
        elif b > 12:
            month, day = a, b
        else:
            month, day = a, b
        _add(y, month, day)

    return found

def _extract_relative_day_dates_for_log(text: str) -> list[date]:
    if not _text_hints_food_log_calories(text):
        return []
    out: list[date] = []
    seen: set[date] = set()
    today = datetime.now().date()
    for pat, d in (
        (r"今天|今日", today),
        (r"昨天", today - timedelta(days=1)),
        (r"前天", today - timedelta(days=2)),
        (r"\btoday\b", today),
        (r"\byesterday\b", today - timedelta(days=1)),
        (r"\bthe\s+day\s+before\s+yesterday\b|\bday\s+before\s+yesterday\b", today - timedelta(days=2)),
    ):
        if re.search(pat, text) and d not in seen:
            seen.add(d)
            out.append(d)
    return out

def _wants_week_food_log_summary(text: str) -> bool:
    if not _text_hints_food_log_calories(text):
        return False
    return bool(
        re.search(
            r"本周|这周|这一周|七天|7\s*天|一周|整周|过去一周|上一周|下一周|那一周|该周|"
            r"\bthis\s+week\b|\bpast\s+week\b|\blast\s+week\b|\bfor\s+the\s+week\b|"
            r"\bweekly\b|\blast\s*7\s*days\b|\bpast\s*7\s*days\b|\b7[-\s]?day\b|\bseven\s+days\b",
            text,
            re.I,
        )
    )


def _week_dates_from_anchor(anchor: date) -> list[date]:
    """Sunday–Saturday week used by /food-log (week starts Sunday)."""
    prev_sunday = anchor - timedelta(days=anchor.weekday() + 1)
    return [prev_sunday + timedelta(days=i) for i in range(7)]


async def _fetch_day_food_log_detail(user_id: int, d: date) -> str:
    stmt = (
        select(FoodCount)
        .where(
            FoodCount.user_id == user_id,
            FoodCount.year == d.year,
            FoodCount.month == d.month,
            FoodCount.day == d.day,
        )
        .order_by(FoodCount.hour, FoodCount.minute, FoodCount.id)
    )
    result = await db_session.execute(stmt)
    entries = list(result.scalars().all())
    iso = d.isoformat()
    weekday_names = (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    )
    wk = weekday_names[d.weekday()]
    if not entries:
        return f"{iso} ({wk}): no food log entries (0 kcal total)."
    total_cal = sum(float(e.calories or 0) for e in entries)
    total_p = sum(float(e.protein or 0) for e in entries)
    total_c = sum(float(e.carbs or 0) for e in entries)
    total_f = sum(float(e.fat or 0) for e in entries)
    lines = [
        f"{iso} ({wk}): total {total_cal:.0f} kcal; protein {total_p:.1f}g, "
        f"carbs {total_c:.1f}g, fat {total_f:.1f}g; {len(entries)} item(s):"
    ]
    for e in entries:
        lines.append(
            f"  - {e.food_name}: {float(e.calories or 0):.0f} kcal "
            f"(P {float(e.protein or 0):.1f}g / C {float(e.carbs or 0):.1f}g / F {float(e.fat or 0):.1f}g) "
            f"at {e.hour:02d}:{e.minute:02d}"
        )
    return "\n".join(lines)


async def _fetch_week_food_log_summary(user_id: int, anchor: date) -> str:
    days = _week_dates_from_anchor(anchor)
    lines = [
        f"Week (Sun–Sat) containing {anchor.isoformat()}: "
        f"{days[0].isoformat()} to {days[-1].isoformat()} — daily totals from food log:"
    ]
    for d in days:
        stmt = (
            select(
                func.sum(FoodCount.calories).label("tc"),
                func.count(FoodCount.id).label("cnt"),
            ).where(
                FoodCount.user_id == user_id,
                FoodCount.year == d.year,
                FoodCount.month == d.month,
                FoodCount.day == d.day,
            )
        )
        res = await db_session.execute(stmt)
        row = res.one()
        tc = float(row.tc or 0)
        cnt = int(row.cnt or 0)
        wk = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[d.weekday()]
        lines.append(f"  {d.isoformat()} ({wk}): {tc:.0f} kcal, {cnt} logged item(s)")
    return "\n".join(lines)

async def _build_food_log_context_for_chat(user_id: int, prompt: str) -> str | None:
    """Returns a plain-text block for the model, or None if nothing to attach."""
    explicit = _extract_explicit_dates_from_text(prompt)
    relative = _extract_relative_day_dates_for_log(prompt)
    week = _wants_week_food_log_summary(prompt)
    dates: list[date] = []
    seen: set[date] = set()
    for d in explicit + relative:
        if d not in seen:
            seen.add(d)
            dates.append(d)
    if not week and not dates:
        return None
    parts: list[str] = []
    if week:
        anchor = explicit[0] if explicit else datetime.now().date()
        parts.append(await _fetch_week_food_log_summary(user_id, anchor))
    else:
        for d in dates:
            parts.append(await _fetch_day_food_log_detail(user_id, d))
    return "\n\n".join(parts) if parts else None


async def _get_owned_conversation(conv_id: int, user_id: int) -> ChatConversation | None:
    stmt = select(ChatConversation).where(
        ChatConversation.id == conv_id,
        ChatConversation.user_id == user_id,
    )
    r = await db_session.execute(stmt)
    return r.scalar_one_or_none()


async def _list_conversations_json(user_id: int, limit: int = 40) -> list[dict]:
    stmt = (
        select(ChatConversation)
        .where(ChatConversation.user_id == user_id)
        .order_by(ChatConversation.updated_at.desc())
        .limit(limit)
    )
    r = await db_session.execute(stmt)
    out = []
    for c in r.scalars().all():
        out.append(
            {
                "id": c.id,
                "title": (c.title or "Chat").strip() or "Chat",
                "updated_at": c.updated_at.isoformat(timespec="seconds")
                if c.updated_at
                else "",
            }
        )
    return out


async def _messages_json_for_ui(conv_id: int) -> list[dict]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conv_id)
        .order_by(ChatMessage.id.asc())
    )
    r = await db_session.execute(stmt)
    return [{"role": m.role, "text": m.body} for m in r.scalars().all()]


async def _messages_for_gemini_history(conv_id: int) -> list[dict]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conv_id)
        .order_by(ChatMessage.id.desc())
        .limit(CHAT_GEMINI_MESSAGE_CAP)
    )
    r = await db_session.execute(stmt)
    rows = list(reversed(r.scalars().all()))
    return [{"role": m.role, "parts": m.body} for m in rows]


async def _resolve_conversation_for_send(user_id: int, body: dict) -> tuple[int, list[dict]]:
    raw_cid = body.get("conversation_id")
    cid = None
    if raw_cid is not None and raw_cid != "":
        try:
            cid = int(raw_cid)
        except (TypeError, ValueError):
            cid = None

    if cid is not None:
        if await _get_owned_conversation(cid, user_id) is None:
            raise ValueError("Invalid conversation_id")
        session["active_chat_conversation_id"] = cid
    else:
        ac = session.get("active_chat_conversation_id")
        ac_int = None
        if ac is not None:
            try:
                ac_int = int(ac)
            except (TypeError, ValueError):
                ac_int = None
            if ac_int is not None and await _get_owned_conversation(ac_int, user_id):
                cid = ac_int
        if cid is None:
            stmt = (
                select(ChatConversation)
                .where(ChatConversation.user_id == user_id)
                .order_by(ChatConversation.updated_at.desc())
                .limit(1)
            )
            r = await db_session.execute(stmt)
            latest = r.scalar_one_or_none()
            if latest is not None:
                cid = latest.id
                session["active_chat_conversation_id"] = cid
            else:
                now = datetime.now()
                conv = ChatConversation(
                    user_id=user_id,
                    title="New chat",
                    created_at=now,
                    updated_at=now,
                )
                db_session.add(conv)
                await db_session.commit()
                await db_session.refresh(conv)
                cid = conv.id
                session["active_chat_conversation_id"] = cid

    hist = await _messages_for_gemini_history(cid)
    return cid, hist


async def _persist_chat_exchange(
    conv_id: int, user_id: int, user_text: str, model_text: str
) -> None:
    conv = await _get_owned_conversation(conv_id, user_id)
    if conv is None:
        return
    cnt_stmt = select(func.count(ChatMessage.id)).where(
        ChatMessage.conversation_id == conv_id
    )
    n_before = (await db_session.execute(cnt_stmt)).scalar() or 0
    now = datetime.now()
    title_update = conv.title
    if n_before == 0 and user_text and user_text.strip():
        title_update = " ".join(user_text.strip().split())[:120]

    db_session.add(
        ChatMessage(
            conversation_id=conv_id,
            role="user",
            body=user_text,
            created_at=now,
        )
    )
    db_session.add(
        ChatMessage(
            conversation_id=conv_id,
            role="model",
            body=model_text,
            created_at=now,
        )
    )
    await db_session.execute(
        update(ChatConversation)
        .where(ChatConversation.id == conv_id)
        .values(title=title_update, updated_at=now)
    )
    await db_session.commit()


@app.route("/api/chat/bootstrap", methods=["GET"])
async def api_chat_bootstrap():
    if "user_id" not in session:
        return jsonify({"persistent": False})
    uid = session["user_id"]
    convs = await _list_conversations_json(uid)
    valid_ids = {c["id"] for c in convs}
    active = session.get("active_chat_conversation_id")
    try:
        active_int = int(active) if active is not None else None
    except (TypeError, ValueError):
        active_int = None
    if active_int is None or active_int not in valid_ids:
        active_int = convs[0]["id"] if convs else None
    if active_int is None:
        session.pop("active_chat_conversation_id", None)
    else:
        session["active_chat_conversation_id"] = active_int
    messages = (
        await _messages_json_for_ui(active_int) if active_int is not None else []
    )
    return jsonify(
        {
            "persistent": True,
            "active_conversation_id": active_int,
            "conversations": convs,
            "messages": messages,
        }
    )


@app.route("/api/chat/conversations", methods=["POST"])
async def api_chat_conversations_create():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    uid = session["user_id"]
    now = datetime.now()
    conv = ChatConversation(
        user_id=uid, title="New chat", created_at=now, updated_at=now
    )
    db_session.add(conv)
    await db_session.commit()
    await db_session.refresh(conv)
    session["active_chat_conversation_id"] = conv.id
    return jsonify({"id": conv.id, "title": conv.title})


@app.route("/api/chat/conversations/<int:cid>/activate", methods=["POST"])
async def api_chat_conversations_activate(cid: int):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    uid = session["user_id"]
    if await _get_owned_conversation(cid, uid) is None:
        return jsonify({"error": "Not found"}), 404
    session["active_chat_conversation_id"] = cid
    messages = await _messages_json_for_ui(cid)
    return jsonify(
        {
            "ok": True,
            "active_conversation_id": cid,
            "messages": messages,
        }
    )

@app.route("/api/chat/conversations/all", methods=["DELETE"])
async def api_chat_conversations_delete_all():
    """Delete every chat conversation (and messages) for the logged-in user."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    uid = session["user_id"]
    conv_ids = select(ChatConversation.id).where(ChatConversation.user_id == uid)
    await db_session.execute(
        delete(ChatMessage).where(ChatMessage.conversation_id.in_(conv_ids))
    )
    await db_session.execute(delete(ChatConversation).where(ChatConversation.user_id == uid))
    await db_session.commit()
    session.pop("active_chat_conversation_id", None)
    return jsonify({"ok": True, "active_conversation_id": None})


@app.route("/api/chat/conversations/<int:cid>", methods=["DELETE"])
async def api_chat_conversations_delete(cid: int):
    """Delete one conversation and all its messages."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    uid = session["user_id"]
    if await _get_owned_conversation(cid, uid) is None:
        return jsonify({"error": "Not found"}), 404

    try:
        ac_raw = session.get("active_chat_conversation_id")
        ac_int = int(ac_raw) if ac_raw is not None else None
    except (TypeError, ValueError):
        ac_int = None

    await db_session.execute(
        delete(ChatMessage).where(ChatMessage.conversation_id == cid)
    )
    await db_session.execute(
        delete(ChatConversation).where(
            ChatConversation.id == cid, ChatConversation.user_id == uid
        )
    )
    await db_session.commit()

    new_active = ac_int
    if ac_int == cid:
        stmt = (
            select(ChatConversation)
            .where(ChatConversation.user_id == uid)
            .order_by(ChatConversation.updated_at.desc())
            .limit(1)
        )
        r = await db_session.execute(stmt)
        nxt = r.scalar_one_or_none()
        if nxt is not None:
            session["active_chat_conversation_id"] = nxt.id
            new_active = nxt.id
        else:
            session.pop("active_chat_conversation_id", None)
            new_active = None

    return jsonify({"ok": True, "active_conversation_id": new_active})


@app.route("/generate", methods=["POST"])
async def generate():
    """Generate a text using the generative AI model"""
    data = await request.get_json()
    if data is None or "prompt" not in data:
        return jsonify({"error": "Invalid input"}), 400

    prompt = data["prompt"]

    uid = session.get("user_id")
    history_for_api = []
    conversation_id_for_response = None

    if uid is not None:
        try:
            cid, hist_rows = await _resolve_conversation_for_send(uid, data)
            conversation_id_for_response = cid
            for m in hist_rows:
                history_for_api.append({"role": m["role"], "parts": [m["parts"]]})
        except ValueError as err:
            return jsonify({"error": str(err)}), 400
    else:
        if "chat_history" not in session:
            session["chat_history"] = []
        for msg in session["chat_history"]:
            history_for_api.append({"role": msg["role"], "parts": [msg["parts"]]})

    food_block = None
    if uid is not None:
        food_block = await _build_food_log_context_for_chat(uid, prompt)
    if food_block:
        message_to_model = (
            "[USER_FOOD_LOG_DATA]\n"
            + food_block
            + "\n\n[USER_MESSAGE]\n"
            + prompt
        )
    else:
        message_to_model = prompt

    try:
        loop = asyncio.get_event_loop()

        async def get_ai_chat_response():
            def _send():
                chat = gemini_chat_model.start_chat(history=history_for_api)
                response = chat.send_message(message_to_model)
                return _gemini_response_text(response)

            return await run_sync_with_timeout(_send)

        response_text = await get_ai_chat_response()
        response_text = response_text or ""
        visible_text, loggable_items = strip_chat_food_log_blocks(response_text)
        if (
            not loggable_items
            and visible_text
            and re.search(r"\d", visible_text)
            and (
                _wants_chat_food_estimate(prompt)
                or _assistant_has_loggable_nutrition_signal(visible_text)
            )
        ):

            def run_fallback():
                return _extract_loggable_items_chat_fallback_sync(prompt, visible_text)

            try:
                fb = await run_sync_with_timeout(run_fallback)
                if fb:
                    loggable_items = fb
            except Exception as ex:
                print(f"Chat food-log fallback error: {ex}")

        if uid is not None:
            await _persist_chat_exchange(
                conversation_id_for_response, uid, prompt, visible_text
            )
        else:
            new_history = session["chat_history"]
            new_history.append({"role": "user", "parts": prompt})
            new_history.append({"role": "model", "parts": visible_text})
            session["chat_history"] = new_history[-10:]

        out = {"text": visible_text}
        if loggable_items:
            out["loggable_items"] = loggable_items
        if conversation_id_for_response is not None:
            out["conversation_id"] = conversation_id_for_response
        return jsonify(out)

    except asyncio.TimeoutError:
        return jsonify(
            {"error": "Chat timed out. Check network or gemini_api_key in .env."}
        ), 504
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze-image", methods=["POST"])
async def analyze_image():
    """Accept an uploaded image and analyze foods with Gemini."""
    files = await request.files
    if 'image' not in files:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    img_file = files['image']

    try:
        img_data = img_file.read()

        contents = [
            "Analyze the foods in this image and respond strictly in the system's 7-line English format per unique food item.",
            {"mime_type": "image/jpeg", "data": img_data}
        ]

        response = await generate_content_with_timeout(
            gemini_food_image_model, contents
        )
        food_name = (response.text or "").strip()

        print(f"--- AI vision result: {food_name} ---")

        items = parse_gemini_food_blocks(food_name)
        if not items and food_name.strip():
            cal_m = re.search(r"Calorie:\s*([\d.]+)\s*kcal", food_name, re.I)
            calories = float(cal_m.group(1)) if cal_m else 0.0
            first_line = next(
                (
                    ln.strip()
                    for ln in food_name.splitlines()
                    if ln.strip()
                    and not re.match(r"^Calorie per 100g:", ln, re.I)
                    and not re.match(r"^Weight:", ln, re.I)
                    and not re.match(r"^Calorie:", ln, re.I)
                    and not re.match(r"^Protein:", ln, re.I)
                    and not re.match(r"^Fat:", ln, re.I)
                    and not re.match(r"^Carbs:", ln, re.I)
                ),
                food_name.strip()[:120],
            )
            items = [
                {
                    "name": first_line[:200],
                    "calories": calories,
                    "protein": 0.0,
                    "fat": 0.0,
                    "carbs": 0.0,
                    "calories_per_100g": None,
                    "weight_g": None,
                }
            ]

        return jsonify({
            "status": "success",
            "food_name": food_name,
            "items": items,
        })

    except asyncio.TimeoutError:
        return jsonify(
            {
                "status": "error",
                "message": "Image analysis timed out. Check network or gemini_api_key.",
            }
        ), 504
    except Exception as e:
        print(f"Image analysis error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # macOS AirPlay uses port 5000; default to 5001 for local dev.
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=app.debug, host="127.0.0.1", port=port)
