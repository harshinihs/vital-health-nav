from pathlib import Path
from datetime import datetime
from typing import Literal
import hashlib
import math

import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Data-Driven Hospital Recommendation API"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "HOSPITAL_DATASET_DSP.csv"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# =========================================================
# REQUIRED FILES / FOLDERS
# =========================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

if not TEMPLATES_DIR.exists():
    raise FileNotFoundError(
        f"Templates folder not found: {TEMPLATES_DIR}"
    )

if not STATIC_DIR.exists():
    raise FileNotFoundError(
        f"Static folder not found: {STATIC_DIR}"
    )


# =========================================================
# STATIC FILES + TEMPLATES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

templates = Jinja2Templates(
    directory=TEMPLATES_DIR
)


# =========================================================
# LOAD HOSPITAL DATASET
# =========================================================

hospitals = pd.read_csv(DATA_FILE)


REQUIRED_COLUMNS = {
    "hospital_id",
    "hospital_name",
    "district",
    "ownership",
    "verified_specialties",
    "latitude",
    "longitude",
}

missing_columns = REQUIRED_COLUMNS - set(hospitals.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


# =========================================================
# CLEAN DATA
# =========================================================

TEXT_COLUMNS = [
    "hospital_id",
    "hospital_name",
    "district",
    "ownership",
    "verified_specialties",
]

for column in TEXT_COLUMNS:
    hospitals[column] = (
        hospitals[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

hospitals["latitude"] = pd.to_numeric(
    hospitals["latitude"],
    errors="coerce"
)

hospitals["longitude"] = pd.to_numeric(
    hospitals["longitude"],
    errors="coerce"
)


TOTAL_HOSPITALS = len(hospitals)

HOSPITALS_WITH_COORDINATES = int(
    hospitals[["latitude", "longitude"]]
    .notna()
    .all(axis=1)
    .sum()
)

print(
    f"Loaded {TOTAL_HOSPITALS} hospitals "
    f"from {DATA_FILE.name}"
)

print(
    f"Hospitals with coordinates: "
    f"{HOSPITALS_WITH_COORDINATES}"
)


# =========================================================
# MEDICAL CATEGORY -> REQUIRED SPECIALIZATION
# =========================================================

CATEGORY_SPECIALTY = {
    "heart": "Cardiology",
    "joint": "Orthopedics",
    "skin": "Dermatology",
    "brain": "Neurology",
    "child": "Pediatrics",
    "womens": "Gynecology",
    "lung": "Pulmonology",
    "kidney": "Nephrology",
    "urinary": "Urology",
    "eye": "Ophthalmology",
    "ent": "ENT",
    "dental": "Dental",
    "cancer": "Oncology",
    "stomach": "Gastrology",
    "mental": "Psychology",
    "fertility": "Fertility",
    "general": "General Medicine",
}


# =========================================================
# REQUEST MODEL
# =========================================================

class RecommendationRequest(BaseModel):
    category: str = Field(..., min_length=1)

    latitude: float = Field(
        ...,
        ge=-90,
        le=90
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180
    )

    hospital_preference: Literal[
        "Government",
        "Private",
        "Any"
    ] = "Any"

    location_source: Literal[
        "device",
        "map"
    ] = "device"


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    language: Literal["en", "ta"] = "en"
    username: str = Field(default="User", max_length=50)


# =========================================================
# DISTANCE
# =========================================================

def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:

    radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return radius_km * c


# =========================================================
# OWNERSHIP
# =========================================================

def ownership_type(ownership: str) -> str:

    value = ownership.strip().lower()

    if "govt" in value or "government" in value:
        return "Government"

    return "Private"


# =========================================================
# SPECIALTY NORMALIZATION
# =========================================================

def normalize_specialty(value: str) -> str:

    return (
        value
        .strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
        .replace("-", " ")
        .replace("&", "and")
        .replace("specialities", "specialty")
    )


MULTI_SPECIALTY_VALUES = {
    "multi speciality",
    "multi specialty",
    "multispeciality",
    "multispecialty",
}

GENERAL_MEDICINE_VALUES = {
    "general medicine",
    "general",
}


# =========================================================
# SPECIALTY TESTS
# =========================================================

def is_multi_speciality(
    verified_specialties: str
) -> bool:

    specialties = [
        normalize_specialty(item)
        for item in verified_specialties.split(",")
        if item.strip()
    ]

    for item in specialties:
        if item in MULTI_SPECIALTY_VALUES:
            return True

        if (
            "multi specialty" in item
            or "multi speciality" in item
            or "multispecialty" in item
            or "multispeciality" in item
        ):
            return True

    return False


def has_general_medicine(
    verified_specialties: str
) -> bool:

    specialties = [
        normalize_specialty(item)
        for item in verified_specialties.split(",")
        if item.strip()
    ]

    return any(
        item in GENERAL_MEDICINE_VALUES
        for item in specialties
    )


def has_specific_specialty(
    verified_specialties: str,
    required_specialty: str
) -> bool:

    specialties = [
        normalize_specialty(item)
        for item in verified_specialties.split(",")
        if item.strip()
    ]

    required = normalize_specialty(
        required_specialty
    )

    return required in specialties


# =========================================================
# RECOMMENDATION TIER
# =========================================================
# For a specific problem:
#   Tier 1 -> Exact specialty + Multi-Speciality
#   Tier 2 -> General Medicine
#
# For General Medicine:
#   Tier 1 -> General Medicine
#   Tier 2 -> Multi-Speciality
# =========================================================

def get_recommendation_tier(
    verified_specialties: str,
    required_specialty: str
) -> int | None:

    required = normalize_specialty(
        required_specialty
    )

    # General Medicine request
    if required == "general medicine":

        if has_general_medicine(
            verified_specialties
        ):
            return 1

        if is_multi_speciality(
            verified_specialties
        ):
            return 2

        return None

    # 1. Exact specialized hospital
    if has_specific_specialty(
        verified_specialties,
        required_specialty
    ):
        return 1

    # 1. Multi-Speciality
    if is_multi_speciality(
        verified_specialties
    ):
        return 1

    # 2. General Medicine
    if has_general_medicine(
        verified_specialties
    ):
        return 2

    return None


# =========================================================
# CROWD MODEL
# =========================================================
# The original crowd model uses:
#   Time + Specialization + Ownership
#
# Those three values alone can produce identical scores for
# hospitals having the same attributes at the same time.
#
# To make hospital cards meaningfully different, this version
# adds a stable hospital-specific estimated baseline.
#
# Modified prototype formula:
#
#   Crowd Score =
#       0.50(Time)
#     + 0.15(Specialization)
#     + 0.15(Ownership)
#     + 0.20(Hospital Baseline)
#
# All components are 0-10.
#
# IMPORTANT:
# Hospital Baseline is a deterministic simulated/estimated
# factor, NOT measured real-time patient occupancy.
# =========================================================

def get_time_score(
    now: datetime
) -> tuple[str, float]:

    hour = now.hour

    if 6 <= hour <= 11:
        # Morning = highest priority
        return "Morning", 10.0

    if 12 <= hour <= 17:
        # Afternoon = medium priority
        return "Afternoon", 5.0

    if 18 <= hour <= 23:
        # Night = high priority
        return "Night", 10.0

    # Midnight = lowest priority
    return "Midnight", 0.0


def get_specialization_score(
    verified_specialties: str
) -> float:

    if is_multi_speciality(
        verified_specialties
    ):
        return 10.0

    if has_general_medicine(
        verified_specialties
    ):
        return 6.67

    return 3.33


def get_hospital_baseline_score(
    hospital_id: str
) -> float:
    """
    Create a stable 0-10 prototype baseline from the hospital ID.

    This is deterministic: the same hospital always gets the
    same baseline. It is not real hospital crowd data.
    """

    digest = hashlib.sha256(
        hospital_id.encode("utf-8")
    ).hexdigest()

    number = int(
        digest[:12],
        16
    )

    return round(
        (number % 10001) / 1000.0,
        2
    )


def calculate_crowd_score(
    hospital_id: str,
    verified_specialties: str,
    ownership: str,
    now: datetime | None = None
) -> tuple[float, str, str, float]:

    now = now or datetime.now()

    time_period, time_score = get_time_score(now)

    specialization_score = (
        get_specialization_score(
            verified_specialties
        )
    )

    ownership_score = (
        10.0
        if ownership_type(ownership) == "Government"
        else 5.0
    )

    hospital_baseline = (
        get_hospital_baseline_score(
            hospital_id
        )
    )

    score = (
        0.50 * time_score
        + 0.15 * specialization_score
        + 0.15 * ownership_score
        + 0.20 * hospital_baseline
    )

    score = round(
        max(0.0, min(10.0, score)),
        2
    )

    # User-defined crowd levels:
    # 0-3       -> Low
    # 3.01-8.99 -> Moderate
    # 9-10      -> High
    if score <= 3:
        level = "Low"
    elif score < 9:
        level = "Moderate"
    else:
        level = "High"

    return (
        score,
        level,
        time_period,
        hospital_baseline
    )




# =========================================================
# VH NAV ASSISTANT CHAT
# =========================================================


def assistant_reply(message: str, language: str, username: str) -> str:

    text = message.strip().lower()
    safe_name = username.strip() or "User"

    if language == "ta":
        if any(word in text for word in ["வணக்கம்", "hello", "hi", "ஹாய்"]):
            return (
                f"வணக்கம் {safe_name}! நான் VH நெவ் உதவியாளர். "
                "உங்கள் மருத்துவமனை தேடலில் நான் வழிகாட்டுகிறேன். "
                "கீழே உள்ள உடல்நலப் பிரிவுகளில் ஒன்றைத் தேர்ந்தெடுக்கலாம்."
            )

        if any(word in text for word in ["எப்படி", "how", "வேலை", "work", "செயல்பட"]):
            return (
                "VH NAV மூன்று முக்கிய தகவல்களைப் பயன்படுத்துகிறது: உடல்நலப் பிரிவு, "
                "உங்கள் இருப்பிடம் மற்றும் மருத்துவமனை வகை. முதலில் உடல்நலப் பிரிவைத் தேர்ந்தெடுத்து, "
                "பின்னர் GPS அல்லது வரைபட இருப்பிடத்தைத் தேர்ந்தெடுத்து, இறுதியில் மருத்துவமனை வகையைத் தேர்வு செய்யுங்கள்."
            )

        if any(word in text for word in ["location", "இருப்பிடம்", "gps", "வரைபடம்", "map"]):
            return (
                "இருப்பிடத்திற்கு இரண்டு வழிகள் உள்ளன: சாதனத்தின் தற்போதைய GPS அல்லது வரைபடத்தில் ஒரு பினை தேர்வு செய்தல். "
                "GPS பயன்படுத்தினால் உலாவி இருப்பிட அனுமதியை கேட்கலாம்."
            )

        if any(word in text for word in ["government", "private", "அரசு", "தனியார்"]):
            return (
                "மருத்துவமனை வகையில் அரசு, தனியார் அல்லது ஏதேனும் ஒன்றைத் தேர்ந்தெடுக்கலாம். "
                "இந்த விருப்பம் முடிவுகளில் உள்ள மருத்துவமனைகளை வடிகட்டும்."
            )

        if any(word in text for word in ["crowd", "கூட்ட", "நெரிசல்"]):
            return (
                "கூட்ட நெரிசல் மதிப்பு தற்போதைய நோயாளர் எண்ணிக்கை அல்ல. "
                "இந்த திட்டத்தில் இது ஒரு prototype estimate ஆக காட்டப்படுகிறது."
            )

        if any(word in text for word in ["help", "உதவி", "என்ன செய்ய", "what can"]):
            return (
                "நான் VH NAV பயன்பாட்டை பயன்படுத்துவது, இருப்பிடத்தைத் தேர்வு செய்வது, "
                "மருத்துவமனை வகையைப் புரிந்துகொள்வது மற்றும் பரிந்துரை செயல்முறை பற்றி வழிகாட்ட முடியும்."
            )

        return (
            f"{safe_name}, உங்கள் கேள்வியைப் புரிந்துகொள்ள முயற்சிக்கிறேன். "
            "மருத்துவமனை பரிந்துரைக்க, கீழே உள்ள உடல்நலப் பிரிவுகளில் பொருத்தமான ஒன்றைத் தேர்வு செய்யுங்கள். "
            "இந்த chat தற்போது வழிகாட்டுதலுக்காக பயன்படுத்தப்படுகிறது; இது நோயறிதல் செய்யாது."
        )

    if any(word in text for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        return (
            f"Hello {safe_name}! I’m VH NAV ASSISTANT. "
            "I can guide you through the hospital search. Choose a health category below, "
            "then select your location and hospital type."
        )

    if any(phrase in text for phrase in ["how does", "how it works", "how does this work", "how to use", "what do i do"]):
        return (
            "VH NAV works in three main steps: choose the health category, provide your location "
            "using GPS or the map, and choose Government, Private or Any. Then the system finds "
            "suitable hospitals using specialty, distance and the project’s estimated crowd score."
        )

    if any(word in text for word in ["location", "gps", "map", "where"]):
        return (
            "You can use your device’s current GPS location or choose a point on the map. "
            "The map option lets you drag the pin before confirming the location."
        )

    if any(word in text for word in ["government", "private", "hospital type"]):
        return (
            "You can filter results to Government hospitals, Private hospitals, or Any. "
            "Choose the option that matches your preference before getting recommendations."
        )

    if any(word in text for word in ["crowd", "busy", "estimated crowd"]):
        return (
            "The crowd value shown by VH NAV is a project prototype estimate. "
            "It is not a live patient count or live traffic-aware hospital occupancy value."
        )

    if any(word in text for word in ["specialty", "specialisation", "specialization", "which category", "problem"]):
        return (
            "For hospital matching, choose the closest health category shown in Step 1. "
            "The system then maps that category to the required medical specialty and filters hospitals accordingly."
        )

    if any(word in text for word in ["help", "what can you do", "what can you help"]):
        return (
            "I can explain how VH NAV works, help you understand the location and hospital-type choices, "
            "and guide you through the recommendation process. I do not diagnose medical conditions."
        )

    return (
        f"I’m here to help, {safe_name}. Ask me about the hospital search, location selection, "
        "hospital type, specialties, or the estimated crowd score. For hospital matching, please "
        "choose a category from Step 1."
    )


@app.post("/assistant")
def assistant_chat(request: AssistantRequest):

    return {
        "reply": assistant_reply(
            request.message,
            request.language,
            request.username,
        )
    }


# =========================================================
# ROOT PAGE
# =========================================================

@app.get("/")
def login_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={}
    )


@app.get("/app")
def app_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

@app.post("/recommend")
def recommend(
    request: RecommendationRequest
):

    category = (
        request.category
        .strip()
        .lower()
    )

    if category not in CATEGORY_SPECIALTY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown category: "
                f"{request.category}"
            )
        )

    required_specialty = (
        CATEGORY_SPECIALTY[category]
    )

    # =====================================================
    # 1. OWNERSHIP FILTER
    # =====================================================

    filtered = hospitals.copy()

    if request.hospital_preference != "Any":

        target_type = request.hospital_preference

        filtered = filtered[
            filtered["ownership"].apply(
                ownership_type
            ) == target_type
        ].copy()

    # =====================================================
    # 2. RECOMMENDATION TIER
    # =====================================================

    filtered["recommendation_tier"] = (
        filtered["verified_specialties"].apply(
            lambda value:
                get_recommendation_tier(
                    value,
                    required_specialty
                )
        )
    )

    filtered = filtered[
        filtered["recommendation_tier"].notna()
    ].copy()

    if filtered.empty:
        return {
            "category": category,
            "required_specialty": required_specialty,
            "location_source": request.location_source,
            "user_location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
            },
            "results": [],
            "message": (
                "No suitable hospitals found "
                "for the selected criteria."
            ),
        }

    # =====================================================
    # 3. VALID COORDINATES
    # =====================================================

    filtered = filtered.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    if filtered.empty:
        return {
            "category": category,
            "required_specialty": required_specialty,
            "location_source": request.location_source,
            "user_location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
            },
            "results": [],
            "message": (
                "No suitable hospitals with valid "
                "coordinates were found."
            ),
        }

    # =====================================================
    # 4. DISTANCE
    # =====================================================

    now = datetime.now()

    filtered["distance_km"] = filtered.apply(
        lambda row:
            haversine_distance(
                request.latitude,
                request.longitude,
                float(row["latitude"]),
                float(row["longitude"]),
            ),
        axis=1,
    )

    # =====================================================
    # 5. CROWD SCORE
    # =====================================================

    crowd_results = filtered.apply(
        lambda row:
            calculate_crowd_score(
                row["hospital_id"],
                row["verified_specialties"],
                row["ownership"],
                now,
            ),
        axis=1,
    )

    filtered["crowd_score"] = crowd_results.apply(
        lambda item: item[0]
    )

    filtered["crowd_level"] = crowd_results.apply(
        lambda item: item[1]
    )

    filtered["time_period"] = crowd_results.apply(
        lambda item: item[2]
    )

    filtered["hospital_baseline"] = crowd_results.apply(
        lambda item: item[3]
    )

    # =====================================================
    # 6. FINAL SORT
    # =====================================================
    # Tier 1 always stays above Tier 2.
    # Inside a tier:
    #   nearest hospital first
    #   then lower crowd score
    # =====================================================

    filtered = filtered.sort_values(
        by=[
            "recommendation_tier",
            "distance_km",
            "crowd_score"
        ],
        ascending=[
            True,
            True,
            True
        ],
    )

    # =====================================================
    # 7. TOP 10
    # =====================================================

    top_results = filtered.head(10)

    # =====================================================
    # 8. RESPONSE
    # =====================================================

    results = []

    for _, row in top_results.iterrows():

        tier = int(
            row["recommendation_tier"]
        )

        if category == "general":
            if tier == 1:
                recommendation_group = (
                    "General Medicine"
                )
            else:
                recommendation_group = (
                    "Multi-Speciality"
                )
        else:
            if tier == 1:
                if has_specific_specialty(
                    row["verified_specialties"],
                    required_specialty
                ):
                    recommendation_group = (
                        "Specialized"
                    )
                else:
                    recommendation_group = (
                        "Multi-Speciality"
                    )
            else:
                recommendation_group = (
                    "General Medicine"
                )

        results.append({
            "hospital_id": row["hospital_id"],
            "hospital_name": row["hospital_name"],
            "district": row["district"],
            "ownership": row["ownership"],
            "hospital_ownership_type": ownership_type(
                row["ownership"]
            ),
            "verified_specialties": row[
                "verified_specialties"
            ],
            "recommendation_tier": tier,
            "recommendation_group": recommendation_group,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "distance_km": round(
                float(row["distance_km"]),
                2
            ),
            "crowd_score": float(
                row["crowd_score"]
            ),
            "crowd_level": row["crowd_level"],
            "time_period": row["time_period"],
            "hospital_baseline": float(
                row["hospital_baseline"]
            ),
        })

    return {
        "category": category,
        "required_specialty": required_specialty,
        "location_source": request.location_source,
        "user_location": {
            "latitude": request.latitude,
            "longitude": request.longitude,
        },
        "results": results,
    }
