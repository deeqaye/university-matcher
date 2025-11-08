# pyright: reportMissingImports=false
"""Streamlit front-end for the University Matcher dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import importlib.util
import pandas as pd
import streamlit as st  # type: ignore
from urllib.parse import quote
import re
import base64
from html import escape
import requests
import django
from functools import lru_cache
from types import ModuleType
from django.apps import apps as django_apps

# Ensure the Django project modules are importable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

STREAMLIT_ENABLE_DJANGO = (
    os.getenv("STREAMLIT_ENABLE_DJANGO", "false").strip().lower() in {"1", "true", "yes"}
)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "university_matcher.settings")


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


try:
    uni_find_module = load_module(
        "apps.universities.uni_find", BASE_DIR / "apps" / "universities" / "uni_find.py"
    )
except Exception as exc:  # pragma: no cover - diagnostics only
    raise RuntimeError(f"Unable to load matching utilities: {exc}") from exc

calculate_match_score = uni_find_module.calculate_match_score
preprocess_university_data = uni_find_module.preprocess_university_data


@lru_cache(maxsize=1)
def load_django_views_module() -> ModuleType:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "university_matcher.settings")
    if not django_apps.ready:
        django.setup()
    try:
        from apps.universities import views as views_module  # type: ignore
    except Exception:
        views_module = load_module(
            "apps.universities.views", BASE_DIR / "apps" / "universities" / "views.py"
        )
    return views_module


st.set_page_config(page_title="University Matcher", layout="wide")

STATIC_IMAGE_DIR = BASE_DIR / "static" / "images" / "universities"
PLACEHOLDER_IMAGE_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABAgEAffZciQAAAABJRU5ErkJggg=="
)

st.markdown(
    """
    <style>
    body, .match-card, .body-text {font-family:'Inter', 'Segoe UI', sans-serif !important;font-size:1rem;line-height:1.7;color:#263044;}
    .match-card {background-color:#ffffff;padding:1.8rem;border-radius:20px;box-shadow:0 18px 35px rgba(40,47,60,0.12);margin-bottom:2.6rem;border:1px solid rgba(102,126,234,0.08);}
    .match-card h3 {margin-bottom:0.4rem;font-size:1.55rem;color:#1d2540;}
    .match-card .match-meta {color:#5b6dee;font-weight:600;margin-bottom:1.1rem;}
    .fact-pill {background:rgba(102,126,234,0.12);padding:6px 12px;border-radius:999px;font-size:0.85rem;margin-right:8px;margin-bottom:8px;display:inline-block;color:#43527c;font-weight:600;}
    .body-text {margin-bottom:1rem;}
    .body-text strong {color:#1d2540;}
    .image-caption {font-size:0.85rem;color:#69758c;text-align:center;margin-top:0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = BASE_DIR / "data.csv"
INFO_PATH_CANDIDATES = [
    BASE_DIR / "info.csv",
    BASE_DIR.parent / "info.csv",
    BASE_DIR.parent.parent / "info.csv",
]
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


@st.cache_data(show_spinner=False)
def load_datasets() -> Dict[str, pd.DataFrame]:
    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_university_data(raw_df.copy())
    return {"raw": raw_df, "processed": processed_df}


@st.cache_data(show_spinner=False)
def load_info_dataframe() -> Optional[pd.DataFrame]:
    for candidate in INFO_PATH_CANDIDATES:
        if candidate.exists():
            try:
                info_df = pd.read_csv(candidate)
            except Exception as exc:  # pragma: no cover - diagnostics only
                st.warning(f"Unable to read info.csv from {candidate}: {exc}")
                continue
            info_df["__normalized_name"] = info_df["University Name"].fillna("").str.lower().str.strip()
            info_df["__normalized_name"] = info_df["__normalized_name"].str.replace("(ucl)", "", regex=False)
            info_df["__normalized_name"] = info_df["__normalized_name"].str.replace("(uol)", "", regex=False)
            info_df["__normalized_name"] = info_df["__normalized_name"].str.replace("  ", " ", regex=False)
            info_df["__normalized_country"] = info_df["Country"].fillna("").str.lower().str.strip()
            info_df["__normalized_country"] = info_df["__normalized_country"].str.replace(r"\s*\(.*\)", "", regex=True)
            return info_df
    return None


def find_info_row(info_df: Optional[pd.DataFrame], university: str, country: str) -> Optional[pd.Series]:
    if info_df is None or info_df.empty:
        return None

    uni_name = university.lower().strip()
    uni_name = uni_name.replace("(ucl)", "").replace("(uol)", "").strip()
    country_norm = country.lower().strip()

    matches = info_df[
        info_df["__normalized_name"].str.contains(uni_name, case=False, na=False)
        & info_df["__normalized_country"].str.contains(country_norm, case=False, na=False)
    ]

    if matches.empty:
        matches = info_df[info_df["__normalized_name"].str.contains(uni_name, case=False, na=False)]

    if matches.empty:
        return None

    return matches.iloc[0]


def to_float(value) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", value_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def format_currency(value) -> str:
    amount = to_float(value)
    if amount is None:
        return str(value) if value not in (None, "") else "N/A"
    return f"${amount:,.0f}"


def paragraph_html(text: str) -> str:
    return escape(text).replace("\n", "<br>")


@st.cache_data(show_spinner=False)
def get_university_image_bytes(university: str, country: str) -> bytes:
    safe_name = re.sub(r"[^\w\s-]", "", university.lower())
    safe_name = re.sub(r"[-\s]+", "-", safe_name)

    for extension in (".jpg", ".jpeg", ".png"):
        static_path = STATIC_IMAGE_DIR / f"{safe_name}{extension}"
        if static_path.exists():
            try:
                return static_path.read_bytes()
            except Exception:
                continue

    query = quote(f"{university} {country} campus")
    unsplash_url = f"https://source.unsplash.com/900x600/?{query}"
    try:
        response = requests.get(unsplash_url, timeout=10)
        if response.status_code == 200 and response.content:
            return response.content
    except Exception:
        pass

    return PLACEHOLDER_IMAGE_BYTES


def build_description(university: str, country: str, stats: Dict[str, Any], info_row: Optional[pd.Series]) -> str:
    institution_type = "public" if stats.get("is_public") == 1 else "private"
    tuition = stats.get("international_cost_max")
    tuition_text = f" around {format_currency(tuition)}" if tuition not in (None, "", "N/A") else ""

    return (
        f"{university} is a {institution_type} option in {country}. "
        "It emphasises an English-speaking academic environment and welcomes international applications. "
        "Plan for tuition" + tuition_text + "."
    )


def build_preference_explanation(
    university: str,
    country: str,
    row: pd.Series,
    info_row: Optional[pd.Series],
    user_input: Dict[str, Any],
    stats: Dict[str, Any],
) -> str:
    user_note = user_input.get("preferences_text", "").strip()
    if user_note:
        return user_note

    lines: List[str] = []

    selected_countries = [c for c in user_input.get("countries", []) if c]
    if selected_countries:
        selected = ", ".join(selected_countries)
        if country in selected_countries:
            lines.append(f"You asked for options in {selected}, and this university keeps you in {country}.")
        else:
            lines.append(f"This adds {country} alongside your preferred countries ({selected}).")

    languages = user_input.get("languages", [])
    levels = user_input.get("lang_levels", [])
    if languages:
        formatted_langs = []
        for idx, lang in enumerate(languages):
            level = levels[idx] if idx < len(levels) else None
            formatted_langs.append(f"{lang} ({level})" if level else lang)
        lines.append(f"You told us you can study in {', '.join(formatted_langs)}.")

    user_gpa = user_input.get("gpa")
    if user_gpa:
        lines.append(f"A GPA of {user_gpa:.2f} shows strong academic preparation.")

    user_sat = user_input.get("sat")
    if user_sat:
        lines.append(f"Your SAT score of {user_sat} demonstrates solid test readiness.")

    user_ielts = user_input.get("ielts")
    if user_ielts:
        lines.append(f"An IELTS score of {user_ielts:.1f} confirms you can learn comfortably in English.")

    budget = user_input.get("budget_max")
    if budget:
        lines.append(f"You've planned for an annual budget of about {format_currency(budget)} to support your studies.")

    public_pref = user_input.get("public_preference")
    if public_pref == 1:
        lines.append("You mentioned a preference for public institutions.")
    elif public_pref == 0:
        lines.append("You mentioned a preference for private institutions.")

    if not lines:
        lines.append(f"{university} aligns with the academic profile you shared.")

    return " ".join(lines)


def render_result_card(
    index: int,
    record: Dict[str, Any],
    info_row: Optional[pd.Series],
    user_input: Dict[str, Any],
    llm_meta: Optional[Dict[str, Any]],
    uni_views: Optional[ModuleType],
) -> None:
    university = record["university"]
    country = record["country"]
    match_score = int(record.get("match_score", 0))

    stats = {
        "GPA_min": record.get("GPA_min"),
        "SAT_min": record.get("SAT_min"),
        "IELTS_min": record.get("IELTS_min"),
        "international_cost_max": record.get("international_cost_max"),
        "is_public": record.get("is_public"),
    }

    preference_text: Optional[str] = None
    if uni_views is not None and hasattr(uni_views, "generate_preference_paragraph"):
        try:
            preference_text = uni_views.generate_preference_paragraph(university, country, user_input, user_input.get("preferences_text"))
        except Exception as exc:
            st.warning(f"Preference paragraph failed for {university}: {exc}")

    if not preference_text:
        preference_text = build_preference_explanation(
            university, country, pd.Series(record), info_row, user_input, stats
        )

    short_description = ""
    if uni_views is not None and getattr(uni_views, "GEMINI_ENABLED", False):
        try:
            short_description = uni_views.get_short_university_description(university, country, info_row)
        except Exception:
            short_description = ""

    if not short_description:
        short_description = build_description(university, country, stats, info_row)

    long_description = build_description(university, country, stats, info_row)

    image_bytes = get_university_image_bytes(university, country)

    fact_items: List[tuple[str, str]] = []
    if info_row is not None:
        fact_items.extend(
            [
                ("Acceptance rate", str(info_row.get("Acceptance rate (%)", "N/A"))),
                ("Global ranking", str(info_row.get("Global ranking", "N/A"))),
                ("National ranking", str(info_row.get("National ranking", "N/A"))),
                ("Student-faculty ratio", str(info_row.get("Student to faculty ratio", "N/A"))),
                ("Scholarships", str(info_row.get("Scholarship (yes/no)", "N/A"))),
            ]
        )

    fact_items.extend(
        [
            ("Teaching languages", str(info_row.get("Language of teaching", "N/A")) if info_row is not None else str(record.get("language", "N/A"))),
            ("Minimum GPA", record.get("GPA_min", "N/A")),
            ("Minimum IELTS", record.get("IELTS_min", "N/A")),
            ("Minimum SAT", record.get("SAT_min", "N/A")),
            ("Annual tuition (international)", format_currency(record.get("international_cost_max"))),
        ]
    )

    fact_items = [
        (label, value)
        for label, value in fact_items
        if value not in (None, "", "N/A") and str(value).lower() not in {"nan", "none"}
    ]

    website = ""
    if info_row is not None:
        website = str(info_row.get("Link to official website", "")).strip()

    with st.container(border=True):
        st.markdown(f"### #{index} {university}")
        st.markdown(f"**Match score:** {match_score} / 100")

        col_text, col_image = st.columns([1.8, 1], gap="large")

        with col_text:
            st.markdown(f"<p class='body-text'>{paragraph_html(preference_text)}</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p class='body-text'><strong>Snapshot:</strong> {paragraph_html(short_description)}</p>",
                unsafe_allow_html=True,
            )

            if fact_items:
                facts_html = "<ul class='body-text'>" + "".join(
                    f"<li><strong>{escape(label)}:</strong> {escape(str(value))}</li>" for label, value in fact_items
                ) + "</ul>"
                st.markdown(facts_html, unsafe_allow_html=True)

            with st.expander("Comprehensive overview"):
                st.markdown(f"<p class='body-text'>{paragraph_html(long_description)}</p>", unsafe_allow_html=True)

            if website:
                st.markdown(f"[Visit official site ↗]({website})")

        with col_image:
            st.image(image_bytes, caption=f"{university} • {country}", use_column_width=True)


def main() -> None:
    st.title("University Matcher – Streamlit Edition")
    st.write("Discover universities that align with your background, proficiency, and budget in just a few clicks.")

    try:
        datasets = load_datasets()
    except FileNotFoundError:
        st.error(
            "The core dataset `data.csv` is missing. Please ensure it is present "
            "at `streamlit_app/data.csv` (or alongside `manage.py`) and redeploy."
        )
        st.stop()
    except Exception as exc:  # pragma: no cover - diagnostics only
        st.error(f"Unable to load the dataset: {exc}")
        st.stop()

    try:
        info_df = load_info_dataframe()
    except Exception as exc:  # pragma: no cover - diagnostics only
        st.warning(f"Failed to load `info.csv`: {exc}")
        info_df = None

    raw_df = datasets["raw"]
    processed_df = datasets["processed"]

    with st.sidebar:
        st.header("Your Profile")

        country_options = sorted(raw_df["country"].unique().tolist())
        language_options = sorted({lang.strip() for langs in raw_df["language"].dropna() for lang in str(langs).split(",")})

        selected_countries = st.multiselect("Preferred countries", country_options, default=country_options[:1])
        selected_languages = st.multiselect("Languages you can study in", language_options, default=["English"])

        language_levels: List[str] = []
        for lang in selected_languages:
            if lang.lower() == "english":
                level = st.selectbox(
                    "English proficiency (CEFR equivalent)",
                    CEFR_LEVELS,
                    index=4,
                    key=f"level_{lang}",
                )
            else:
                level = st.selectbox(
                    f"{lang} proficiency",
                    CEFR_LEVELS,
                    index=3,
                    key=f"level_{lang}",
                )
            language_levels.append(level)

        ielts_score = st.number_input("IELTS score", min_value=0.0, max_value=9.0, value=6.5, step=0.5)
        gpa_score = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.2, step=0.1)
        sat_score = st.number_input("SAT score", min_value=0, max_value=1600, value=1200, step=10)
        budget = st.number_input("Maximum annual budget (USD)", min_value=0, value=50000, step=1000)

        public_preference_label = st.selectbox("Institution type preference", ["No preference", "Public", "Private"])
        public_preference = {"No preference": -1, "Public": 1, "Private": 0}[public_preference_label]

        user_preferences = st.text_area(
            "Describe your ideal university experience (optional)",
            placeholder="Share preferences like campus vibe, research focus, city type, extracurriculars...",
            height=120,
        )

        top_n = st.slider("How many matches should we show?", min_value=1, max_value=10, value=5)
        submitted = st.button("Find universities", use_container_width=True)

    if not submitted:
        st.info("Configure your preferences in the sidebar and click 'Find universities' to begin.")
        return

    if not selected_countries:
        st.warning("Please select at least one preferred country.")
        return

    if not selected_languages:
        st.warning("Please select at least one language option.")
        return

    preferences_prompt = user_preferences.strip()

    user_input = {
        "countries": selected_countries,
        "languages": selected_languages,
        "lang_levels": language_levels,
        "ielts": float(ielts_score),
        "gpa": float(gpa_score),
        "sat": int(sat_score),
        "budget_max": float(budget),
        "public_preference": public_preference,
        "preferences_text": preferences_prompt,
    }

    matches_df = calculate_match_score(processed_df.copy(), user_input)

    if matches_df.empty:
        st.error("No universities matched your criteria. Try adjusting your filters or widening your preferences.")
        return

    st.success(f"Found {len(matches_df)} matching universities. Showing top {top_n} results.")

    matches_records = matches_df.to_dict("records")
    selected_records = matches_records[:top_n]
    llm_meta_map: Dict[str, Dict[str, Any]] = {}

    uni_views_module: Optional[ModuleType] = None
    django_error: Optional[str] = None
    if STREAMLIT_ENABLE_DJANGO:
        try:
            uni_views_module = load_django_views_module()
        except Exception as exc:  # pragma: no cover - diagnostics only
            django_error = str(exc)
            st.warning(
                "Django-powered enrichments are disabled for this session. "
                "The app will continue with built-in descriptions."
            )

    if preferences_prompt and uni_views_module is not None and getattr(uni_views_module, "GEMINI_ENABLED", False):
        try:
            llm_selected = uni_views_module.select_top_universities_with_llm(matches_records, preferences_prompt, user_input)
            if llm_selected:
                llm_meta_map = {item["university"]: item for item in llm_selected}
                ordered_names = [item["university"] for item in llm_selected]
                selected_records = [rec for rec in matches_records if rec["university"] in ordered_names]
                selected_records = sorted(selected_records, key=lambda rec: ordered_names.index(rec["university"]))
                selected_records = selected_records[:top_n]
        except Exception as exc:
            st.warning(f"LLM prioritisation failed: {exc}")

    for idx, record in enumerate(selected_records, start=1):
        info_row = find_info_row(info_df, record["university"], record["country"])
        render_result_card(idx, record, info_row, user_input, llm_meta_map.get(record["university"]), uni_views_module)
        st.divider()


if __name__ == "__main__":
    main()

