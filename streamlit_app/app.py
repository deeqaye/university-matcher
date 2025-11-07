"""Streamlit front-end for the University Matcher dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import pandas as pd
import streamlit as st
from urllib.parse import quote
import re
import base64
import requests
import django

# Ensure the Django project modules are importable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "university_matcher.settings")
django.setup()

from apps.universities.uni_find import (  # type: ignore  # pylint: disable=import-error
    calculate_match_score,
    preprocess_university_data,
)
from apps.universities import views as uni_views  # type: ignore  # pylint: disable=import-error


st.set_page_config(page_title="University Matcher", layout="wide")

STATIC_IMAGE_DIR = BASE_DIR / "static" / "images" / "universities"
PLACEHOLDER_IMAGE_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABAgEAffZciQAAAABJRU5ErkJggg=="
)

st.markdown(
    """
    <style>
    .match-card {background-color:#ffffff;padding:1.75rem;border-radius:20px;box-shadow:0 18px 35px rgba(40,47,60,0.1);margin-bottom:2.25rem;border:1px solid rgba(102,126,234,0.08);}
    .match-card h3 {margin-bottom:0.25rem;font-size:1.5rem;}
    .match-card .match-meta {color:#667eea;font-weight:600;margin-bottom:1rem;}
    .match-card .preference-text {font-size:1rem;line-height:1.65;margin-bottom:1.25rem;color:#2f3b52;}
    .match-card ul {padding-left:1.2rem;margin-bottom:1.1rem;}
    .match-card ul li {margin-bottom:0.35rem;}
    .fact-pill {background:rgba(102,126,234,0.12);padding:6px 12px;border-radius:999px;font-size:0.85rem;margin-right:8px;margin-bottom:8px;display:inline-block;color:#43527c;font-weight:600;}
    .image-caption {font-size:0.85rem;color:#69758c;text-align:center;margin-top:0.5rem;}
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
    sentences: List[str] = []

    institution_type = str(info_row.get("Public/Private", "")).strip() if info_row is not None else ""
    global_rank = str(info_row.get("Global ranking", "")).strip() if info_row is not None else ""
    national_rank = str(info_row.get("National ranking", "")).strip() if info_row is not None else ""
    acceptance_rate = str(info_row.get("Acceptance rate (%)", "")).strip() if info_row is not None else ""
    language = str(info_row.get("Language of teaching", "")).strip() if info_row is not None else ""
    scholarships = str(info_row.get("Scholarship (yes/no)", "")).strip() if info_row is not None else ""
    student_ratio = str(info_row.get("Student to faculty ratio", "")).strip() if info_row is not None else ""
    international_cost = str(info_row.get("Cost per year \nfor interational students", "")).strip() if info_row is not None else ""

    intro_clause = "a leading institution"
    if institution_type:
        intro_clause = f"a {institution_type.lower()} institution"
    sentences.append(f"{university} is {intro_clause} based in {country}, welcoming an international cohort of ambitious students each year.")

    if global_rank and global_rank.upper() != "N/A":
        ranking_sentence = f"It is recognised globally with a ranking of {global_rank}"
        if national_rank and national_rank.upper() != "N/A":
            ranking_sentence += f" and stands at {national_rank} within national league tables"
        ranking_sentence += "."
        sentences.append(ranking_sentence)

    if acceptance_rate and acceptance_rate.upper() != "N/A":
        sentences.append(f"Recent admissions data points to an acceptance rate near {acceptance_rate}, signalling an academically driven student body.")

    if language and language.upper() != "N/A":
        sentences.append(f"Teaching is primarily delivered in {language}, supported by well-resourced faculties and modern learning environments.")

    if scholarships and scholarships.upper() != "N/A":
        sentences.append("Multiple scholarship routes are available, giving international students additional financial flexibility.")

    if student_ratio and student_ratio.upper() != "N/A":
        sentences.append(f"A student-to-faculty ratio of {student_ratio} keeps seminars collaborative and mentoring accessible.")

    intl_cost_value = international_cost if international_cost and international_cost.upper() != "N/A" else stats.get("international_cost_max")
    if intl_cost_value not in (None, "", "N/A"):
        sentences.append(f"International tuition averages around {format_currency(intl_cost_value)}, helping you plan the annual investment upfront.")

    sentences.append("Beyond academics, the campus blends research hubs, cultural societies, and tailored support services that help newcomers settle quickly.")

    return " ".join(sentences)


def build_preference_explanation(
    university: str,
    country: str,
    row: pd.Series,
    info_row: Optional[pd.Series],
    user_input: Dict[str, Any],
    stats: Dict[str, Any],
) -> str:
    sentences: List[str] = []

    selected_countries = user_input.get("countries", [])
    if selected_countries:
        if country in selected_countries:
            if len(selected_countries) == 1:
                sentences.append(f"You asked for universities in {selected_countries[0]}, and {university} keeps you right in {country}.")
            else:
                chosen = ", ".join(selected_countries)
                sentences.append(f"{university} sits in {country}, one of your preferred study destinations ({chosen}).")
        else:
            sentences.append(f"{university} is located in {country}, giving you an additional option beyond your initial list ({', '.join(selected_countries)}).")

    language_levels = {}
    for idx, lang in enumerate(user_input.get("languages", [])):
        if idx < len(user_input.get("lang_levels", [])):
            language_levels[lang.lower()] = user_input["lang_levels"][idx]

    teaching_languages_source = info_row.get("Language of teaching", "") if info_row is not None else row.get("language", "")
    teaching_languages = [lang.strip().lower() for lang in str(teaching_languages_source).split(",") if lang.strip()]

    # Language fit
    if teaching_languages:
        shown_language = None
        for lang, level in language_levels.items():
            if lang in teaching_languages:
                if lang == "english":
                    required_ielts = to_float(stats.get("IELTS_min"))
                    if required_ielts is not None and user_input.get("ielts") is not None:
                        user_ielts = user_input.get("ielts")
                        if user_ielts >= required_ielts:
                            sentences.append(
                                f"Your IELTS {user_ielts:.1f} comfortably meets the {required_ielts:.1f} English requirement for this campus."
                            )
                            shown_language = "english"
                            break
                else:
                    sentences.append(
                        f"You indicated {lang.title()} at level {level}, which matches the teaching language options offered here."
                    )
                    shown_language = lang
                    break
        if shown_language is None and "english" in teaching_languages:
            sentences.append("This university teaches in English, so IELTS will be the key requirement when you are ready to apply.")

    # Academic profile fit
    gpa_required = to_float(stats.get("GPA_min"))
    if gpa_required is not None:
        user_gpa = user_input.get("gpa")
        if user_gpa is not None and user_gpa >= gpa_required:
            sentences.append(f"Your GPA of {user_gpa:.2f} is above the expected {gpa_required:.2f} threshold.")

    sat_required = to_float(stats.get("SAT_min"))
    if sat_required:
        user_sat = user_input.get("sat")
        if user_sat and user_sat >= sat_required:
            sentences.append(f"A SAT score of {user_sat} exceeds the stated benchmark of {int(sat_required)}.")

    tuition_required = to_float(stats.get("international_cost_max"))
    if tuition_required is None and info_row is not None:
        tuition_required = to_float(info_row.get("Cost per year \nfor interational students"))
    if tuition_required is not None:
        budget = user_input.get("budget_max")
        if budget:
            if budget >= tuition_required:
                sentences.append(f"The annual cost (~{format_currency(tuition_required)}) stays within your budget of {format_currency(budget)}.")
            else:
                sentences.append(
                    f"Keep in mind the annual cost (~{format_currency(tuition_required)}) is above your stated budget of {format_currency(budget)}."
                )

    public_pref = user_input.get("public_preference")
    if public_pref in (0, 1):
        desired_type = "public" if public_pref == 1 else "private"
        if info_row is not None:
            actual_type = str(info_row.get("Public/Private", "")).strip().lower()
        else:
            public_flag = row.get("is_public")
            actual_type = "public" if public_flag == 1 else "private" if public_flag == 0 else ""
        if actual_type:
            if desired_type in actual_type:
                sentences.append(f"You preferred {desired_type} institutions, and this university fits that preference.")
            else:
                sentences.append(f"This institution is {actual_type}, which differs from your stated preference for {desired_type} options.")

    scholarship_info = str(info_row.get("Scholarship (yes/no)", "")).strip().lower() if info_row is not None else ""
    if scholarship_info == "yes":
        sentences.append("Scholarship opportunities are available, giving you more flexibility to manage costs.")

    if not sentences:
        sentences.append(f"{university} aligns well with your academic profile and study preferences.")

    return " ".join(sentences)


def render_result_card(index: int, record: Dict[str, Any], info_row: Optional[pd.Series], user_input: Dict[str, Any], llm_meta: Optional[Dict[str, Any]]) -> None:
    university = record["university"]
    country = record["country"]
    match_score = int(record.get("match_score", 0))

    stats = {
        "GPA_min": record.get("GPA_min"),
        "SAT_min": record.get("SAT_min"),
        "IELTS_min": record.get("IELTS_min"),
        "international_cost_max": record.get("international_cost_max"),
    }

    preference_text = (llm_meta or {}).get("preference_explanation") or build_preference_explanation(
        university, country, pd.Series(record), info_row, user_input, stats
    )

    short_description = ""
    if getattr(uni_views, "GEMINI_ENABLED", False):
        try:
            short_description = uni_views.get_short_university_description(university, country, info_row)
        except Exception:
            short_description = ""

    if not short_description:
        short_description = build_description(university, country, stats, info_row)

    if getattr(uni_views, "GEMINI_ENABLED", False) and getattr(uni_views, "ENABLE_LLM_ENRICHMENT", True):
        try:
            long_description = uni_views.get_ai_description(university, country, stats, info_row)
        except Exception:
            long_description = build_description(university, country, stats, info_row)
    else:
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

    fact_items = [(label, value) for label, value in fact_items if value not in (None, "", "N/A")]

    website = ""
    if info_row is not None:
        website = str(info_row.get("Link to official website", "")).strip()

    with st.container(border=True):
        st.markdown(f"### #{index} {university}")
        st.markdown(f"**Match score:** {match_score} / 100")

        col_text, col_image = st.columns([1.8, 1], gap="large")

        with col_text:
            st.write(preference_text)
            st.markdown(f"_Summary_: {short_description}")

            if fact_items:
                st.markdown("**Key facts**")
                for label, value in fact_items:
                    st.markdown(f"• **{label}:** {value}")

            with st.expander("Comprehensive overview"):
                st.write(long_description)

            if website:
                st.markdown(f"[Visit official site ↗]({website})")

        with col_image:
            st.image(image_bytes, caption=f"{university} • {country}", use_column_width=True)


def main() -> None:
    st.title("University Matcher – Streamlit Edition")
    st.write("Discover universities that align with your background, proficiency, and budget in just a few clicks.")

    datasets = load_datasets()
    info_df = load_info_dataframe()

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

    user_input = {
        "countries": selected_countries,
        "languages": selected_languages,
        "lang_levels": language_levels,
        "ielts": float(ielts_score),
        "gpa": float(gpa_score),
        "sat": int(sat_score),
        "budget_max": float(budget),
        "public_preference": public_preference,
    }

    matches_df = calculate_match_score(processed_df.copy(), user_input)

    if matches_df.empty:
        st.error("No universities matched your criteria. Try adjusting your filters or widening your preferences.")
        return

    st.success(f"Found {len(matches_df)} matching universities. Showing top {top_n} results.")

    matches_records = matches_df.to_dict("records")
    selected_records = matches_records[:top_n]
    llm_meta_map: Dict[str, Dict[str, Any]] = {}

    if user_preferences and getattr(uni_views, "GEMINI_ENABLED", False):
        try:
            llm_selected = uni_views.select_top_universities_with_llm(matches_records, user_preferences, user_input)
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
        render_result_card(idx, record, info_row, user_input, llm_meta_map.get(record["university"]))
        st.divider()


if __name__ == "__main__":
    main()

