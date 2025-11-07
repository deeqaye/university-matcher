"""Streamlit front-end for the University Matcher dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from urllib.parse import quote

# Ensure the Django project modules are importable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from apps.universities.uni_find import (  # type: ignore  # pylint: disable=import-error
    calculate_match_score,
    preprocess_university_data,
)


st.set_page_config(page_title="University Matcher", layout="wide")

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


def build_description(university: str, country: str, stats: Dict[str, str], info_row: Optional[pd.Series]) -> str:
    parts: List[str] = []

    if info_row is not None:
        institution_type = str(info_row.get("Public/Private", "")).strip()
        if institution_type:
            parts.append(f"{university} is a {institution_type.lower()} institution located in {country}.")
        else:
            parts.append(f"{university} is located in {country} and welcomes international students.")

        global_rank = str(info_row.get("Global ranking", "")).strip()
        if global_rank and global_rank.upper() != "N/A":
            parts.append(f"It currently holds a global ranking of {global_rank}.")

        acceptance_rate = str(info_row.get("Acceptance rate (%)", "")).strip()
        if acceptance_rate and acceptance_rate.upper() != "N/A":
            parts.append(f"Recent acceptance rates hover around {acceptance_rate}.")

        language = str(info_row.get("Language of teaching", "")).strip()
        if language and language.upper() != "N/A":
            parts.append(f"Courses are primarily taught in {language}.")

        scholarships = str(info_row.get("Scholarship (yes/no)", "")).strip()
        if scholarships and scholarships.upper() != "N/A":
            parts.append(f"Scholarships are {scholarships} for qualified applicants.")

        cost_international = str(info_row.get("Cost per year \nfor interational students", "")).strip()
        if cost_international and cost_international.upper() != "N/A":
            parts.append(f"Estimated annual cost for international students is {cost_international}.")
    else:
        parts.append(f"{university} is located in {country} and provides a welcoming environment for global learners.")

    admission_bits: List[str] = []
    gpa_min = stats.get("GPA_min")
    sat_min = stats.get("SAT_min")
    ielts_min = stats.get("IELTS_min")

    if gpa_min not in (None, "N/A", 0):
        admission_bits.append(f"GPA {gpa_min}+")
    if sat_min not in (None, "N/A", 0):
        admission_bits.append(f"SAT {sat_min}+")
    if ielts_min not in (None, "N/A", 0):
        admission_bits.append(f"IELTS {ielts_min}+")

    if admission_bits:
        parts.append("Admissions typically expect " + ", ".join(admission_bits) + ".")

    return " ".join(parts)


def get_university_image(university: str, country: str) -> str:
    query = quote(f"{university} {country} campus")
    return f"https://source.unsplash.com/800x600/?{query}"


def render_result_card(index: int, row: pd.Series, info_row: Optional[pd.Series]) -> None:
    university = row["university"]
    country = row["country"]

    stats = {
        "GPA_min": row.get("GPA_min"),
        "SAT_min": row.get("SAT_min"),
        "IELTS_min": row.get("IELTS_min"),
        "international_cost_max": row.get("international_cost_max"),
    }

    description = build_description(university, country, stats, info_row)
    image_url = get_university_image(university, country)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"#{index} {university}")
        st.markdown(f"**Match score:** {row.get('match_score', 0)} / 100")
        st.markdown(description)

        detail_lines = []
        if info_row is not None:
            detail_lines.extend(
                [
                    ("Acceptance rate", info_row.get("Acceptance rate (%)", "N/A")),
                    ("Global ranking", info_row.get("Global ranking", "N/A")),
                    ("National ranking", info_row.get("National ranking", "N/A")),
                    ("Student-faculty ratio", info_row.get("Student to faculty ratio", "N/A")),
                    ("Scholarships", info_row.get("Scholarship (yes/no)", "N/A")),
                ]
            )

        detail_lines.extend(
            [
                ("Teaching languages", row.get("language", "N/A")),
                ("Minimum GPA", row.get("GPA_min", "N/A")),
                ("Minimum IELTS", row.get("IELTS_min", "N/A")),
                ("Minimum SAT", row.get("SAT_min", "N/A")),
                ("Annual tuition (international)", row.get("international_cost_max", "N/A")),
            ]
        )

        for label, value in detail_lines:
            st.markdown(f"- **{label}:** {value if value not in (None, '') else 'N/A'}")

        if info_row is not None:
            website = info_row.get("Link to official website", "").strip()
            if website:
                st.markdown(f"[Official website]({website})")

    with col2:
        st.image(image_url, caption=f"{university} campus", use_column_width=True)


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

    for idx, (_, row) in enumerate(matches_df.head(top_n).iterrows(), start=1):
        info_row = find_info_row(info_df, row["university"], row["country"])
        render_result_card(idx, row, info_row)
        st.divider()


if __name__ == "__main__":
    main()

