# Streamlit Deployment

This folder contains a lightweight Streamlit interface for the University Matcher project. It reuses the existing matching logic from `apps/universities/uni_find.py` and presents the results in a single-page app that can run locally or on Streamlit Community Cloud.

## Local preview

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

The UI exposes the same filters as the Django app (countries, languages, proficiency, IELTS, GPA, SAT, and budget) and displays a ranked list of matching universities together with quick descriptions and supporting data.

## Streamlit Community Cloud deployment

1. Push the repository to GitHub if you have not already.
2. Log into [share.streamlit.io](https://share.streamlit.io) (Streamlit Community Cloud) and click **Deploy an app**.
3. Point to your repository and select `streamlit_app/app.py` as the entry point.
4. Under **Advanced Settings → Python packages**, paste the contents of `streamlit_app/requirements.txt` (or upload the file).
5. Set the working directory to `streamlit_app` so relative paths resolve correctly. Streamlit will install dependencies and start the app.

### Environment variables

The Streamlit build reads `data.csv` and (optionally) `info.csv` from the project root. No additional environment variables are required. If `info.csv` is not present on the server the app will still operate, but extended university details will be limited.

## Customisation tips

- Adjust the number of results via the “How many matches should we show?” slider in the sidebar.
- To add Gemini-generated writeups within Streamlit, wire your API calls into `build_description()`; the current implementation stays purely CSV-based for faster rendering and to avoid quota issues.
- The helper functions (`build_description`, `find_info_row`, etc.) can be moved into a shared module if you plan to reuse them elsewhere.

