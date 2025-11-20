        # Reader App - Streamlit

        This is a single-file Streamlit app that allows users to analyze articles, track reading time, and save per-user reading sessions in a local SQLite database.


        ## Quick start (local)

1. Create a Python 3.11 venv and activate it:

```powershell
py -3.11 -m venv "%USERPROFILE%\streamenv311"
& "%USERPROFILE%\streamenv311\Scripts\Activate.ps1"
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Run the app:

```powershell
python -m streamlit run app_with_profiles.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On Streamlit Cloud create a new app pointing to this repo and `app_with_profiles.py`.
3. Add any secrets via the app settings if needed.

Note: For production persistence, use a managed Postgres and set `DATABASE_URL` in Streamlit Cloud secrets.
