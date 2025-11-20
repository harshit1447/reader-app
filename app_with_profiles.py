"""
Streamlit Article Reader — Login page + profiles + persistent reading tracker

Behavior:
- If not signed in: show full-screen auth page (Login / Sign up)
- On successful login/signup: set session_state.user_id and go to main app
- Main app contains article analysis, timer, and per-user dashboard (sessions stored in SQLite)
"""

import streamlit as st
import sqlite3
import os
import time
from io import BytesIO
import datetime as dt
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import docx
from passlib.hash import pbkdf2_sha256

# Ensure NLTK data
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

DB_FILE = "reader_app.db"

# -------------------- DB helpers and schema --------------------
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # Users table: include display_name and email (nullable)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        display_name TEXT,
        email TEXT,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reading_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        seconds INTEGER NOT NULL,
        source TEXT,
        words_count INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- Auth functions --------------------
def create_user(username: str, password: str, display_name: str = None, email: str = None):
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password are required."
    conn = get_conn()
    cur = conn.cursor()
    try:
        password_hash = pbkdf2_sha256.hash(password)
        cur.execute(
            "INSERT INTO users (username, password_hash, display_name, email, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, display_name, email, dt.datetime.utcnow().isoformat())
        )
        conn.commit()
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_user(username: str, password: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        if pbkdf2_sha256.verify(password, row["password_hash"]):
            return int(row["id"])
    except Exception:
        return None
    return None

def get_profile(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, display_name, email, created_at FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)

# -------------------- Sessions storage --------------------
def store_session(user_id: int, seconds: int, source: str = None, words_count: int = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO reading_sessions (user_id, seconds, source, words_count, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, int(seconds), source, words_count, dt.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def fetch_user_sessions(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM reading_sessions WHERE user_id = ? ORDER BY created_at DESC", conn, params=(user_id,))
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["id","user_id","seconds","source","words_count","created_at"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

# -------------------- Article extraction & analysis --------------------
def extract_text_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","noscript","header","footer","svg"]):
            tag.extract()
        article_text = []
        article = soup.find("article")
        if article:
            article_text.extend([p.get_text(strip=True) for p in article.find_all("p")])
        if not article_text:
            main = soup.find("main")
            if main:
                article_text.extend([p.get_text(strip=True) for p in main.find_all("p")])
        if not article_text:
            article_text.extend([p.get_text(strip=True) for p in soup.find_all("p")])
        text = "\n".join([p for p in article_text if p])
        return text if text.strip() else ""
    except Exception:
        return ""

def load_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception:
        return ""

def load_text_from_docx(file_bytes: bytes) -> str:
    try:
        with BytesIO(file_bytes) as bio:
            doc = docx.Document(bio)
            paras = [p.text for p in doc.paragraphs if p.text]
            return "\n".join(paras)
    except Exception:
        return ""

def safe_load_uploaded_file(uploaded) -> str:
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        return load_text_from_pdf(data)
    elif name.endswith(".docx"):
        return load_text_from_docx(data)
    else:
        try:
            return data.decode("utf-8")
        except Exception:
            try:
                return data.decode("latin-1")
            except Exception:
                return ""

def word_and_char_counts(text: str):
    tokens = word_tokenize(text)
    words_filtered = [t for t in tokens if any(c.isalnum() for c in t)]
    return len(words_filtered), len(text), words_filtered

def estimate_reading_seconds(words_count: int, wpm: int) -> float:
    if wpm <= 0:
        return 0.0
    return (words_count / float(wpm)) * 60.0

def pretty_time_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m} min {s} sec" if m else f"{s} sec"

def sentence_clusters(text: str, n_clusters: int = 3):
    sentences = sent_tokenize(text)
    if not sentences:
        return {}, {}
    n_clusters = max(1, min(n_clusters, len(sentences)))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(sentences)
    if len(sentences) == 1:
        return {0: [(0, sentences[0])]}, {0: []}
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    clusters = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[int(lab)].append((i, sentences[i]))
    terms = np.array(vectorizer.get_feature_names_out())
    centers = kmeans.cluster_centers_
    keywords = {}
    for i, center in enumerate(centers):
        top_idx = center.argsort()[-8:][::-1]
        keywords[int(i)] = terms[top_idx].tolist()
    return dict(clusters), keywords

# -------------------- Streamlit UI & routing --------------------
st.set_page_config(page_title="Reader — Login & Tracking", layout="wide")

# session flags
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = "auth"  # "auth" or "app"

# --- AUTH PAGE (full screen) ---
def auth_page():
    st.markdown("<h1 style='text-align:center'>Welcome to Reader</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 0.6, 1])
    with col2:
        st.subheader("Login or create an account")
        mode = st.radio("Select", ["Login", "Sign up"], horizontal=True)

        if mode == "Sign up":
            st.write("Create a new account")
            su_display = st.text_input("Display name (optional)", key="su_disp")
            su_user = st.text_input("Username", key="su_user")
            su_email = st.text_input("Email (optional)", key="su_email")
            su_pass = st.text_input("Password", type="password", key="su_pass")
            if st.button("Create account"):
                ok, msg = create_user(su_user, su_pass, display_name=su_display or None, email=su_email or None)
                if ok:
                    uid = verify_user(su_user, su_pass)
                    st.session_state.user_id = int(uid)
                    st.session_state.page = "app"
                    st.success("Account created and signed in — welcome!")
                    st.experimental_rerun()
                else:
                    st.error(msg)

        else:
            st.write("Sign in to your account")
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Log in"):
                uid = verify_user(li_user, li_pass)
                if uid:
                    st.session_state.user_id = int(uid)
                    st.session_state.page = "app"
                    st.success("Signed in")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")

        st.markdown("---")
        st.write("Or continue as guest (history won't be saved).")
        if st.button("Continue as guest"):
            st.session_state.user_id = None
            st.session_state.page = "app"
            st.experimental_rerun()

# --- MAIN APP PAGE ---
def app_page():
    # header & profile
    profile = get_profile(st.session_state.user_id) if st.session_state.user_id else None
    cols = st.columns([1, 2, 1])
    with cols[0]:
        if profile:
            st.markdown(f"**Signed in as:** {profile.get('display_name') or profile.get('username')}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.page = "auth"
                st.experimental_rerun()
        else:
            st.markdown("**Guest**")
            if st.button("Back to Login"):
                st.session_state.page = "auth"
                st.experimental_rerun()

    # main app UI (left sidebar controls)
    with st.sidebar:
        st.header("Input")
        input_mode = st.radio("Provide article", ["Paste text", "Upload file", "Website URL"])
        if input_mode == "Paste text":
            text = st.text_area("Paste your article here", height=300)
            src_label = "paste"
        elif input_mode == "Upload file":
            uploaded = st.file_uploader("Upload .txt / .pdf / .docx", type=["txt","pdf","docx"])
            text = safe_load_uploaded_file(uploaded) if uploaded else ""
            src_label = getattr(uploaded, "name", "upload")
        else:
            url = st.text_input("Enter website URL")
            if st.button("Fetch article from URL"):
                text = extract_text_from_url(url.strip()) if url.strip() else ""
                src_label = url.strip()
            else:
                text = ""
                src_label = None

        st.markdown("---")
        st.header("Reading speed")
        wpm_choice = st.selectbox("WPM preset", ["200 (average)", "150 (slow)", "250 (fast)", "Custom"])
        if wpm_choice == "200 (average)":
            wpm_val = 200
        elif wpm_choice == "150 (slow)":
            wpm_val = 150
        elif wpm_choice == "250 (fast)":
            wpm_val = 250
        else:
            wpm_val = st.number_input("Custom WPM", min_value=50, max_value=1000, value=200, step=10)

        st.markdown("---")
        st.header("Clustering")
        n_clusters = st.slider("Number of sentence clusters", 1, 8, 3)

        st.markdown("---")
        st.header("Tracker / session")
        st.write("Start / Stop the timer — sessions are stored when stopped (if signed in).")
        if "timer_running" not in st.session_state:
            st.session_state.timer_running = False
            st.session_state.timer_start = None

        if st.button("Start timer"):
            st.session_state.timer_running = True
            st.session_state.timer_start = time.time()
            st.success("Timer started")

        if st.button("Stop timer"):
            if st.session_state.timer_running and st.session_state.timer_start:
                elapsed = int(round(time.time() - st.session_state.timer_start))
                st.session_state.timer_running = False
                st.session_state.timer_start = None
                st.success(f"Stopped — recorded {elapsed} seconds")
                if st.session_state.user_id:
                    wc = None
                    if text:
                        wc, _, _ = word_and_char_counts(text)
                    store_session(st.session_state.user_id, elapsed, src_label, wc)
                else:
                    st.info("Guest session — not saved. Create an account to persist history.")
            else:
                st.info("Timer wasn't running.")

        add_min = st.number_input("Add minutes manually", min_value=0, max_value=1000, value=0, step=1)
        if st.button("Add minutes"):
            if add_min > 0:
                seconds = int(add_min * 60)
                if st.session_state.user_id:
                    wc = None
                    if text:
                        wc, _, _ = word_and_char_counts(text)
                    store_session(st.session_state.user_id, seconds, src_label, wc)
                    st.success(f"Added {add_min} minutes to your history.")
                else:
                    st.info("Guest session — not saved. Create an account to persist history.")

    # Validate text input
    if not text or not text.strip():
        st.info("Paste text, upload a file, or fetch a URL from the left panel to analyze an article.")
        return

    # Analysis & dashboard UI
    words_count, char_count, words_list = word_and_char_counts(text)
    estimated_seconds_total = estimate_reading_seconds(words_count, wpm_val)

    st.subheader("Summary")
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("Words", f"{words_count:,}")
    c2.metric("Characters", f"{char_count:,}")
    c3.metric("Est. read time", pretty_time_seconds(estimated_seconds_total))

    st.markdown("---")
    clusters, keywords = sentence_clusters(text, n_clusters=n_clusters)
    st.subheader("Clusters")
    if not clusters:
        st.info("No sentences found for clustering.")
    else:
        ordered = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        for lab, sents in ordered:
            st.markdown(f"**Cluster {lab} — {len(sents)} sentences**")
            kw = keywords.get(lab, [])
            if kw:
                st.caption("Keywords: " + ", ".join(kw[:8]))
            cluster_text = " ".join([s for _, s in sents])
            wc, _, _ = word_and_char_counts(cluster_text)
            secs = estimate_reading_seconds(wc, wpm_val)
            st.write(f"Cluster read time: {pretty_time_seconds(secs)} — {wc} words")
            for idx, sent in sents[:6]:
                st.write(f"- {sent}")
            st.markdown("")

    st.markdown("---")
    st.subheader("Per-sentence (first 50)")
    sentences = sent_tokenize(text)
    rows = []
    for i, s in enumerate(sentences[:50]):
        wc = len([t for t in word_tokenize(s) if any(c.isalnum() for c in t)])
        secs = estimate_reading_seconds(wc, wpm_val)
        rows.append((i+1, wc, pretty_time_seconds(secs), s))
    if rows:
        df_rows = pd.DataFrame(rows, columns=["#", "Words", "Est time", "Sentence"])
        st.dataframe(df_rows, use_container_width=True)

    # Dashboard (only for signed in users)
    st.markdown("---")
    st.header("Your Reading Dashboard")
    if not st.session_state.user_id:
        st.info("Create an account or log in to see your persistent dashboard.")
        return

    user_id = st.session_state.user_id
    sessions_df = fetch_user_sessions(user_id)
    if sessions_df.empty:
        st.info("No reading history yet — start and stop the timer to record sessions.")
        return

    sessions_df["date"] = sessions_df["created_at"].dt.date
    today = dt.date.today()
    seconds_today = int(sessions_df[sessions_df["date"] == today]["seconds"].sum())
    seconds_7d = int(sessions_df[sessions_df["created_at"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=7))]["seconds"].sum())
    seconds_month = int(sessions_df[sessions_df["created_at"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=30))]["seconds"].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Today", pretty_time_seconds(seconds_today))
    m2.metric("Last 7 days", pretty_time_seconds(seconds_7d))
    m3.metric("Last 30 days", pretty_time_seconds(seconds_month))

    # Time series last 90 days
    last_n = 90
    start_date = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=last_n)
    series = sessions_df.set_index("created_at").resample("D")["seconds"].sum().reindex(pd.date_range(start=start_date, end=pd.Timestamp.utcnow().normalize(), freq="D"), fill_value=0)
    series.index = pd.to_datetime(series.index).date
    st.subheader("Daily reading (last 90 days)")
    st.line_chart(series)

    monthly = sessions_df.set_index("created_at").resample("M")["seconds"].sum().sort_index()
    if not monthly.empty:
        st.subheader("Monthly reading")
        monthly_df = (monthly/60).rename("minutes").to_frame()
        st.bar_chart(monthly_df)

    st.subheader("Sessions")
    display_df = sessions_df.copy()
    display_df["human_time"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["minutes"] = (display_df["seconds"]/60).round(2)
    st.dataframe(display_df[["human_time","minutes","source","words_count"]].rename(columns={"human_time":"when","minutes":"minutes","source":"source","words_count":"words"}), use_container_width=True)

    csv = display_df.to_csv(index=False)
    st.download_button("Export history as CSV", data=csv, file_name=f"{get_profile(user_id)['username']}_history.csv", mime="text/csv")

# Router: show auth or app page
if st.session_state.page == "auth" or st.session_state.user_id is None:
    auth_page()
else:
    app_page()
