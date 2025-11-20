"""
Streamlit Article Reader — Profiles + Persistent Reading Tracker (SQLite)

Features:
- Signup / Login (username + password hashed)
- Analyze articles (paste / upload / url)
- Start/Stop reading timer (stores sessions in SQLite)
- Manual add minutes (stores session)
- Dashboard with daily/monthly totals and charts
"""

import streamlit as st
import sqlite3
import os
import time
import json
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

# Ensure required NLTK data is present (download quietly if missing)
for pkg in ("punkt","punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# Database file (SQLite) - stored locally in app folder
DB_FILE = "reader_app.db"

# -------------------- Database helpers --------------------
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(\"\"\"
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    \"\"\")
    cur.execute(\"\"\"
    CREATE TABLE IF NOT EXISTS reading_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        seconds INTEGER NOT NULL,
        source TEXT,
        words_count INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    \"\"\")
    conn.commit()
    conn.close()

init_db()

# -------------------- Auth helpers --------------------
def create_user(username: str, password: str) -> (bool, str):
    username = username.strip()
    if not username or not password:
        return False, "Username and password required."
    conn = get_conn()
    cur = conn.cursor()
    try:
        hash_ = pbkdf2_sha256.hash(password)
        cur.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, hash_, dt.datetime.utcnow().isoformat()))
        conn.commit()
        return True, "User created."
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
    user_id = row["id"]
    hash_ = row["password_hash"]
    if pbkdf2_sha256.verify(password, hash_):
        return user_id
    return None

# -------------------- Reading session storage --------------------
def store_session(user_id: int, seconds: int, source: str = None, words_count: int = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO reading_sessions (user_id, seconds, source, words_count, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, int(seconds), source, words_count, dt.datetime.utcnow().isoformat()))
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

# -------------------- Article extraction + analysis --------------------
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
        text = "\\n".join([p for p in article_text if p])
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
        return "\\n".join(pages)
    except Exception:
        return ""

def load_text_from_docx(file_bytes: bytes) -> str:
    try:
        with BytesIO(file_bytes) as bio:
            doc = docx.Document(bio)
            paras = [p.text for p in doc.paragraphs if p.text]
            return "\\n".join(paras)
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
    if wpm<=0:
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
    if len(sentences)==1:
        return {0:[(0,sentences[0])]}, {0:[]}
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

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Reader — profiles & tracking", layout="wide")
st.title("Reader — profiles & persistent tracking")

# --- Authentication area
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

auth_mode = st.sidebar.selectbox("Account", ["Login","Sign up","Continue as Guest"])

if auth_mode == "Sign up":
    st.sidebar.header("Create account")
    su_user = st.sidebar.text_input("Choose username")
    su_pass = st.sidebar.text_input("Choose password", type="password")
    if st.sidebar.button("Create account"):
        ok, msg = create_user(su_user, su_pass)
        if ok:
            st.sidebar.success("Account created — please switch to Login to sign in.")
        else:
            st.sidebar.error(msg)
    st.sidebar.write("---")
elif auth_mode == "Login":
    st.sidebar.header("Log in")
    li_user = st.sidebar.text_input("Username")
    li_pass = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Log in"):
        uid = verify_user(li_user, li_pass)
        if uid:
            st.session_state.user_id = int(uid)
            st.session_state.username = li_user
            st.sidebar.success(f"Signed in as {li_user}")
        else:
            st.sidebar.error("Invalid username or password")
    if st.sidebar.button("Log out"):
        st.session_state.user_id = None
        st.session_state.username = None
        st.sidebar.info("Logged out")
elif auth_mode == "Continue as Guest":
    st.sidebar.info("You will not have persistent tracking as a guest. Sign up to save your history.")

# -------------------- Main app inputs --------------------
with st.sidebar:
    st.markdown("---")
    st.header("Input")
    input_mode = st.radio("Provide article", ["Paste text","Upload file","Website URL"])
    if input_mode == "Paste text":
        text = st.text_area("Paste your article here", height=250)
        source_label = "paste"
    elif input_mode == "Upload file":
        uploaded = st.file_uploader("Upload .txt / .pdf / .docx", type=["txt","pdf","docx"])
        text = safe_load_uploaded_file(uploaded) if uploaded else ""
        source_label = getattr(uploaded, "name", "upload")
    else:
        url = st.text_input("Enter website URL")
        if st.button("Fetch article from URL"):
            text = extract_text_from_url(url.strip()) if url.strip() else ""
            source_label = url.strip()
        else:
            text = ""
            source_label = None

    st.markdown("---")
    st.header("Reading speed")
    wpm = st.selectbox("WPM", ["150 (slow)","200 (average)","250 (fast)","Custom"])
    if isinstance(wpm, str):
        if wpm.startswith("150"):
            wpm_val = 150
        elif wpm.startswith("200"):
            wpm_val = 200
        elif wpm.startswith("250"):
            wpm_val = 250
        else:
            wpm_val = st.number_input("Custom WPM", min_value=50, max_value=1000, value=200, step=10)
    else:
        wpm_val = 200

    st.markdown("---")
    st.header("Clustering")
    n_clusters = st.slider("Number of clusters", 1, 8, 3)

    st.markdown("---")
    st.header("Tracker / session")
    st.write("Start/Stop timer to record a reading session (stored when stopped).")
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
            # store session if logged in
            if st.session_state.user_id:
                # compute words count if text is loaded
                wc = None
                if text:
                    wc, _, _ = word_and_char_counts(text)
                store_session(st.session_state.user_id, elapsed, source_label, wc)
            else:
                st.info("Not logged in — session not saved. Sign up / Login to persist history.")
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
                store_session(st.session_state.user_id, seconds, source_label, wc)
                st.success(f"Added {add_min} minutes to your stored history.")
            else:
                st.info("Not logged in — session not saved. Sign up / Login to persist history.")

# Validate text
if not text or not text.strip():
    st.info("Paste / upload / fetch a URL to analyze an article.")
    st.stop()

# -------------------- Analysis (same as before) --------------------
words_count, char_count, words_list = word_and_char_counts(text)
estimated_seconds_total = estimate_reading_seconds(words_count, wpm_val)

st.header("Analysis")
col1, col2, col3 = st.columns(3)
col1.metric("Words", f"{words_count:,}")
col2.metric("Characters", f"{char_count:,}")
col3.metric("Est. read time", pretty_time_seconds(estimated_seconds_total))

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

# -------------------- User dashboard --------------------
st.markdown("---")
st.header("Your Reading Dashboard")
if not st.session_state.user_id:
    st.info("Login to see your persistent reading stats. You can still track manually as guest but it won't be saved.")
    st.stop()

user_id = st.session_state.user_id
sessions_df = fetch_user_sessions(user_id)

if sessions_df.empty:
    st.info("No reading history yet — start reading and stop the timer to record sessions.")
    st.stop()

# aggregation
sessions_df["date"] = sessions_df["created_at"].dt.date
# totals
today = dt.date.today()
seconds_today = int(sessions_df[sessions_df["date"] == today]["seconds"].sum())
seconds_7d = int(sessions_df[sessions_df["created_at"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=7))]["seconds"].sum())
seconds_month = int(sessions_df[sessions_df["created_at"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=30))]["seconds"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Today", pretty_time_seconds(seconds_today))
c2.metric("Last 7 days", pretty_time_seconds(seconds_7d))
c3.metric("Last 30 days", pretty_time_seconds(seconds_month))

# Daily time series (last 90 days)
last_n = 90
start_date = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=last_n)
series = sessions_df.set_index("created_at").resample("D")["seconds"].sum().reindex(pd.date_range(start=start_date, end=pd.Timestamp.utcnow().normalize(), freq="D"), fill_value=0)
series.index = pd.to_datetime(series.index).date
st.subheader("Daily reading (last 90 days)")
st.line_chart(series)

# Monthly aggregation
monthly = sessions_df.set_index("created_at").resample("M")["seconds"].sum()
monthly = monthly.sort_index()
if not monthly.empty:
    st.subheader("Monthly reading")
    monthly_df = (monthly/60).rename("minutes").to_frame()
    st.bar_chart(monthly_df)

# Session table & export
st.subheader("Sessions")
display_df = sessions_df.copy()
display_df["human_time"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
display_df["minutes"] = (display_df["seconds"]/60).round(2)
st.dataframe(display_df[["human_time","minutes","source","words_count"]].rename(columns={"human_time":"when","minutes":"minutes","source":"source","words_count":"words"}), use_container_width=True)

csv = display_df.to_csv(index=False)
st.download_button("Export history as CSV", data=csv, file_name=f"{st.session_state.username}_history.csv", mime="text/csv")

st.success("Dashboard loaded. Keep reading! (Each Stop or Add minutes stores a session.)")
