"""
Streamlit Article Reader — Save articles to profile dashboard

Flow:
1. User pastes a website URL.
2. App extracts article text, shows words, sentences, estimated read time.
3. User clicks ADD to save the article (URL, title/snippet, category, words, sentences, time, timestamp).
4. Dashboard shows saved articles for signed-in user and total time.

Notes:
- Uses local SQLite DB: reader_app.db
- Category is inferred as the top TF-IDF term (simple heuristic).
- For JS-heavy sites, extraction may fail; consider 'trafilatura' later.
"""

import streamlit as st
import sqlite3
import time
from io import BytesIO
import datetime as dt
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
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

# -------------------- DB init --------------------
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # Users table (with display_name)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        display_name TEXT,
        email TEXT,
        created_at TEXT NOT NULL
    )""")
    # Articles table - stores saved articles metadata
    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        url TEXT NOT NULL,
        title TEXT,
        category TEXT,
        words INTEGER,
        sentences INTEGER,
        estimated_seconds INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    conn.commit()
    conn.close()

init_db()

# -------------------- Auth --------------------
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

# -------------------- Article extraction & utils --------------------
def extract_text_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
            tag.extract()
        # prefer article or main, otherwise all <p>
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

def word_and_sentence_counts(text: str):
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    words_filtered = [t for t in tokens if any(c.isalnum() for c in t)]
    return len(words_filtered), len(sentences)

def estimate_reading_seconds(words_count: int, wpm: int = 200) -> int:
    if wpm <= 0:
        return 0
    minutes = words_count / float(wpm)
    return int(round(minutes * 60))

def infer_category(text: str, top_k=1):
    # Simple heuristic: compute TF-IDF across the single article and return top terms.
    # For short text this is noisy but works as a lightweight topic label.
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        X = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        # get tf-idf scores for the single doc
        scores = np.asarray(X.todense()).ravel()
        if scores.sum() == 0:
            return None
        top_idx = scores.argsort()[-top_k:][::-1]
        top_terms = [terms[i] for i in top_idx if i < len(terms)]
        return ", ".join(top_terms)
    except Exception:
        return None

# -------------------- DB interactions for articles --------------------
def save_article(user_id: int, url: str, title: str, category: str, words: int, sentences: int, seconds: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO articles (user_id, url, title, category, words, sentences, estimated_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, url, title, category, words, sentences, int(seconds), dt.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def fetch_user_articles(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM articles WHERE user_id = ? ORDER BY created_at DESC", conn, params=(user_id,))
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["id","user_id","url","title","category","words","sentences","estimated_seconds","created_at"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Reader Dashboard", layout="wide")
st.title("Reader — Save articles to profile dashboard")

# session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = "auth"

# --- Authentication UI (simple full-screen) ---
def auth_page():
    st.markdown("<h2 style='text-align:center'>Sign up or Login</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 0.7, 1])
    with col2:
        mode = st.radio("Choose", ["Login", "Sign up"], horizontal=True)
        if mode == "Sign up":
            dn = st.text_input("Display name (optional)", key="su_disp")
            un = st.text_input("Username", key="su_user")
            em = st.text_input("Email (optional)", key="su_email")
            pw = st.text_input("Password", type="password", key="su_pass")
            if st.button("Create account"):
                ok, msg = create_user(un, pw, display_name=dn or None, email=em or None)
                if ok:
                    uid = verify_user(un, pw)
                    st.session_state.user_id = int(uid) if uid else None
                    st.session_state.page = "app"
                    st.success("Account created — you're signed in.")
                    st.experimental_rerun()
                else:
                    st.error(msg)
        else:
            un = st.text_input("Username", key="li_user")
            pw = st.text_input("Password", type="password", key="li_pass")
            if st.button("Log in"):
                uid = verify_user(un, pw)
                if uid:
                    st.session_state.user_id = int(uid)
                    st.session_state.page = "app"
                    st.success("Signed in.")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
        st.markdown("---")
        if st.button("Continue as guest"):
            st.session_state.user_id = None
            st.session_state.page = "app"
            st.experimental_rerun()

# --- Main app page ---
def app_page():
    profile = get_profile(st.session_state.user_id) if st.session_state.user_id else None
    header_cols = st.columns([1,2,1])
    with header_cols[0]:
        if profile:
            st.write(f"Signed in as: **{profile.get('display_name') or profile.get('username')}**")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.page = "auth"
                st.experimental_rerun()
        else:
            st.write("Guest")
            if st.button("Back to Login"):
                st.session_state.page = "auth"
                st.experimental_rerun()

    # -- URL input area (center-top)
    st.subheader("Add article from website")
    url_input = st.text_input("Paste website URL (include https://)", value=st.session_state.get("url_input",""))
    if st.button("Fetch & Analyze"):
        st.session_state.url_input = url_input
        if not url_input or not url_input.strip():
            st.warning("Please paste a valid URL.")
        else:
            st.info("Fetching article — may take a few seconds...")
            fetched = extract_text_from_url(url_input.strip())
            if not fetched:
                st.error("Could not extract article text. Try another URL (or a simpler site).")
                st.session_state.fetched_text = ""
                st.session_state.fetched_title = ""
            else:
                st.session_state.fetched_text = fetched
                # simple title: first non-empty line up to 120 chars
                first_line = next((line.strip() for line in fetched.splitlines() if line.strip()), "")
                st.session_state.fetched_title = first_line[:120]
                st.success("Fetched article text.")

    # show extracted metrics if present
    fetched_text = st.session_state.get("fetched_text", "")
    fetched_title = st.session_state.get("fetched_title", "")
    if fetched_text:
        words, sentences = word_and_sentence_counts(fetched_text)
        est_seconds = estimate_reading_seconds(words, wpm=200)
        category = infer_category(fetched_text) or "General"
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Words", f"{words:,}")
        col2.metric("Sentences", f"{sentences:,}")
        col3.metric("Est. read time", f"{int(est_seconds/60)} min {est_seconds%60} sec")
        col4.metric("Category", category)
        st.markdown("**Preview / Title:**")
        st.write(fetched_title)

        # Add button saves article for logged-in user
        if st.button("Add to my dashboard"):
            if not st.session_state.user_id:
                st.info("You are not signed in. Create an account or login to save this article.")
            else:
                save_article(
                    st.session_state.user_id,
                    st.session_state.url_input.strip(),
                    fetched_title,
                    category,
                    words,
                    sentences,
                    est_seconds
                )
                st.success("Article saved to your dashboard.")
                # clear fetched state optionally
                st.session_state.fetched_text = ""
                st.session_state.fetched_title = ""
                st.session_state.url_input = ""
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("Your saved articles")
    if not st.session_state.user_id:
        st.info("Sign up / Log in to see your saved articles and totals.")
        return

    articles_df = fetch_user_articles(st.session_state.user_id)
    if articles_df.empty:
        st.info("No saved articles yet. Add articles from the box above.")
        return

    # display table
    display = articles_df.copy()
    display["when"] = display["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display["time_minutes"] = (display["estimated_seconds"]/60).round(2)
    display = display[["when","url","title","category","words","sentences","time_minutes"]]
    display = display.rename(columns={
        "when":"Added at",
        "url":"URL",
        "title":"Title (preview)",
        "category":"Category",
        "words":"Words",
        "sentences":"Sentences",
        "time_minutes":"Minutes"
    })
    st.dataframe(display, use_container_width=True)

    # total time
    total_seconds = int(articles_df["estimated_seconds"].sum())
    st.markdown("---")
    st.metric("Total saved reading time", f"{int(total_seconds/60)} min {total_seconds%60} sec")

# Router
if st.session_state.page == "auth" or st.session_state.user_id is None:
    auth_page()
else:
    app_page()
