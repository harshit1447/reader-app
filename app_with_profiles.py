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

# -------------------- Custom CSS --------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Card containers */
    .card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 20px 0;
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        padding: 40px 0 20px 0;
        color: white;
    }
    
    .app-title {
        font-size: 3.5em;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .app-subtitle {
        font-size: 1.2em;
        margin-top: 10px;
        opacity: 0.9;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: none;
    }
    
    div[data-testid="metric-container"] label {
        color: white !important;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2em;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 12px;
        font-size: 1em;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Radio button styling */
    .stRadio>div {
        background: white;
        padding: 10px;
        border-radius: 12px;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Auth container */
    .auth-container {
        max-width: 450px;
        margin: 50px auto;
        background: white;
        border-radius: 25px;
        padding: 40px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
    }
    
    /* User badge */
    .user-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Section headers */
    .section-header {
        color: white;
        font-size: 2em;
        font-weight: 700;
        margin: 30px 0 20px 0;
        text-align: center;
    }
    
    /* Preview card */
    .preview-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        border-radius: 12px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav", "iframe"]):
            tag.extract()
        
        article_text = []
        
        # Strategy 1: Medium-specific selectors
        if "medium.com" in url.lower():
            # Try multiple Medium-specific selectors
            medium_selectors = [
                "article section",
                "article div",
                ".postArticle-content",
                "[data-selectable-paragraph]"
            ]
            for selector in medium_selectors:
                elements = soup.select(selector)
                if elements:
                    for elem in elements:
                        paragraphs = elem.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
                        article_text.extend([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    if article_text:
                        break
        
        # Strategy 2: Look for article tag
        if not article_text:
            article = soup.find("article")
            if article:
                # Get all text-containing elements
                for elem in article.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
                    text = elem.get_text(strip=True)
                    if text and len(text) > 20:  # Filter out very short snippets
                        article_text.append(text)
        
        # Strategy 3: Look for main tag
        if not article_text:
            main = soup.find("main")
            if main:
                for elem in main.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
                    text = elem.get_text(strip=True)
                    if text and len(text) > 20:
                        article_text.append(text)
        
        # Strategy 4: Look for common content containers
        if not article_text:
            content_selectors = [
                ".article-content",
                ".post-content",
                ".entry-content",
                "#content",
                ".content"
            ]
            for selector in content_selectors:
                container = soup.select_one(selector)
                if container:
                    for elem in container.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
                        text = elem.get_text(strip=True)
                        if text and len(text) > 20:
                            article_text.append(text)
                    if article_text:
                        break
        
        # Strategy 5: Fallback to all paragraphs
        if not article_text:
            all_paragraphs = soup.find_all("p")
            article_text = [p.get_text(strip=True) for p in all_paragraphs if len(p.get_text(strip=True)) > 30]
        
        # Clean up and deduplicate
        seen = set()
        cleaned_text = []
        for text in article_text:
            if text and text not in seen and len(text) > 20:
                seen.add(text)
                cleaned_text.append(text)
        
        final_text = "\n\n".join(cleaned_text)
        return final_text if final_text.strip() else ""
    except Exception as e:
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
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        X = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
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
st.set_page_config(page_title="Reader Dashboard", layout="wide", initial_sidebar_state="collapsed")
load_custom_css()

# session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = "auth"

# --- Authentication UI ---
def auth_page():
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">📚 Reader Dashboard</h1>
        <p class="app-subtitle">Save, track, and manage your reading journey</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    mode = st.radio("", ["Login", "Sign up"], horizontal=True, label_visibility="collapsed")
    
    if mode == "Sign up":
        st.markdown("### Create Your Account")
        dn = st.text_input("Display name (optional)", key="su_disp", placeholder="Enter your name")
        un = st.text_input("Username", key="su_user", placeholder="Choose a username")
        em = st.text_input("Email (optional)", key="su_email", placeholder="your@email.com")
        pw = st.text_input("Password", type="password", key="su_pass", placeholder="Enter a secure password")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Create account", use_container_width=True):
                ok, msg = create_user(un, pw, display_name=dn or None, email=em or None)
                if ok:
                    uid = verify_user(un, pw)
                    st.session_state.user_id = int(uid) if uid else None
                    st.session_state.page = "app"
                    st.success("🎉 Account created — you're signed in!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.markdown("### Welcome Back")
        un = st.text_input("Username", key="li_user", placeholder="Enter your username")
        pw = st.text_input("Password", type="password", key="li_pass", placeholder="Enter your password")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Log in", use_container_width=True):
                uid = verify_user(un, pw)
                if uid:
                    st.session_state.user_id = int(uid)
                    st.session_state.page = "app"
                    st.success("✅ Signed in successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Continue as guest", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.page = "app"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main app page ---
def app_page():
    profile = get_profile(st.session_state.user_id) if st.session_state.user_id else None
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">📚 Reader Dashboard</h1>
        <p class="app-subtitle">Your personal reading companion</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User info bar
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if profile:
            st.markdown(f'<div class="user-badge">👤 {profile.get("display_name") or profile.get("username")}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="user-badge">👤 Guest Mode</div>', unsafe_allow_html=True)
    
    with col3:
        if profile:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.user_id = None
                st.session_state.page = "auth"
                st.rerun()
        else:
            if st.button("🔑 Login", use_container_width=True):
                st.session_state.page = "auth"
                st.rerun()

    # Main content area
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔗 Add New Article")
    
    # Add tabs for URL vs Manual entry
    tab1, tab2 = st.tabs(["📎 From URL", "✍️ Paste Text"])
    
    with tab1:
        url_input = st.text_input("Paste website URL", 
                                   value=st.session_state.get("url_input",""),
                                   placeholder="https://example.com/article",
                                   label_visibility="collapsed")
        
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            fetch_button = st.button("🔍 Fetch & Analyze", use_container_width=True, key="fetch_btn")
        
        if fetch_button:
            st.session_state.url_input = url_input
            if not url_input or not url_input.strip():
                st.warning("⚠️ Please paste a valid URL.")
            else:
                with st.spinner("📡 Fetching article..."):
                    fetched = extract_text_from_url(url_input.strip())
                    if not fetched:
                        st.error("❌ Could not extract article text. The site may be JavaScript-heavy or have anti-scraping protection.")
                        st.info("💡 **Tip:** Try the 'Paste Text' tab to manually add the article content, or remove URL query parameters (everything after '?')")
                        st.session_state.fetched_text = ""
                        st.session_state.fetched_title = ""
                    else:
                        st.session_state.fetched_text = fetched
                        first_line = next((line.strip() for line in fetched.splitlines() if line.strip()), "")
                        st.session_state.fetched_title = first_line[:120]
                        st.session_state.manual_mode = False
                        st.success("✅ Article fetched successfully!")
    
    with tab2:
        st.info("📝 For sites that don't work with automatic extraction (like Medium), paste the article text directly here.")
        manual_title = st.text_input("Article Title", placeholder="Enter article title", key="manual_title")
        manual_url = st.text_input("Article URL (optional)", placeholder="https://...", key="manual_url")
        manual_text = st.text_area("Article Text", height=300, placeholder="Paste the full article text here...", key="manual_text")
        
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            if st.button("📊 Analyze Text", use_container_width=True, key="manual_btn"):
                if not manual_text.strip():
                    st.warning("⚠️ Please paste some article text.")
                elif not manual_title.strip():
                    st.warning("⚠️ Please enter an article title.")
                else:
                    st.session_state.fetched_text = manual_text.strip()
                    st.session_state.fetched_title = manual_title.strip()
                    st.session_state.url_input = manual_url.strip() if manual_url.strip() else "manual-entry"
                    st.session_state.manual_mode = True
                    st.success("✅ Text analyzed successfully!")

    st.markdown('</div>', unsafe_allow_html=True)

    # Show extracted metrics
    fetched_text = st.session_state.get("fetched_text", "")
    fetched_title = st.session_state.get("fetched_title", "")
    
    if fetched_text:
        words, sentences = word_and_sentence_counts(fetched_text)
        est_seconds = estimate_reading_seconds(words, wpm=200)
        category = infer_category(fetched_text) or "General"
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Article Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Words", f"{words:,}")
        col2.metric("💬 Sentences", f"{sentences:,}")
        col3.metric("⏱️ Read Time", f"{int(est_seconds/60)}m {est_seconds%60}s")
        col4.metric("🏷️ Category", category)
        
        st.markdown(f'<div class="preview-card"><strong>📄 Preview:</strong><br>{fetched_title}</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("➕ Add to Dashboard", use_container_width=True, type="primary"):
                if not st.session_state.user_id:
                    st.info("🔐 Sign in to save articles to your dashboard.")
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
                    st.success("🎉 Article saved to your dashboard!")
                    st.session_state.fetched_text = ""
                    st.session_state.fetched_title = ""
                    st.session_state.url_input = ""
                    time.sleep(1)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Saved articles section
    st.markdown('<h2 class="section-header">📖 Your Reading Library</h2>', unsafe_allow_html=True)
    
    if not st.session_state.user_id:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.info("🔐 Sign in to view your saved articles and reading statistics.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    articles_df = fetch_user_articles(st.session_state.user_id)
    
    if articles_df.empty:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.info("📭 No saved articles yet. Add your first article above!")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Display statistics
    total_articles = len(articles_df)
    total_words = int(articles_df["words"].sum())
    total_seconds = int(articles_df["estimated_seconds"].sum())
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("📚 Total Articles", f"{total_articles}")
    stat_col2.metric("📝 Total Words", f"{total_words:,}")
    stat_col3.metric("⏱️ Total Time", f"{int(total_seconds/3600)}h {int((total_seconds%3600)/60)}m")
    
    st.markdown("---")
    
    # Display table
    display = articles_df.copy()
    display["when"] = display["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    display["time_minutes"] = (display["estimated_seconds"]/60).round(1)
    display = display[["when","title","category","words","sentences","time_minutes","url"]]
    display = display.rename(columns={
        "when":"📅 Added",
        "title":"📄 Title",
        "category":"🏷️ Category",
        "words":"📝 Words",
        "sentences":"💬 Sentences",
        "time_minutes":"⏱️ Minutes",
        "url":"🔗 URL"
    })
    
    st.dataframe(display, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Router
if st.session_state.page == "auth" or st.session_state.user_id is None:
    auth_page()
else:
    app_page()
