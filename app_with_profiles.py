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

# Try to import newspaper3k for better article extraction
try:
    from newspaper import Article as NewspaperArticle
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

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
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app styling with better background */
    .main {
        background: #0a0e27;
        padding: 0;
    }
    
    /* Add decorative gradient overlay */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 400px;
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.15) 0%, transparent 100%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        position: relative;
        z-index: 1;
    }
    
    /* Card containers with better contrast */
    .glass-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 35px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 25px 0;
    }
    
    /* Modern header styling */
    .app-header {
        text-align: center;
        padding: 60px 20px 40px 20px;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }
    
    .app-title {
        font-size: 3.8em;
        font-weight: 800;
        margin: 0 0 15px 0;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.3);
    }
    
    .app-subtitle {
        font-size: 1.3em;
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
        font-weight: 400;
    }
    
    /* User info bar with better visibility */
    .user-info-bar {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 15px 25px;
        margin: 0 auto 30px auto;
        max-width: 1200px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .user-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 30px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        font-size: 0.95em;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
    }
    
    /* Metric cards with better contrast */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.6);
    }
    
    div[data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 600;
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2.2em;
        font-weight: 800;
    }
    
    /* Section headers with better visibility */
    .section-header {
        color: white;
        font-size: 2.2em;
        font-weight: 700;
        margin: 40px 0 25px 0;
        text-align: center;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .subsection-header {
        color: #1a1a2e;
        font-size: 1.5em;
        font-weight: 700;
        margin: 0 0 20px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Button styling with better contrast */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 1em;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        border: 2px solid transparent;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.7);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Primary button variant */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.5);
    }
    
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.7);
    }
    
    /* Input styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 14px;
        border: 2px solid #e0e0e0;
        padding: 14px 18px;
        font-size: 1em;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
        outline: none;
    }
    
    .stTextInput>label,
    .stTextArea>label {
        font-weight: 600;
        color: #1a1a2e;
        font-size: 0.95em;
        margin-bottom: 8px;
    }
    
    /* Tabs styling with better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.12);
        color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1a1a2e !important;
        border-color: white !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Radio button styling */
    .stRadio>div {
        background: transparent;
        padding: 15px;
        border-radius: 14px;
        gap: 12px;
    }
    
    .stRadio>div>label {
        background: rgba(255, 255, 255, 0.12);
        padding: 12px 24px;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stRadio>div>label:hover {
        background: rgba(255, 255, 255, 0.18);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .stRadio>div>label[data-checked="true"] {
        background: white;
        color: #1a1a2e;
        border-color: white;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
    }
    
    /* Preview card */
    .preview-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-left: 5px solid #667eea;
        padding: 24px;
        border-radius: 14px;
        margin: 20px 0;
        font-size: 1.05em;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Messages styling */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 14px;
        padding: 16px 20px;
        font-weight: 500;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.12);
        border-radius: 12px;
        font-weight: 600;
        padding: 14px 18px;
    }
    
    /* Auth container with better contrast */
    .auth-container {
        max-width: 480px;
        margin: 40px auto;
        background: white;
        border-radius: 28px;
        padding: 50px 45px;
        box-shadow: 0 25px 80px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .auth-title {
        font-size: 2em;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .auth-subtitle {
        color: #666;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.05em;
    }
    
    /* Caption styling */
    .caption {
        color: #999;
        font-size: 0.9em;
        font-style: italic;
        margin-top: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
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
def extract_text_from_url(url: str) -> tuple:
    """
    Extract article text from URL. Returns (text, title, success_method)
    """
    # Strategy 1: Try newspaper3k first (best for news/blog articles)
    if NEWSPAPER_AVAILABLE:
        try:
            article = NewspaperArticle(url)
            article.download()
            article.parse()
            if article.text and len(article.text.strip()) > 100:
                return article.text.strip(), article.title or "", "newspaper3k"
        except Exception:
            pass
    
    # Strategy 2: Enhanced BeautifulSoup extraction
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Check for og:title (better for social media sites)
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title.get("content")
        
        # Try h1 as fallback
        if not title and soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        
        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav", "iframe", "aside", "button", "form"]):
            tag.extract()
        
        article_text = []
        
        # Substack-specific extraction
        if "substack.com" in url.lower():
            # Try Substack's post body class
            post_body = soup.find("div", class_=lambda x: x and "body" in x.lower() if x else False)
            if post_body:
                for elem in post_body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
                    text = elem.get_text(strip=True)
                    if text and len(text) > 15:
                        article_text.append(text)
            
            # Try finding post content areas
            if not article_text:
                content_divs = soup.find_all("div", class_=lambda x: x and ("post" in x.lower() or "content" in x.lower()) if x else False)
                for div in content_divs:
                    for elem in div.find_all(["p", "h2", "h3", "h4"]):
                        text = elem.get_text(strip=True)
                        if text and len(text) > 15:
                            article_text.append(text)
        
        # Medium-specific extraction
        if "medium.com" in url.lower() and not article_text:
            # Try Medium's data attributes
            paragraphs = soup.find_all(attrs={"data-selectable-paragraph": True})
            if paragraphs:
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 15:
                        article_text.append(text)
            
            # Try Medium's article structure
            if not article_text:
                article_sections = soup.find_all("section")
                for section in article_sections:
                    for elem in section.find_all(["p", "h1", "h2", "h3", "h4"]):
                        text = elem.get_text(strip=True)
                        if text and len(text) > 15:
                            article_text.append(text)
        
        # Standard article extraction
        if not article_text:
            article = soup.find("article")
            if article:
                for elem in article.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
                    text = elem.get_text(strip=True)
                    if text and len(text) > 20:
                        article_text.append(text)
        
        # Main tag extraction
        if not article_text:
            main = soup.find("main")
            if main:
                for elem in main.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
                    text = elem.get_text(strip=True)
                    if text and len(text) > 20:
                        article_text.append(text)
        
        # Common content containers
        if not article_text:
            content_selectors = [
                ".article-content", ".post-content", ".entry-content",
                "#content", ".content", ".post-body", ".article-body",
                ".post__content"
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
        
        # Fallback to all meaningful paragraphs
        if not article_text:
            all_paragraphs = soup.find_all("p")
            article_text = [p.get_text(strip=True) for p in all_paragraphs if len(p.get_text(strip=True)) > 30]
        
        # Clean up and deduplicate
        seen = set()
        cleaned_text = []
        for text in article_text:
            # Skip common footer/header text
            lower_text = text.lower()
            if any(skip in lower_text for skip in ["cookie", "subscribe", "sign up", "follow us", "share this"]):
                if len(text) < 100:  # Only skip if it's short
                    continue
            
            if text and text not in seen and len(text) > 20:
                seen.add(text)
                cleaned_text.append(text)
        
        final_text = "\n\n".join(cleaned_text)
        if final_text.strip() and len(final_text) > 200:  # Ensure we got substantial content
            return final_text.strip(), title, "beautifulsoup"
        
    except Exception as e:
        pass
    
    return "", "", "failed"

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
    
    st.markdown('<h2 class="auth-title">Welcome</h2>', unsafe_allow_html=True)
    st.markdown('<p class="auth-subtitle">Sign in to start tracking your articles</p>', unsafe_allow_html=True)
    
    mode = st.radio("", ["Login", "Sign up"], horizontal=True, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if mode == "Sign up":
        dn = st.text_input("Display name (optional)", key="su_disp", placeholder="Your name")
        un = st.text_input("Username", key="su_user", placeholder="Choose a username")
        em = st.text_input("Email (optional)", key="su_email", placeholder="your@email.com")
        pw = st.text_input("Password", type="password", key="su_pass", placeholder="Create a password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Create account", use_container_width=True, type="primary"):
            ok, msg = create_user(un, pw, display_name=dn or None, email=em or None)
            if ok:
                uid = verify_user(un, pw)
                st.session_state.user_id = int(uid) if uid else None
                st.session_state.page = "app"
                st.success("🎉 Account created!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(msg)
    else:
        un = st.text_input("Username", key="li_user", placeholder="Your username")
        pw = st.text_input("Password", type="password", key="li_pass", placeholder="Your password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Log in", use_container_width=True, type="primary"):
            uid = verify_user(un, pw)
            if uid:
                st.session_state.user_id = int(uid)
                st.session_state.page = "app"
                st.success("✅ Welcome back!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
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
    st.markdown('<div class="user-info-bar">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if profile:
            st.markdown(f'<div class="user-badge">👤 {profile.get("display_name") or profile.get("username")}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="user-badge">👤 Guest Mode</div>', unsafe_allow_html=True)
    
    with col3:
        if profile:
            if st.button("Logout", use_container_width=True):
                st.session_state.user_id = None
                st.session_state.page = "auth"
                st.rerun()
        else:
            if st.button("Sign In", use_container_width=True):
                st.session_state.page = "auth"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">🔗 Add New Article</div>', unsafe_allow_html=True)
    
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
                    fetched_text, fetched_title, method = extract_text_from_url(url_input.strip())
                    if not fetched_text:
                        st.error("❌ Could not extract article text. The site may be JavaScript-heavy or have anti-scraping protection.")
                        
                        # Show helpful troubleshooting box
                        with st.expander("💡 Troubleshooting Tips", expanded=True):
                            site_type = ""
                            if "medium.com" in url_input.lower():
                                site_type = "Medium"
                            elif "substack.com" in url_input.lower():
                                site_type = "Substack"
                            elif "linkedin.com" in url_input.lower():
                                site_type = "LinkedIn"
                            
                            tips = f"**For {site_type} articles:**\n\n" if site_type else "**Quick Solutions:**\n\n"
                            
                            st.markdown(tips + """
                            1. **✅ Best Solution - Use Manual Entry:**
                               - Click the "**✍️ Paste Text**" tab above
                               - Open the article in your browser
                               - Select all text (Ctrl+A or Cmd+A)
                               - Copy and paste here with the title
                            
                            2. **Try simplifying the URL:**
                               - Remove everything after `?` in the URL
                               - Example: Keep only the base article URL
                            
                            3. **Optional - Better extraction:**
                               - Install newspaper3k: `pip install newspaper3k`
                               - Restart the app for improved extraction
                            
                            **Why this happens:** Sites like Medium, Substack, and LinkedIn load content 
                            dynamically with JavaScript, which basic web scrapers can't read. Manual entry 
                            works 100% of the time! 📝
                            """)
                        
                        st.session_state.fetched_text = ""
                        st.session_state.fetched_title = ""
                    else:
                        st.session_state.fetched_text = fetched_text
                        # Use extracted title or fallback to first line
                        if not fetched_title:
                            first_line = next((line.strip() for line in fetched_text.splitlines() if line.strip()), "")
                            fetched_title = first_line[:120]
                        st.session_state.fetched_title = fetched_title
                        st.session_state.manual_mode = False
                        st.success(f"✅ Article fetched successfully using {method}!")
                        if not NEWSPAPER_AVAILABLE:
                            st.info("💡 Install `newspaper3k` for even better article extraction from news sites and Medium.")
    
    with tab2:
        st.markdown('<div class="subsection-header" style="font-size: 1.2em; margin-top: 15px;">📝 Manual Article Entry</div>', unsafe_allow_html=True)
        
        # Instructions with visual steps
        with st.expander("📖 How to manually add an article (3 easy steps)", expanded=False):
            st.markdown("""
            ### Quick Guide:
            
            **Step 1:** Open the article in your browser
            
            **Step 2:** Select all text
            - **Windows/Linux:** Press `Ctrl + A`
            - **Mac:** Press `Cmd + A`
            
            **Step 3:** Copy & paste here
            - **Windows/Linux:** `Ctrl + C` then `Ctrl + V`
            - **Mac:** `Cmd + C` then `Cmd + V`
            
            ---
            
            ✅ **Works perfectly for:** Medium, Substack, LinkedIn, paywalled sites, PDFs (copy text first)
            
            ⏱️ **Takes:** ~30 seconds
            """)
        
        st.info("💡 This method works 100% of the time for any article you can read!")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            manual_title = st.text_input("📄 Article Title", placeholder="Enter the article title here", key="manual_title")
        with col2:
            manual_category = st.text_input("🏷️ Category (optional)", placeholder="e.g., Technology", key="manual_category")
        
        manual_url = st.text_input("🔗 Article URL (optional)", placeholder="https://example.com/article", key="manual_url")
        manual_text = st.text_area("📝 Article Content", 
                                   height=350, 
                                   placeholder="Paste the full article text here...\n\nTip: Select all (Ctrl+A / Cmd+A), copy, and paste.",
                                   key="manual_text")
        
        # Show character count
        if manual_text:
            char_count = len(manual_text)
            word_count_preview = len(manual_text.split())
            st.caption(f"✍️ {char_count:,} characters · ~{word_count_preview:,} words")
        else:
            st.markdown('<p class="caption">* Title and content are required</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📊 Analyze Pasted Text", use_container_width=True, type="primary", key="manual_btn"):
                if not manual_text.strip():
                    st.warning("⚠️ Please paste the article text.")
                elif not manual_title.strip():
                    st.warning("⚠️ Please enter an article title.")
                else:
                    st.session_state.fetched_text = manual_text.strip()
                    st.session_state.fetched_title = manual_title.strip()
                    st.session_state.url_input = manual_url.strip() if manual_url.strip() else f"manual-entry-{int(time.time())}"
                    st.session_state.manual_mode = True
                    st.session_state.manual_category = manual_category.strip() if manual_category.strip() else None
                    st.success("✅ Text analyzed successfully!")
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Show extracted metrics
    fetched_text = st.session_state.get("fetched_text", "")
    fetched_title = st.session_state.get("fetched_title", "")
    
    if fetched_text:
        words, sentences = word_and_sentence_counts(fetched_text)
        est_seconds = estimate_reading_seconds(words, wpm=200)
        
        # Use manual category if provided, otherwise infer
        if st.session_state.get("manual_mode") and st.session_state.get("manual_category"):
            category = st.session_state.manual_category
        else:
            category = infer_category(fetched_text) or "General"
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="subsection-header">📊 Article Analysis</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Words", f"{words:,}")
        col2.metric("💬 Sentences", f"{sentences:,}")
        col3.metric("⏱️ Read Time", f"{int(est_seconds/60)}m {est_seconds%60}s")
        col4.metric("🏷️ Category", category)
        
        # Show preview with better formatting
        st.markdown("---")
        st.markdown("**📄 Article Preview:**")
        preview_text = fetched_title if len(fetched_title) > 50 else fetched_text[:200]
        st.markdown(f'<div class="preview-card">{preview_text}{"..." if len(fetched_text) > 200 else ""}</div>', unsafe_allow_html=True)

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
