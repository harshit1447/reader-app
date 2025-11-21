"""
Reader App — full single-file Streamlit app
Features:
- Signup / Login with SQLite
- Small centered login box (positioned visually behind header)
- Input modes: Website URL, Paste text, Upload (.txt .pdf .docx)
- Robust extraction: newspaper3k (optional) + BeautifulSoup heuristics
- Word & sentence counts, estimated read time
- Simple category inference (top TF-IDF terms)
- Add to Dashboard: saves (url/title/category/words/sentences/seconds/timestamp)
- Timer / tracker (Start/Stop / Add minutes)
- Per-user dashboard with totals, charts, CSV export
- CSS for layout; login box positioned up behind header (negative margin)
- Uses SQLite DB file: reader_app.db (created automatically)
- Optional sample image path (local) for header preview: SAMPLE_IMAGE
"""

import streamlit as st
import sqlite3
import time
import datetime as dt
from io import BytesIO
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from passlib.hash import pbkdf2_sha256

# Optional imports (handle gracefully)
try:
    from newspaper import Article as NewspaperArticle
    NEWSPAPER_AVAILABLE = True
except Exception:
    NEWSPAPER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF_AVAILABLE = True
except Exception:
    PYPDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

# Sample image path (from conversation assets) — you can remove or replace it
SAMPLE_IMAGE = "/mnt/data/38296530-7105-43ab-9705-14f13f6b28e0.png"

# Ensure NLTK punkt tokenizer
for pkg in ("punkt",):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

DB_FILE = "reader_app.db"

# -------------------- DB helpers --------------------
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        display_name TEXT,
        email TEXT,
        created_at TEXT NOT NULL
    )""")
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
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reading_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        seconds INTEGER NOT NULL,
        source TEXT,
        words_count INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    conn.commit()
    conn.close()

init_db()

# -------------------- Auth helpers --------------------
def create_user(username: str, password: str, display_name: str=None, email: str=None):
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password required"
    conn = get_conn()
    cur = conn.cursor()
    try:
        ph = pbkdf2_sha256.hash(password)
        cur.execute("INSERT INTO users (username, password_hash, display_name, email, created_at) VALUES (?, ?, ?, ?, ?)",
                    (username, ph, display_name, email, dt.datetime.utcnow().isoformat()))
        conn.commit()
        return True, "Account created"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
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

# -------------------- Article extraction --------------------
def _extract_text_newspaper(url: str):
    try:
        art = NewspaperArticle(url)
        art.download()
        art.parse()
        text = (art.text or "").strip()
        title = (art.title or "").strip()
        if text and len(text) > 80:
            return text, title or ""
    except Exception:
        pass
    return "", ""

def _extract_text_bs(url: str, min_len=200):
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # get title
        title = ""
        t = soup.find("meta", property="og:title")
        if t and t.get("content"):
            title = t.get("content").strip()
        if not title and soup.title:
            title = soup.title.get_text(strip=True)
        # remove unwanted tags
        for tag in soup(["script","style","noscript","header","footer","svg","nav","iframe","aside","form","button"]):
            tag.extract()
        # try article/main
        article_text = []
        article = soup.find("article")
        if article:
            for p in article.find_all(["p","h1","h2","h3","li"]):
                txt = p.get_text(" ", strip=True)
                if txt and len(txt)>30:
                    article_text.append(txt)
        if not article_text:
            main = soup.find("main")
            if main:
                for p in main.find_all(["p","h1","h2","h3","li"]):
                    txt = p.get_text(" ", strip=True)
                    if txt and len(txt)>30:
                        article_text.append(txt)
        if not article_text:
            # fallback to important selectors
            selectors = [".article-content", ".post-content", ".entry-content", ".content", ".post-body", ".article-body"]
            for sel in selectors:
                cont = soup.select_one(sel)
                if cont:
                    for p in cont.find_all(["p","h1","h2","h3","li"]):
                        txt = p.get_text(" ", strip=True)
                        if txt and len(txt)>30:
                            article_text.append(txt)
                    if article_text:
                        break
        if not article_text:
            # last fallback: all paragraphs longer than 50 chars
            ps = soup.find_all("p")
            for p in ps:
                txt = p.get_text(" ", strip=True)
                if txt and len(txt) > 50:
                    article_text.append(txt)
        # clean and dedupe
        cleaned = []
        seen = set()
        for block in article_text:
            if block not in seen:
                seen.add(block)
                cleaned.append(block)
        joined = "\n\n".join(cleaned)
        if joined and len(joined) >= min_len:
            return joined, title
    except Exception:
        pass
    return "", ""

def extract_text_from_url(url: str):
    # Try newspaper first (if installed)
    if NEWSPAPER_AVAILABLE:
        txt, title = _extract_text_newspaper(url)
        if txt:
            return txt, title, "newspaper3k"
    # Try BeautifulSoup heuristics
    txt, title = _extract_text_bs(url)
    if txt:
        return txt, title, "beautifulsoup"
    # failed
    return "", "", "failed"

def load_text_from_pdf_bytes(data: bytes):
    if not PYPDF_AVAILABLE:
        return ""
    try:
        reader = PdfReader(BytesIO(data))
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n\n".join(pages)
    except Exception:
        return ""

def load_text_from_docx_bytes(data: bytes):
    if not DOCX_AVAILABLE:
        return ""
    try:
        with BytesIO(data) as bio:
            doc = docx.Document(bio)
            paras = [p.text for p in doc.paragraphs if p.text]
            return "\n\n".join(paras)
    except Exception:
        return ""

def safe_load_uploaded_file(uploaded):
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        return load_text_from_pdf_bytes(data)
    if name.endswith(".docx"):
        return load_text_from_docx_bytes(data)
    try:
        return data.decode("utf-8")
    except Exception:
        try:
            return data.decode("latin-1")
        except Exception:
            return ""

# -------------------- Analysis helpers --------------------
def word_and_sentence_counts(text):
    sents = sent_tokenize(text)
    toks = word_tokenize(text)
    words = [t for t in toks if any(c.isalnum() for c in t)]
    return len(words), len(sents)

def estimate_reading_seconds(words_count, wpm=200):
    if wpm <= 0:
        return 0
    minutes = words_count / float(wpm)
    return int(round(minutes * 60))

def infer_category(text, top_k=1):
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2000)
        X = vec.fit_transform([text])
        terms = vec.get_feature_names_out()
        scores = np.asarray(X.todense()).ravel()
        if scores.sum() == 0:
            return "General"
        top_idx = scores.argsort()[-top_k:][::-1]
        top_terms = [terms[i] for i in top_idx if i < len(terms)]
        return ", ".join(top_terms)
    except Exception:
        return "General"

# -------------------- DB article/session actions --------------------
def save_article_to_db(user_id, url, title, category, words, sentences, seconds):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO articles (user_id, url, title, category, words, sentences, estimated_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, url, title, category, words, sentences, int(seconds), dt.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def fetch_user_articles(user_id):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM articles WHERE user_id = ? ORDER BY created_at DESC", conn, params=(user_id,))
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["id","user_id","url","title","category","words","sentences","estimated_seconds","created_at"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def store_session(user_id, seconds, source=None, words_count=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO reading_sessions (user_id, seconds, source, words_count, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, int(seconds), source, words_count, dt.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def fetch_user_sessions(user_id):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM reading_sessions WHERE user_id = ? ORDER BY created_at DESC", conn, params=(user_id,))
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["id","user_id","seconds","source","words_count","created_at"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

# -------------------- Small UI utilities --------------------
def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.stop()
    except Exception:
        st.stop()

# -------------------- CSS and layout --------------------
def load_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        * {{ font-family: 'Inter', sans-serif; }}

        /* Header area */
        .app-header {{
            text-align: center;
            padding-top: 40px;
            padding-bottom: 12px;
        }}
        .app-title {{
            font-size: 44px;
            font-weight: 800;
            margin: 0;
            color: #7c6ef5;
            letter-spacing: -0.5px;
        }}
        .app-subtitle {{
            font-size: 16px;
            color: #9aa0a6;
            margin: 8px 0 28px 0;
        }}

        /* Auth container - lifted up behind header */
        .auth-container {{
            margin-top: -70px !important;   /* <- lifts it upward behind header */
            max-width: 420px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            background: #ffffff !important;
            border-radius: 18px !important;
            padding: 24px 22px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.35) !important;
            text-align: center !important;
        }}

        .auth-title {{ font-size: 18px; font-weight:700; margin-bottom: 6px; color:#111827; }}
        .auth-subtitle {{ color:#6b7280; margin-bottom: 10px; }}

        .stTextInput > div > div > input {{
            border-radius: 10px !important;
            border: 1px solid #d1d5db !important;
            padding: 8px 12px !important;
            font-size: 0.95em !important;
            max-width: 280px !important;
            margin: 6px auto !important;
            text-align: center !important;
        }}

        .stPasswordInput > div > div > input {{
            border-radius: 10px !important;
            border: 1px solid #d1d5db !important;
            padding: 8px 12px !important;
            font-size: 0.95em !important;
            max-width: 280px !important;
            margin: 6px auto !important;
            text-align: center !important;
        }}

        .glass-card {{
            background: rgba(255,255,255,0.96);
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 10px 30px rgba(2,6,23,0.08);
            border: 1px solid rgba(0,0,0,0.04);
        }}

        .preview-card {{
            padding: 14px;
            border-left: 4px solid #7c6ef5;
            background: #fbfdff;
            border-radius: 8px;
        }}

        /* Metric subtle styling */
        div[data-testid="metric-container"] {{
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0 8px 22px rgba(2,6,23,0.04);
        }}

        /* Buttons */
        .stButton>button {{
            background: linear-gradient(90deg,#7c6ef5,#5a67f2) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 10px 18px !important;
            font-weight: 700 !important;
            border: none !important;
        }}

        /* small screens adjustments */
        @media (max-width: 600px) {{
            .app-title {{ font-size: 28px; }}
            .auth-container {{ margin-top: -40px !important; padding: 18px !important; }}
            .stTextInput > div > div > input {{ max-width: 200px !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Reader — Full App", layout="wide")
load_css()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = "auth"

# persisted content holders
if "fetched_text" not in st.session_state:
    st.session_state.fetched_text = ""
if "fetched_title" not in st.session_state:
    st.session_state.fetched_title = ""
if "fetched_source" not in st.session_state:
    st.session_state.fetched_source = None
if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False
    st.session_state.timer_start = None

# ---------- Auth page ----------
def auth_page():
    # header (kept separate so the auth box can overlap)
    st.markdown(f"""
        <div class="app-header">
            <div style="display:flex; align-items:center; justify-content:center; gap:14px;">
                <img src="file://{SAMPLE_IMAGE}" style="width:44px;height:44px;border-radius:8px;object-fit:cover;box-shadow:0 6px 20px rgba(0,0,0,0.35);"/>
                <div>
                    <div class="app-title">Reader Dashboard</div>
                    <div class="app-subtitle">Save, track and manage your reading journey</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # centered auth container (this sits visually on top of header due to negative margin)
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Sign in or create an account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Your reading history will be saved under your profile</div>', unsafe_allow_html=True)

    mode = st.radio("", ["Login", "Sign up"], horizontal=True, label_visibility="collapsed")

    if mode == "Sign up":
        display_name = st.text_input("Display name (optional)", key="su_display")
        username = st.text_input("Username", key="su_username")
        email = st.text_input("Email (optional)", key="su_email")
        password = st.text_input("Password", type="password", key="su_password")
        if st.button("Create account"):
            ok, msg = create_user(username, password, display_name or None, email or None)
            if ok:
                uid = verify_user(username, password)
                st.session_state.user_id = int(uid) if uid else None
                st.session_state.page = "app"
                st.success("Account created — signed in.")
                time.sleep(0.3)
                safe_rerun()
            else:
                st.error(msg)
    else:
        username = st.text_input("Username", key="li_username")
        password = st.text_input("Password", type="password", key="li_password")
        if st.button("Log in"):
            uid = verify_user(username, password)
            if uid:
                st.session_state.user_id = int(uid)
                st.session_state.page = "app"
                st.success("Signed in.")
                time.sleep(0.2)
                safe_rerun()
            else:
                st.error("Invalid credentials")

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("Continue as guest"):
        st.session_state.user_id = None
        st.session_state.page = "app"
        safe_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Main app page ----------
def app_page():
    profile = get_profile(st.session_state.user_id) if st.session_state.user_id else None

    # Header area
    st.markdown("""
        <div style="text-align:center; padding-top:26px; padding-bottom:6px;">
            <div class="app-title">Reader Dashboard</div>
            <div class="app-subtitle">Track your reading, build momentum</div>
        </div>
    """, unsafe_allow_html=True)

    # User info + actions
    cols = st.columns([1,2,1])
    with cols[0]:
        if profile:
            st.markdown(f"**Signed in as:** {profile.get('display_name') or profile.get('username')}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.page = "auth"
                safe_rerun()
        else:
            st.markdown("**Guest**")
            if st.button("Sign in"):
                st.session_state.page = "auth"
                safe_rerun()

    # Left column: input / controls
    left, right = st.columns([1,2])
    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Add an article")
        input_mode = st.radio("Input mode", ["Website URL", "Paste text", "Upload file"], index=0)

        # ensure persisted keys
        if "url_input" not in st.session_state:
            st.session_state.url_input = ""
        if "paste_text" not in st.session_state:
            st.session_state.paste_text = ""
        if "uploaded_name" not in st.session_state:
            st.session_state.uploaded_name = ""

        if input_mode == "Website URL":
            st.session_state.url_input = st.text_input("Paste article URL (include https://)", value=st.session_state.url_input)
            if st.button("Fetch & Analyze"):
                if not st.session_state.url_input.strip():
                    st.warning("Please paste a valid URL.")
                else:
                    with st.spinner("Fetching article..."):
                        txt, title, method = extract_text_from_url(st.session_state.url_input.strip())
                        if not txt:
                            st.error("Could not extract article text. Try the Paste Text option or a simpler URL.")
                            st.session_state.fetched_text = ""
                            st.session_state.fetched_title = ""
                            st.session_state.fetched_source = None
                        else:
                            st.session_state.fetched_text = txt
                            st.session_state.fetched_title = title or (txt.splitlines()[0][:120] if txt else "")
                            st.session_state.fetched_source = st.session_state.url_input.strip()
                            st.session_state.manual_mode = False
                            st.success(f"Fetched (method: {method})")
        elif input_mode == "Paste text":
            st.session_state.paste_text = st.text_area("Paste full article text here", value=st.session_state.paste_text, height=240)
            st.session_state.fetched_text = st.session_state.paste_text
            st.session_state.fetched_title = st.text_input("Article title (optional)", value=st.session_state.fetched_title or "")
            st.session_state.fetched_source = st.text_input("Source URL (optional)", value=st.session_state.fetched_source or "")
            st.session_state.manual_mode = True
            if st.button("Analyze pasted text"):
                if not st.session_state.fetched_text.strip():
                    st.warning("Paste the article text first.")
                else:
                    st.success("Text loaded for analysis.")
        else:  # Upload
            up = st.file_uploader("Upload .txt, .pdf, .docx", type=["txt","pdf","docx"])
            if up:
                st.session_state.uploaded_name = up.name
                loaded = safe_load_uploaded_file(up)
                if not loaded:
                    st.error("Could not extract from uploaded file.")
                else:
                    st.session_state.fetched_text = loaded
                    st.session_state.fetched_title = up.name
                    st.session_state.fetched_source = up.name
                    st.session_state.manual_mode = True
                    st.success("File loaded for analysis.")

        st.markdown("---")
        st.markdown("### Reading speed")
        wpm_choice = st.selectbox("WPM", ["200 (avg)", "150 (slow)", "250 (fast)", "Custom"], index=0)
        if wpm_choice == "200 (avg)":
            wpm = 200
        elif wpm_choice == "150 (slow)":
            wpm = 150
        elif wpm_choice == "250 (fast)":
            wpm = 250
        else:
            wpm = st.number_input("Custom WPM", min_value=50, max_value=1000, value=200, step=10)

        st.markdown("---")
        st.markdown("### Tracker")
        if not st.session_state.timer_running:
            if st.button("Start timer"):
                st.session_state.timer_running = True
                st.session_state.timer_start = time.time()
                st.success("Timer started")
        else:
            if st.button("Stop timer"):
                if st.session_state.timer_running and st.session_state.timer_start:
                    elapsed = int(round(time.time() - st.session_state.timer_start))
                    st.session_state.timer_running = False
                    st.session_state.timer_start = None
                    st.success(f"Stopped — recorded {elapsed} seconds")
                    if st.session_state.user_id:
                        wc = None
                        if st.session_state.fetched_text:
                            wc, _ = word_and_sentence_counts(st.session_state.fetched_text)
                        store_session(st.session_state.user_id, elapsed, st.session_state.fetched_source, wc)
                    else:
                        st.info("Guest session not saved. Log in to persist history.")
                else:
                    st.info("Timer wasn't running.")

        add_min = st.number_input("Add minutes", min_value=0, max_value=1000, value=0, step=1)
        if st.button("Add minutes"):
            if add_min > 0:
                seconds = int(add_min * 60)
                if st.session_state.user_id:
                    wc = None
                    if st.session_state.fetched_text:
                        wc, _ = word_and_sentence_counts(st.session_state.fetched_text)
                    store_session(st.session_state.user_id, seconds, st.session_state.fetched_source, wc)
                    st.success(f"Added {add_min} minutes.")
                else:
                    st.info("Guest session not saved. Log in to persist history.")

        st.markdown('</div>', unsafe_allow_html=True)  # close left glass-card

    # Right column: analysis & dashboard
    with right:
        # Show analysis if we have text
        txt = st.session_state.get("fetched_text", "")
        title = st.session_state.get("fetched_title", "")
        src = st.session_state.get("fetched_source", None)
        if txt and txt.strip():
            words, sentences = word_and_sentence_counts(txt)
            est_seconds = estimate_reading_seconds(words, wpm)
            category = infer_category(txt) if not st.session_state.manual_mode else (st.session_state.get("manual_category") or infer_category(txt))
            # Analysis card
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Article analysis")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Words", f"{words:,}")
            col2.metric("Sentences", f"{sentences:,}")
            col3.metric("Est. read time", f"{int(est_seconds/60)}m {est_seconds%60}s")
            col4.metric("Category", category)
            st.markdown("---")
            st.markdown("**Preview**")
            preview = title if title and len(title) > 10 else txt[:400]
            st.markdown(f'<div class="preview-card">{preview}{"..." if len(txt) > 400 else ""}</div>', unsafe_allow_html=True)
            st.markdown("")
            if st.button("➕ Add to dashboard"):
                if not st.session_state.user_id:
                    st.info("Sign in to save articles to your dashboard.")
                else:
                    save_article_to_db(st.session_state.user_id, src or f"manual-{int(time.time())}", title or preview[:120], category, words, sentences, est_seconds)
                    st.success("Saved to your dashboard.")
                    # clear fetched state
                    st.session_state.fetched_text = ""
                    st.session_state.fetched_title = ""
                    st.session_state.fetched_source = None
                    st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No article loaded. Use the left panel to fetch, paste, or upload an article.")

        # Dashboard (only for signed-in users)
        st.markdown("<br/>")
        st.markdown("## Your Dashboard")
        if not st.session_state.user_id:
            st.info("Sign in to view saved articles and stats.")
        else:
            df_articles = fetch_user_articles(st.session_state.user_id)
            if df_articles.empty:
                st.info("No saved articles yet.")
            else:
                total_articles = len(df_articles)
                total_words = int(df_articles["words"].sum())
                total_seconds = int(df_articles["estimated_seconds"].sum())
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                s1.metric("Total articles", f"{total_articles}")
                s2.metric("Total words", f"{total_words:,}")
                s3.metric("Total time", f"{int(total_seconds/3600)}h {int((total_seconds%3600)/60)}m")
                st.markdown("---")
                display = df_articles.copy()
                display["Added at"] = display["created_at"].dt.strftime("%Y-%m-%d %H:%M")
                display["Minutes"] = (display["estimated_seconds"]/60).round(1)
                display = display[["Added at","title","category","words","sentences","Minutes","url"]]
                display = display.rename(columns={"title":"Title","category":"Category","words":"Words","sentences":"Sentences","url":"URL"})
                st.dataframe(display, use_container_width=True)
                csv = df_articles.to_csv(index=False)
                st.download_button("Export articles CSV", data=csv, file_name="my_articles.csv", mime="text/csv")
                st.markdown('</div>', unsafe_allow_html=True)

        # Sessions / time chart
        if st.session_state.user_id:
            sessions = fetch_user_sessions(st.session_state.user_id)
            if not sessions.empty:
                # totals
                today = dt.date.today()
                sessions["date"] = sessions["created_at"].dt.date
                seconds_today = int(sessions[sessions["date"] == today]["seconds"].sum())
                seconds_7d = int(sessions[sessions["created_at"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=7))]["seconds"].sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Today", f"{int(seconds_today/60)} min")
                m2.metric("Last 7 days", f"{int(seconds_7d/60)} min")
                m3.metric("Sessions stored", f"{len(sessions)}")
                st.markdown("### Daily reading (last 90 days)")
                series = sessions.set_index("created_at").resample("D")["seconds"].sum().reindex(pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=90), fill_value=0)
                series.index = pd.to_datetime(series.index).date
                st.line_chart(series)
    # end right column

# Router
if st.session_state.page == "auth" and not st.session_state.user_id:
    auth_page()
else:
    app_page()
