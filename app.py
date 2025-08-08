import os
from datetime import datetime
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Orchestrator (lite, safe for Streamlit)
from agent_orchestrator import bind_youtube_functions, route, execute

# ========= Page Setup =========
st.set_page_config(page_title="Oracle Fusion Omniverse", page_icon="🛠️", layout="wide")
st.set_option("client.showErrorDetails", True)

# ========= Helpers: Secrets / Keys =========
def get_key(name: str) -> Optional[str]:
    return (st.secrets.get(name) if name in st.secrets else os.getenv(name)) or None

OPENAI_API_KEY = get_key("OPENAI_API_KEY") or get_key("OPENAI_KEY")
GEMINI_API_KEY = get_key("GEMINI_API_KEY") or get_key("GOOGLE_API_KEY")

# Keep some shared session state for sidebar outputs
if "sidebar_agent_out" not in st.session_state:
    st.session_state.sidebar_agent_out = ""
if "sidebar_yt_out" not in st.session_state:
    st.session_state.sidebar_yt_out = ""

# ========= Sidebar (Navigation + Quick Actions) =========
with st.sidebar:
    st.header("🔧 Settings")
    st.caption("Keys are optional — offline playbook works without them.")
    new_openai = st.text_input("OpenAI API Key (optional)", type="password", value=OPENAI_API_KEY or "")
    new_gemini = st.text_input("Google Gemini API Key (optional)", type="password", value=GEMINI_API_KEY or "")
    if new_openai:
        os.environ["OPENAI_API_KEY"] = new_openai
        OPENAI_API_KEY = new_openai
    if new_gemini:
        os.environ["GEMINI_API_KEY"] = new_gemini
        GEMINI_API_KEY = new_gemini

    if OPENAI_API_KEY or GEMINI_API_KEY:
        st.markdown(
            """
            <div style="border:1px solid #e3b341;padding:10px;border-radius:8px;background:#fff8e1">
            <b>Heads up:</b> You entered a key here. For permanent use, move it to
            <i>Settings → Secrets</i> so it never appears in the UI.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("Secrets format:\nOPENAI_API_KEY='sk-...'\nGEMINI_API_KEY='...'")
    st.write(f"OpenAI: {'✅' if OPENAI_API_KEY else '—'}  |  Gemini: {'✅' if GEMINI_API_KEY else '—'}")

    st.markdown("---")
    st.subheader("🧭 Navigation")
    nav_choice = st.radio(
        "Go to section",
        ["Overview", "Diagnose", "Analytics", "Security", "Optimize", "Integrations", "Voice Mode", "YouTube AI"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("⚡ Quick Actions (runs here)")

    # Quick Agent Console (sidebar output)
    agent_q = st.text_input(
        "Agent Console input",
        placeholder="Paste YouTube URL(s) or an ORA- error…",
        key="sb_agent_q",
    )
    if st.button("Run Agent (sidebar)"):
        s = route(agent_q)
        st.session_state.sidebar_agent_out = execute(s)

    if st.session_state.sidebar_agent_out:
        st.caption("Agent output")
        st.code(st.session_state.sidebar_agent_out)

    # Quick YT summarize (sidebar output)
    yt_quick = st.text_input(
        "YouTube URL (quick summarize)",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
        key="sb_yt_url",
    )
    if st.button("Summarize YT (sidebar)"):
        out = None
        try:
            out = None  # set in YouTube helpers later via binding
        except Exception as e:
            out = f"Error: {e}"
        # We’ll fill this after we define the functions; for now keep slot.
        st.session_state.sidebar_yt_out = "(Scroll to 'YouTube AI' tab for full summary.)"

    if st.session_state.sidebar_yt_out:
        st.caption("YouTube summary (snippet)")
        st.write(st.session_state.sidebar_yt_out)

# ========= Lightweight Oracle/OIC Error Playbook =========
PLAYBOOK = {
    "ORA-00942": [
        "Table or view does not exist.",
        "✅ Verify schema prefix (APPS.table vs. your schema).",
        "✅ Use fully-qualified names or ensure synonyms exist.",
        "✅ Check grants: GRANT SELECT/INSERT/UPDATE on required objects.",
        "✅ In Fusion SaaS: ensure the report/OTBI subject area includes this object."
    ],
    "ORA-06550": [
        "PL/SQL compilation error.",
        "✅ Inspect the next PLS-00xxx line for exact cause.",
        "✅ Check missing vars, wrong datatypes, or missing semicolons.",
        "✅ Confirm the object is compiled in the correct schema."
    ],
    "ORA-00001": [
        "Unique constraint violated.",
        "✅ Check for duplicate keys before INSERT/MERGE.",
        "✅ Use sequences/identity properly; deduplicate interface data.",
        "✅ Add safe upsert logic and composite keys where needed."
    ],
    "INSUFFICIENT FUNDS": [
        "Budget check failure.",
        "✅ Confirm control budget, ledger, and period are open.",
        "✅ Verify account combination is enabled.",
        "✅ Review consumption rules (commitment/obligation/actual) and funds available.",
        "✅ Use budget inquiry by segment and project/grant to see balances."
    ],
    "OIC-": [
        "OIC Integration error.",
        "✅ Open failed run → expand Activity Stream.",
        "✅ Check mapper nulls and optional XSD elements.",
        "✅ Re-authenticate connections; verify certs.",
        "✅ Add try/catch fault handler, log diagnostics."
    ]
}

def rule_based_fix(text: str) -> list[str]:
    t = (text or "").upper()
    hits = []
    for key, steps in PLAYBOOK.items():
        if key in t:
            hits.append((key, steps))
    if not hits:
        return [
            "⚙️ Generic triage:",
            "1) Capture full error stack (code + message + line).",
            "2) Identify module (GL, AP, AR, Projects, OIC, OTBI...).",
            "3) Reproduce with smallest input.",
            "4) Check roles/data-access (BU, Ledger, Project/Grant).",
            "5) Review recent setups and CVR rules.",
            "6) For interfaces: inspect staging/INT tables and invalid data."
        ]
    merged = []
    for k, steps in hits:
        merged.append(f"**Matched:** {k}")
        merged.extend(steps)
        merged.append("---")
    return merged

# ========= Optional LLM Helpers =========
def llm_fix_with_openai(prompt: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        sys = (
            "You are an Oracle Fusion + OIC + SQL/PLSQL senior support analyst. "
            "Return actionable, numbered steps. Keep it concise and accurate."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(OpenAI error: {e})"

def llm_fix_with_gemini(prompt: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        r = model.generate_content(
            "Act as Oracle Fusion/OIC support SME. Return precise steps.\n\n" + prompt
        )
        return r.text
    except Exception as e:
        return f"(Gemini error: {e})"

# ========= UI Tabs =========
tabs = st.tabs([
    "Overview", "Diagnose", "Analytics", "Security", "Optimize", "Integrations", "Voice Mode", "YouTube AI"
])

# --- Overview ---
with tabs[0]:
    st.title("🛠️ Oracle Fusion Omniverse")
    st.write(
        "Personal support console for Oracle ERP, OIC, and SQL/PLSQL. "
        "Paste errors, upload screenshots/logs, and get instant step-by-step fixes."
    )
    st.info("Works offline with a built-in playbook. Add API keys in the sidebar to enable AI reasoning.")
    st.markdown("**Build:** " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

    # Agent Console (router demo)
    st.markdown("---")
    st.subheader("🧠 Agent Console (router demo)")
    agent_q_main = st.text_input(
        "Ask anything (paste 1+ YouTube URLs to trigger YouTube; ORA-xxxx to trigger CrewAI)",
        placeholder="Summarize https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    if st.button("Run Agent"):
        s = route(agent_q_main)
        out = execute(s)
        st.code(out)

# --- Diagnose ---
with tabs[1]:
    st.header("🔍 Diagnose & Fix")
    err = st.text_area(
        "Paste error message / log / description",
        height=200,
        placeholder="Example: ORA-00942: table or view does not exist"
    )
    uploads = st.file_uploader(
        "Optional: attach screenshots/logs (txt/json/csv/png/jpg)",
        type=["txt", "json", "csv", "log", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    colA, colB, colC = st.columns(3)
    with colA:
        use_rulebook = st.checkbox("Use built-in playbook", value=True)
    with colB:
        use_openai = st.checkbox("Use OpenAI (if key set)", value=bool(OPENAI_API_KEY))
    with colC:
        use_gemini = st.checkbox("Use Gemini (if key set)", value=False)

    if st.button("Generate Fix Steps", type="primary"):
        if not err and not uploads:
            st.warning("Please paste an error or attach a file.")
        else:
            blob = err or ""
            for f in uploads or []:
                try:
                    if f.type.startswith("image/"):
                        blob += f"\n[image attached: {f.name}]"
                    else:
                        blob += "\n\n" + f.read().decode("utf-8", errors="ignore")
                except Exception:
                    blob += f"\n[attached file: {f.name}]"

            sections = []
            if use_rulebook:
                steps = rule_based_fix(blob)
                sections.append("### 🧭 Playbook Suggestions\n" + "\n".join(f"- {s}" for s in steps))
            if use_openai and OPENAI_API_KEY:
                ai = llm_fix_with_openai(blob)
                if ai:
                    sections.append("### 🤖 OpenAI\n" + ai)
            if use_gemini and GEMINI_API_KEY:
                g = llm_fix_with_gemini(blob)
                if g:
                    sections.append("### 🌟 Gemini\n" + g)

            st.markdown("---")
            st.subheader("Recommended Resolution")
            st.markdown("\n\n".join(sections))

# --- Analytics ---
with tabs[2]:
    st.header("📊 Quick Analytics (demo)")
    rng = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
    df = pd.DataFrame({"date": rng, "Incidents": (np.random.poisson(2, size=len(rng))).cumsum()}).set_index("date")
    st.line_chart(df)

# --- Security ---
with tabs[3]:
    st.header("🔐 Security Checklist")
    st.checkbox("Correct roles/data access (BU, Ledger, Project/Grant)")
    st.checkbox("Sensitive logs redacted before sharing")
    st.checkbox("API keys stored in Streamlit secrets (not hardcoded)")
    st.checkbox("OIC connections have valid certs/tokens")

# --- Optimize ---
with tabs[4]:
    st.header("⚡ Optimize")
    st.markdown("""
- Add OIC fault handlers with retry + DLQ  
- Pre-validate account combos before interface  
- Use idempotent upserts and deduplication  
- Cache reference data for faster triage  
    """)

# --- Integrations ---
with tabs[5]:
    st.header("🔗 Integrations (notes)")
    st.markdown("""
**JIRA**: webhook or token to create ticket from fix summary  
**Teams/Slack**: post steps via incoming webhook  
**Email**: send steps with SMTP (keep light for free hosting)  
    """)

# --- Voice Mode ---
with tabs[6]:
    st.header("🎙️ Voice Mode (placeholder)")
    st.info("Voice libraries are heavy; keep cloud app fast. Add Whisper locally if needed.")

# --- YouTube AI ---
with tabs[7]:
    import re

    st.header("📺 YouTube Summarizer + Deviations")

    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
    expected_topic = st.text_input(
        "Expected topic / keywords (optional)",
        placeholder="e.g., Oracle Grants budget consumption, funds check, ORA-00942"
    )
    max_minutes = st.slider("Analyze first N minutes", 2, 60, 20)

    def _video_id(u: str) -> Optional[str]:
        if not u:
            return None
        m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", u)
        return m.group(1) if m else None

    # ---- Backward-compatible transcript fetcher ----
    def get_best_transcript(vid: str) -> List[dict]:
        """
        Prefer manual/auto English; otherwise translate to English.
        Works with both new and older youtube-transcript-api versions.
        """
        from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

        # Newer versions: list_transcripts exists
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            try:
                listing = YouTubeTranscriptApi.list_transcripts(vid)
            except VideoUnavailable:
                raise RuntimeError("Video unavailable or private.")
            except TranscriptsDisabled:
                raise RuntimeError("Transcripts are disabled for this video.")
            except Exception as e:
                raise RuntimeError(f"Could not list transcripts: {e}")

            # Prefer manual English
            try:
                tr = listing.find_transcript(['en', 'en-US', 'en-GB'])
                return tr.fetch()
            except Exception:
                pass

            # Auto English
            try:
                tr = listing.find_transcript(['en'])
                if tr.is_generated:
                    return tr.fetch()
            except Exception:
                pass

            # First available → translate to English
            try:
                first = next(iter(listing))
                tr_en = first.translate('en')
                return tr_en.fetch()
            except StopIteration:
                raise RuntimeError("No transcripts exist for this video.")
            except NoTranscriptFound:
                raise RuntimeError("No transcript found in any language.")
            except Exception as e:
                raise RuntimeError(f"Failed to translate transcript to English: {e}")

        # Older versions (no list_transcripts)
        try:
            return YouTubeTranscriptApi.get_transcript(vid, languages=['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            pass
        except TranscriptsDisabled:
            raise RuntimeError("Transcripts are disabled for this video.")
        except VideoUnavailable:
            raise RuntimeError("Video unavailable or private.")
        except Exception:
            pass

        # Last attempt: any transcript (language unknown)
        try:
            return YouTubeTranscriptApi.get_transcript(vid)
        except Exception:
            raise RuntimeError("No transcript available for this video (or blocked/age-restricted).")

    def cut_by_minutes(trans: List[dict], minutes: int) -> List[dict]:
        limit = minutes * 60
        return [t for t in trans if t["start"] <= limit]

    def join_transcript(trans: List[dict]) -> str:
        return "\n".join(f'[{int(t["start"]//60):02d}:{int(t["start"]%60):02d}] {t["text"]}' for t in trans)

    def summarize_with_llm(text: str, topic_hint: str) -> Optional[str]:
        prompt = (
            "You are an expert meeting summarizer.\n"
            "Input is a YouTube transcript with [MM:SS] timestamps.\n\n"
            "Return THREE sections:\n"
            "1) **Concise Summary** (5–10 bullets)\n"
            "2) **Key Takeaways** (numbered)\n"
            "3) **Deviations / Off-topic Segments** — each as [MM:SS–MM:SS] + one-line reason.\n"
            f"Expected topic (if any): '{topic_hint}'.\n\n"
            "Transcript:\n" + text[:15000]
        )
        # OpenAI first
        try:
            if OPENAI_API_KEY:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"Be precise and faithful to the transcript."},
                              {"role":"user","content":prompt}],
                    temperature=0.2,
                )
                return r.choices[0].message.content
        except Exception as e:
            st.warning(f"OpenAI error: {e}")

        # Gemini fallback
        try:
            if GEMINI_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                out = model.generate_content(prompt)
                return out.text
        except Exception as e:
            st.warning(f"Gemini error: {e}")

        return None

    def heuristic_deviations(trans: List[dict], topic_hint: str) -> List[Tuple[str, str]]:
        if not topic_hint:
            return []
        import re as _re
        kws = {w.lower() for w in _re.findall(r"[A-Za-z0-9]+", topic_hint) if len(w) > 3}
        if not kws:
            return []
        wins = []
        bucket, start = [], None
        for t in trans:
            if start is None:
                start = t["start"]
            bucket.append(t)
            if t["start"] - start >= 45:
                text = " ".join(x["text"].lower() for x in bucket)
                hits = sum(1 for k in kws if k in text)
                if hits <= max(1, len(kws)//6):
                    wins.append((start, t["start"]))
                bucket, start = [], None
        out = []
        for s, e in wins[:12]:
            out.append((f"{int(s//60):02d}:{int(s%60):02d}-{int(e//60):02d}:{int(e%60):02d}",
                        "Low topic keyword density"))
        return out

    # --- Bind orchestrator to existing summarizer functions ---
    def _safe_ai_summary(vid_or_url: str) -> str:
        v = _video_id(vid_or_url) or vid_or_url
        try:
            trans = get_best_transcript(v)
        except Exception as e:
            return f"Transcript error: {e}"
        trans = cut_by_minutes(trans, max_minutes)
        text = join_transcript(trans)
        ai = summarize_with_llm(text, expected_topic or "")
        return ai or "AI summary unavailable."

    def _single_sum(url: str) -> str:
        return _safe_ai_summary(url)

    def _batch_sum(urls: List[str]) -> str:
        outs = []
        for u in urls:
            outs.append(f"--- {u} ---\n{_safe_ai_summary(u)}")
        return "\n\n".join(outs)

    bind_youtube_functions(_single_sum, _batch_sum)

    # --- UI action (main area) ---
    if st.button("Summarize Video", type="primary"):
        vid = _video_id(url)
        if not vid:
            st.error("Please paste a valid YouTube URL.")
        else:
            try:
                trans = get_best_transcript(vid)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            trans = cut_by_minutes(trans, max_minutes)
            text = join_transcript(trans)

            llm = summarize_with_llm(text, expected_topic or "")
            if llm:
                st.markdown("### ✅ AI Summary")
                st.markdown(llm)
            else:
                st.info("AI summary unavailable (no key or API error). Showing heuristic deviations only.")

            devs = heuristic_deviations(trans, expected_topic or "")
            if devs:
                st.markdown("### 🚥 Deviations (backup heuristic)")
                for span, why in devs:
                    st.write(f"- [{span}] {why}")

            with st.expander("Raw transcript (first minutes)"):
                st.code(text, language="text")

    # --- Fancy HTML UI embed (visual layer) ---
    import streamlit.components.v1 as components
    from pathlib import Path
    try:
        html_ui = Path("static/yt_agent.html").read_text(encoding="utf-8")
        components.html(html_ui, height=1200, scrolling=True)
    except Exception as e:
        st.info(f"Custom HTML UI not found ({e}). Create static/yt_agent.html to show the Tailwind interface.")
