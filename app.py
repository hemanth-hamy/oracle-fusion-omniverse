import os
import io
import json
import textwrap
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Oracle Fusion Omniverse",
    page_icon="🛠️",
    layout="wide"
)

# ---------- Helper: Secrets / Keys ----------
def get_key(name: str) -> str | None:
    # From Streamlit Secrets or ENV (both supported on Streamlit Cloud)
    return st.secrets.get(name) if name in st.secrets else os.getenv(name)

OPENAI_API_KEY = get_key("OPENAI_API_KEY") or get_key("OPENAI_KEY")
GEMINI_API_KEY = get_key("GEMINI_API_KEY") or get_key("GOOGLE_API_KEY")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔧 Settings")
    st.caption("Keys here are optional—local rule-based fixes work without them.")
    openai_in = st.text_input("OpenAI API Key (optional)", type="password", value=OPENAI_API_KEY or "")
    gemini_in = st.text_input("Google Gemini API Key (optional)", type="password", value=GEMINI_API_KEY or "")
    if openai_in:
        os.environ["OPENAI_API_KEY"] = openai_in
        OPENAI_API_KEY = openai_in
    if gemini_in:
        os.environ["GEMINI_API_KEY"] = gemini_in
        GEMINI_API_KEY = gemini_in

    st.markdown("---")
    st.write("**Status**")
    st.write(f"OpenAI: {'✅' if OPENAI_API_KEY else '—'}  |  Gemini: {'✅' if GEMINI_API_KEY else '—'}")

    st.markdown("---")
    st.caption("Tip: set secrets in Streamlit → App → **Settings → Secrets**:")
    st.code(
        "OPENAI_API_KEY = 'sk-...'\nGEMINI_API_KEY = '...' ",
        language="bash"
    )

# ---------- Lightweight Oracle/OIC Error Playbook ----------
PLAYBOOK = {
    # Oracle DB / PL/SQL
    "ORA-00942": [
        "Table or view does not exist.",
        "✅ Verify schema prefix (e.g., APPS.table vs. your schema).",
        "✅ Confirm synonyms exist or use fully-qualified name.",
        "✅ Check object grants: GRANT SELECT/INSERT/UPDATE as needed.",
        "✅ In Fusion SaaS reports: ensure subject area or OTBI data model includes the object.",
    ],
    "ORA-06550": [
        "PL/SQL compilation error.",
        "✅ Open the next error line (PLS-00xxx) to pinpoint the exact issue.",
        "✅ Check missing variables, mismatched datatypes, or semicolons.",
        "✅ Ensure functions/procedures are created in the right schema.",
    ],
    "ORA-00001": [
        "Unique constraint violated.",
        "✅ Check duplicate key values before insert/merge.",
        "✅ Consider seq/identity usage and data dedup.",
        "✅ For interface tables, add a unique composite key or handle upserts safely.",
    ],
    # Budgetary control / Grants examples
    "INSUFFICIENT FUNDS": [
        "Budget check failure.",
        "✅ Confirm control budget, ledger, and period are open.",
        "✅ Verify account combination is enabled.",
        "✅ Review consumption rules (commitment/obligation/actual) and funds available.",
        "✅ Use budget inquiry to see balances by segment and project/grant.",
    ],
    # OIC / Integration examples
    "OIC-": [
        "OIC Integration error.",
        "✅ Open the failed run log and expand 'Activity Stream'.",
        "✅ Check mapper null values and XSD optional elements.",
        "✅ Re-authenticate connections; verify certificates.",
        "✅ Add try/catch fault handler and write to a diagnostic log.",
    ],
}

def rule_based_fix(text: str) -> list[str]:
    """Return step-by-step guidance from the playbook based on error text."""
    t = text.upper()
    hits = []
    for key, steps in PLAYBOOK.items():
        if key in t:
            hits.append((key, steps))
    # If nothing matched, return a generic triage
    if not hits:
        return [
            "⚙️ Generic triage:",
            "1) Capture full error stack (code + message + line).",
            "2) Identify module (GL, AP, AR, Projects, OIC, OTBI, etc.).",
            "3) Reproduce with smallest input.",
            "4) Check roles/data-access security for the user.",
            "5) Review recent setups (ledger, BU, grants, projects, cross-validations).",
            "6) If interface: inspect staging/INT tables and invalid data.",
        ]
    merged = []
    for k, steps in hits:
        merged.append(f"**Matched:** {k}")
        merged.extend(steps)
        merged.append("---")
    return merged

# ---------- Optional LLM helpers (import only if keys exist) ----------
def llm_fix_with_openai(prompt: str) -> str | None:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        sys = (
            "You are an Oracle Fusion + OIC + SQL/PLSQL senior support analyst. "
            "Return actionable, numbered steps. Keep it concise and accurate. "
            "If the user pasted logs, infer root cause and remediation."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(OpenAI fallback used due to error: {e})"

def llm_fix_with_gemini(prompt: str) -> str | None:
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
        return f"(Gemini fallback used due to error: {e})"

# ---------- UI Tabs ----------
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
        type=["txt","json","csv","log","png","jpg","jpeg"],
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
            # Aggregate text
            blob = err or ""
            for f in uploads or []:
                try:
                    if f.type.startswith("image/"):
                        blob += f"\n[image attached: {f.name}]"
                    else:
                        blob += "\n\n" + f.read().decode("utf-8", errors="ignore")
                except Exception:
                    blob += f"\n[attached file: {f.name}]"

            # Start with rule-based
            out_sections = []
            if use_rulebook:
                steps = rule_based_fix(blob)
                out_sections.append("### 🧭 Playbook Suggestions\n" + "\n".join(f"- {s}" for s in steps))

            # OpenAI
            if use_openai and OPENAI_API_KEY:
                ai = llm_fix_with_openai(blob)
                if ai:
                    out_sections.append("### 🤖 OpenAI\n" + ai)

            # Gemini
            if use_gemini and GEMINI_API_KEY:
                g = llm_fix_with_gemini(blob)
                if g:
                    out_sections.append("### 🌟 Gemini\n" + g)

            st.markdown("---")
            st.subheader("Recommended Resolution")
            st.markdown("\n\n".join(out_sections))

# --- Analytics ---
with tabs[2]:
    st.header("📊 Quick Analytics (demo)")
    rng = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
    df = pd.DataFrame({
        "date": rng,
        "Incidents": (np.random.poisson(2, size=len(rng))).cumsum()
    })
    df = df.set_index("date")
    st.line_chart(df)

# --- Security ---
with tabs[3]:
    st.header("🔐 Security Checklist")
    st.checkbox("Users have correct roles / data access (BU, Ledger, Project/Grant).")
    st.checkbox("Sensitive logs redacted before sharing.")
    st.checkbox("API keys stored in Streamlit secrets (not hardcoded).")
    st.checkbox("Connections (OIC) use latest certs and tokens.")

# --- Optimize ---
with tabs[4]:
    st.header("⚡ Optimize")
    st.markdown("""
- Add fault handlers in OIC with retry + DLQ pattern  
- For GL/Grants: validate account combinations before interface  
- Use incremental loads, dedup keys, and idempotent upserts  
- Cache reference data in-memory for faster triage  
    """)

# --- Integrations ---
with tabs[5]:
    st.header("🔗 Integrations (readme)")
    st.markdown("""
**JIRA:** Use a webhook or API token to create a ticket from a fix summary.  
**Teams/Slack:** Post fix steps to a channel via incoming webhook.  
**Email:** Send the steps using a lightweight SMTP service.  
(We keep code light here for free hosting; enterprise connectors can be added later.)
    """)

# --- Voice Mode ---
with tabs[6]:
    st.header("🎙️ Voice Mode (placeholder)")
    st.info("Voice libraries are heavy; we keep the Cloud app fast. For local use, add Whisper/pyttsx3 later.")

# --- YouTube AI ---
with tabs[7]:
    st.header("📺 YouTube AI (placeholder)")
    st.info("Paste a link → we can summarize with OpenAI/Gemini if keys are set. Coming as a light add-on.")
