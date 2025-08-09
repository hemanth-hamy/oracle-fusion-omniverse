import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import re
from fastapi.responses import HTMLResponse
from pathlib import Path

# Transcript libs
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# Optional fallback
USE_YTDLP = os.getenv("USE_YTDLP", "0") in ("1", "true", "True")

# Optional LLMs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="YouTube Summarizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Models ---------
class TranscriptRequest(BaseModel):
    url_or_id: str
    minutes: int = 20

class SummarizeRequest(BaseModel):
    transcript: str
    topic_hint: Optional[str] = ""
    minutes: int = 20

class YoutubeSummarizeRequest(BaseModel):
    url_or_id: str
    topic_hint: Optional[str] = ""
    minutes: int = 20

# --------- Helpers ---------
def extract_video_id(url_or_id: str) -> Optional[str]:
    """Handles watch, youtu.be, shorts, or raw ID."""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id or ""):
        return url_or_id
    if not url_or_id:
        return None
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url_or_id)
    return m.group(1) if m else None

def ytdlp_fallback(vid: str) -> Optional[List[Dict[str, Any]]]:
    if not USE_YTDLP:
        return None
    try:
        import yt_dlp, webvtt, tempfile, os as _os
        url_full = f"https://www.youtube.com/watch?v={vid}"
        tmpdir = tempfile.mkdtemp()
        outtmpl = _os.path.join(tmpdir, "%(id)s")
        opts = {
            "skip_download": True,
            "quiet": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "writeautomaticsub": True,
            "writesubtitles": True,
            "outtmpl": outtmpl,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url_full, download=True)
        vtt_file = None
        for fname in _os.listdir(tmpdir):
            if fname.endswith(".vtt"):
                vtt_file = _os.path.join(tmpdir, fname)
                break
        if not vtt_file:
            return None

        items: List[Dict[str, Any]] = []
        for cue in webvtt.read(vtt_file):
            def to_sec(ts: str) -> int:
                h, m, s = ts.replace(",", ".").split(":")
                return int(float(h) * 3600 + float(m) * 60 + float(s))
            items.append({"start": to_sec(cue.start), "text": cue.text})
        return items or None
    except Exception:
        return None

def get_transcript(vid: str) -> List[Dict[str, Any]]:
    # Newer API style
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        try:
            listing = YouTubeTranscriptApi.list_transcripts(vid)
        except VideoUnavailable:
            raise HTTPException(404, "Video unavailable or private.")
        except TranscriptsDisabled:
            raise HTTPException(404, "Transcripts are disabled for this video.")
        except Exception as e:
            raise HTTPException(500, f"Could not list transcripts: {e}")

        # Prefer manual English
        try:
            tr = listing.find_transcript(["en", "en-US", "en-GB"])
            return tr.fetch()
        except Exception:
            pass
        # Auto English
        try:
            tr = listing.find_transcript(["en"])
            if tr.is_generated:
                return tr.fetch()
        except Exception:
            pass
        # Translate first available
        try:
            first = next(iter(listing))
            return first.translate("en").fetch()
        except StopIteration:
            raise HTTPException(404, "No transcripts exist for this video.")
        except NoTranscriptFound:
            raise HTTPException(404, "No transcript found in any language.")
        except Exception as e:
            raise HTTPException(500, f"Failed to translate transcript: {e}")

    # Older API style
    try:
        return YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US", "en-GB"])
    except NoTranscriptFound:
        pass
    except TranscriptsDisabled:
        raise HTTPException(404, "Transcripts are disabled for this video.")
    except VideoUnavailable:
        raise HTTPException(404, "Video unavailable or private.")
    except Exception:
        pass

    # Last attempt
    try:
        return YouTubeTranscriptApi.get_transcript(vid)
    except Exception:
        alt = ytdlp_fallback(vid)
        if alt:
            return alt
        raise HTTPException(404, "No transcript available (or blocked/age-restricted).")

def cut_by_minutes(trans: List[Dict[str, Any]], minutes: int) -> List[Dict[str, Any]]:
    lim = max(1, minutes) * 60
    return [t for t in trans if t.get("start", 0) <= lim]

def transcript_to_text(trans: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for t in trans:
        s = int(t.get("start", 0))
        out.append(f"[{s//60:02d}:{s%60:02d}] {t.get('text','')}")
    return "\n".join(out)

def summarize_openai(text: str, topic_hint: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "You are an expert meeting summarizer.\n"
            "Input is a YouTube transcript with [MM:SS] timestamps.\n\n"
            "Return THREE sections:\n"
            "1) **Concise Summary** (5–10 bullets)\n"
            "2) **Key Takeaways** (numbered)\n"
            "3) **Deviations / Off-topic Segments** — each as [MM:SS–MM:SS] + one-line reason.\n"
            f"Expected topic (if any): '{topic_hint}'.\n\n"
            f"Transcript:\n{text[:15000]}"
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be precise and faithful to the transcript."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"(OpenAI error: {e})"

def summarize_gemini(text: str, topic_hint: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an expert meeting summarizer.\n"
            "Input is a YouTube transcript with [MM:SS] timestamps.\n\n"
            "Return THREE sections:\n"
            "1) **Concise Summary** (5–10 bullets)\n"
            "2) **Key Takeaways** (numbered)\n"
            "3) **Deviations / Off-topic Segments** — each as [MM:SS–MM:SS] + one-line reason.\n"
            f"Expected topic (if any): '{topic_hint}'.\n\n"
            f"Transcript:\n{text[:15000]}"
        )
        r = model.generate_content(prompt)
        return r.text
    except Exception as e:
        return f"(Gemini error: {e})"

def summarize(text: str, topic_hint: str) -> str:
    # Prefer OpenAI; fallback to Gemini; else a stub
    out = summarize_openai(text, topic_hint)
    if out: return out
    out = summarize_gemini(text, topic_hint)
    if out: return out
    return (
        "AI summary unavailable (no key or API error).\n\n"
        "Use OPENAI_API_KEY or GEMINI_API_KEY to enable AI summary."
    )

def heuristic_deviations(trans: List[Dict[str, Any]], topic_hint: str) -> List[str]:
    if not topic_hint:
        return []
    kws = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", topic_hint) if len(w) > 3}
    if not kws:
        return []
    wins, bucket, start = [], [], None
    for t in trans:
        if start is None:
            start = t["start"]
        bucket.append(t)
        if t["start"] - start >= 45:
            text = " ".join(x.get("text", "").lower() for x in bucket)
            hits = sum(1 for k in kws if k in text)
            if hits <= max(1, len(kws) // 6):
                wins.append((start, t["start"]))
            bucket, start = [], None
    out = []
    for s, e in wins[:12]:
        out.append(f"[{s//60:02d}:{s%60:02d}-{e//60:02d}:{e%60:02d}] low topic relevance")
    return out

# --------- API ---------
@app.get("/", response_class=HTMLResponse)
def root_page():
    html_path = Path(__file__).with_name("yt_agent.html")
    if not html_path.exists():
        return HTMLResponse("<h1>Missing yt_agent.html</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "utc": datetime.utcnow().isoformat() + "Z",
        "use_ytdlp": USE_YTDLP,
        "openai": bool(OPENAI_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
    }

@app.post("/api/transcript")
def api_transcript(req: TranscriptRequest):
    vid = extract_video_id(req.url_or_id)
    if not vid:
        raise HTTPException(400, "Invalid YouTube URL or ID.")
    trans = get_transcript(vid)
    trans = cut_by_minutes(trans, req.minutes)
    return {"ok": True, "video_id": vid, "items": trans, "text": transcript_to_text(trans)}

@app.post("/api/summarize")
def api_summarize(req: SummarizeRequest):
    if not req.transcript.strip():
        raise HTTPException(400, "Empty transcript.")
    text = req.transcript.strip()
    out = summarize(text, req.topic_hint or "")
    return {"ok": True, "summary": out}

@app.post("/api/youtube")
def api_youtube(req: YoutubeSummarizeRequest):
    vid = extract_video_id(req.url_or_id)
    if not vid:
        raise HTTPException(400, "Invalid YouTube URL or ID.")
    trans = get_transcript(vid)
    trans = cut_by_minutes(trans, req.minutes)
    text = transcript_to_text(trans)
    summary = summarize(text, req.topic_hint or "")
    deviations = heuristic_deviations(trans, req.topic_hint or "")
    return {
        "ok": True,
        "video_id": vid,
        "minutes": req.minutes,
        "summary": summary,
        "deviations": deviations,
        "transcript": text,
    }
