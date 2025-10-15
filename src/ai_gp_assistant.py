# ai_gp_assistant.py
from __future__ import annotations
import os, re, textwrap
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_LLM = bool(OPENAI_API_KEY)

OPENF1_BASE = "https://api.openf1.org/v1"

# ---------- OpenF1 tiny client ----------
def _get(url: str, params: Dict[str, Any]) -> pd.DataFrame:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data)

def list_sessions(year: int, country: Optional[str]=None, meeting_name: Optional[str]=None) -> pd.DataFrame:
    params = {"year": year}
    if country: params["country"] = country
    if meeting_name: params["meeting_name"] = meeting_name
    return _get(f"{OPENF1_BASE}/sessions", params)

def session_results(session_key: int) -> pd.DataFrame:
    # OpenF1 results endpoint (classification)
    return _get(f"{OPENF1_BASE}/results", {"session_key": session_key})

def laps(session_key: int, driver_number: Optional[int]=None) -> pd.DataFrame:
    params = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    return _get(f"{OPENF1_BASE}/laps", params)

def stints(session_key: int, driver_number: Optional[int]=None) -> pd.DataFrame:
    params = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    return _get(f"{OPENF1_BASE}/stints", params)

def pit_stops(session_key: int, driver_number: Optional[int]=None) -> pd.DataFrame:
    params = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    return _get(f"{OPENF1_BASE}/pit", params)

def track_status(session_key: int) -> pd.DataFrame:
    # safety car / VSC windows etc.
    return _get(f"{OPENF1_BASE}/track_status", {"session_key": session_key})

# ---------- Helpers ----------
def find_session_key(year: int, gp_query: str, session_name_hint: str="Race") -> Optional[int]:
    # gp_query can be country ("Singapore"), city, or partial meeting name
    sess = list_sessions(year)
    if sess.empty:
        return None
    q = gp_query.strip().lower()
    subset = sess[
        sess["meeting_name"].str.lower().str.contains(q, na=False)
        | sess["country"].str.lower().str.contains(q, na=False)
        | sess["location"].str.lower().str.contains(q, na=False)
    ]
    if subset.empty:
        return None
    # try to pick by session name (Race, Qualifying, Sprint, FP1/2/3)
    candidates = subset[subset["session_name"].str.contains(session_name_hint, case=False, na=False)]
    target = candidates if not candidates.empty else subset
    # pick most recent (highest session_key)
    return int(target.sort_values("session_key").iloc[-1]["session_key"])

def human_list(xs: List[str]) -> str:
    return ", ".join(xs[:-1]) + f" and {xs[-1]}" if len(xs) > 1 else (xs[0] if xs else "")

# ---------- Intent parsing ----------
INTENTS = {
    "tyres": re.compile(r"\b(tyres?|stints?|compound|pneus?)\b", re.I),
    "fastest_laps": re.compile(r"\b(fastest|best) lap(s)?\b", re.I),
    "results": re.compile(r"\b(result|classification|classement)\b", re.I),
    "pits": re.compile(r"\b(pit|stop|pitstop|arrêt)\b", re.I),
    "status": re.compile(r"\b(safety car|vsc|red flag|track status)\b", re.I),
}

def detect_intent(text: str) -> str:
    for name, rx in INTENTS.items():
        if rx.search(text):
            return name
    return "tyres"  # default to something useful if vague

# ---------- Formatters ----------
def summarize_tyres(df: pd.DataFrame) -> str:
    if df.empty:
        return "No tyre stint data found."
    # group by driver, list compounds + stint lengths
    out = []
    for drv, g in df.groupby(["driver_number", "driver_name"], dropna=False):
        g = g.sort_values("stint_number")
        parts = [f"{int(r['stint_number'])}: {r.get('compound','?')} ({int(r.get('lap_end', r.get('total_laps', 0)) - r.get('lap_start', 0))} laps)"
                 if pd.notna(r.get('lap_start')) and pd.notna(r.get('lap_end'))
                 else f"{int(r['stint_number'])}: {r.get('compound','?')}"
                 for _, r in g.iterrows()]
        out.append(f"{drv[1]} #{int(drv[0])} → " + " | ".join(parts))
    return "\n".join(out[:20]) + ("\n… (truncated)" if len(out) > 20 else "")

def summarize_fastest_laps(df: pd.DataFrame, top: int = 10) -> str:
    if df.empty:
        return "No lap data found."
    # choose per-driver best lap
    cols = [c for c in ["driver_number","driver_name","lap_number","lap_duration","is_pit_out_lap","is_pit_in_lap"] if c in df.columns]
    dd = df.copy()
    dd = dd[cols]
    dd = dd[~dd.get("is_pit_out_lap", False).astype(bool)]
    dd = dd[~dd.get("is_pit_in_lap", False).astype(bool)]
    best = dd.sort_values("lap_duration").groupby(["driver_number","driver_name"], as_index=False).first()
    best = best.sort_values("lap_duration").head(top)
    lines = [f"{int(r.driver_number):>2} {r.driver_name:<15} Lap {int(r.lap_number):<3}  {r.lap_duration:.3f}s"
             for _, r in best.iterrows()]
    return "Fastest laps per driver (top {0}):\n".format(top) + "\n".join(lines)

def summarize_results(df: pd.DataFrame, top: int = 10) -> str:
    if df.empty:
        return "No results data found."
    # expect position/driver_name/team_name columns
    keep = [c for c in ["position", "driver_number", "driver_name", "team_name", "status"] if c in df.columns]
    dd = df[keep].sort_values("position").head(top)
    lines = [f"P{int(r.position):<2}  #{int(r.driver_number):<2} {r.driver_name:<16}  {r.team_name}  ({r.status})"
             for _, r in dd.iterrows()]
    return "Top classification:\n" + "\n".join(lines)

def summarize_pits(df: pd.DataFrame) -> str:
    if df.empty:
        return "No pit stop data found."
    agg = df.groupby(["driver_number","driver_name"]).size().reset_index(name="stops")
    agg = agg.sort_values(["stops","driver_number"], ascending=[False, True])
    lines = [f"#{int(r.driver_number):<2} {r.driver_name:<16}  {int(r.stops)} stop(s)" for _, r in agg.iterrows()]
    return "Pit stops count:\n" + "\n".join(lines)

def summarize_status(df: pd.DataFrame) -> str:
    if df.empty:
        return "No track status events found."
    # Usually has status messages with time ranges (SC, VSC, Red Flag)
    counts = df["message"].value_counts().to_dict() if "message" in df.columns else {}
    if not counts:
        return "Track status present but not parseable."
    parts = [f"{k}: {v} event(s)" for k, v in counts.items()]
    return "Track status summary: " + "; ".join(parts)

# ---------- LLM optional ----------
def llm_refine(system: str, user: str, raw_answer: str) -> str:
    if not USE_LLM:
        return raw_answer
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        msg = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": raw_answer}
            ],
            temperature=0.2,
        )
        return msg.choices[0].message.content.strip()
    except Exception:
        return raw_answer

# ---------- Main entry ----------
def answer_question(text: str, year: int, gp_hint: str, session_hint: str="Race") -> Tuple[str, Dict[str, Any]]:
    """
    Returns (answer_text, debug_info)
    """
    intent = detect_intent(text)
    skey = find_session_key(year, gp_hint, session_hint or "Race")
    if not skey:
        return f"Sorry, I couldn’t find a session matching “{gp_hint} {session_hint}” in {year}. Try another GP or session name (FP1/FP2/FP3/Qualifying/Sprint/Race).", {"intent": intent}

    # dispatch
    if intent == "tyres":
        df = stints(skey)
        raw = summarize_tyres(df)
    elif intent == "fastest_laps":
        df = laps(skey)
        raw = summarize_fastest_laps(df)
    elif intent == "results":
        df = session_results(skey)
        raw = summarize_results(df)
    elif intent == "pits":
        df = pit_stops(skey)
        raw = summarize_pits(df)
    elif intent == "status":
        df = track_status(skey)
        raw = summarize_status(df)
    else:
        df = pd.DataFrame()
        raw = "Not sure what you want. Try asking about tyres, fastest laps, results, pit stops, or safety car."

    system = "You are a concise F1 race analyst. Use bullets and short lines. Keep numbers and units clear."
    refined = llm_refine(system, text, raw)
    return refined, {"intent": intent, "session_key": skey}
