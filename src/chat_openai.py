# app_f1_agent.py
import json
import re
from typing import Dict, Any, List, Optional

import requests
import streamlit as st
from mistralai import Mistral

# -----------------------------
# CONFIG
# -----------------------------
OPENF1_BASE = "https://api.openf1.org/v1"
DEFAULT_YEAR = 2024

# Put your Mistral key in .streamlit/secrets.toml as:
# [mistral]
# api_key = "sk-..."
MISTRAL_API_KEY = "api_Key"
MISTRAL_MODEL = "mistral-small"   # or "mistral-large-latest"

client = Mistral(api_key=MISTRAL_API_KEY)

# -----------------------------
# OPENF1 HELPERS
# -----------------------------
def fetch_openf1(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{OPENF1_BASE}/{endpoint}"
    r = requests.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def get_sessions(year: int, session_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """OpenF1 sessions for a given year. session_name examples: 'Race', 'Qualifying', 'Sprint'."""
    params = {"year": year}
    if session_name:
        params["session_name"] = session_name
    return fetch_openf1("sessions", params)

def get_session_results(session_key: int) -> List[Dict[str, Any]]:
    """Results for a given session (works for Race, Sprint, etc.)."""
    return fetch_openf1("session_results", {"session_key": session_key})

def get_starting_grid(session_key: int) -> List[Dict[str, Any]]:
    """Starting grid for a given Race session."""
    return fetch_openf1("starting_grid", {"session_key": session_key})

def get_drivers(year: int) -> List[Dict[str, Any]]:
    """Driver list by year."""
    return fetch_openf1("drivers", {"year": year})

# -----------------------------
# STANDINGS (Computed from OpenF1 session_results)
# -----------------------------
FIA_RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # top 10
FIA_SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]           # top 8

def compute_season_driver_standings(year: int) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate Race (and Sprint) session results into a season driver standings table.
    Notes:
      - Uses Race + Sprint sessions found in OpenF1.
      - Does not award fastest-lap bonus (requires explicit fastest-lap field; easy to add if present).
    Returns dict keyed by driver_number with {'name','team','points','wins',...}.
    """
    standings: Dict[str, Dict[str, Any]] = {}

    # Races
    race_sessions = get_sessions(year, session_name="Race")
    for sess in race_sessions:
        skey = sess["session_key"]
        results = get_session_results(skey)
        # Sort by final position if provided
        results_sorted = sorted(results, key=lambda x: x.get("position", 9999))

        # Assign race points
        for idx, row in enumerate(results_sorted):
            driver_num = str(row.get("driver_number"))
            if not driver_num:
                continue
            driver_name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
            team_name = row.get("team_name") or row.get("team") or ""
            pos = row.get("position")
            d = standings.setdefault(driver_num, {"name": driver_name, "team": team_name, "points": 0, "wins": 0, "podiums": 0})

            # points
            if isinstance(pos, int) and 1 <= pos <= len(FIA_RACE_POINTS):
                d["points"] += FIA_RACE_POINTS[pos - 1]

            # wins/podiums
            if pos == 1:
                d["wins"] += 1
                d["podiums"] += 1
            elif pos in (2, 3):
                d["podiums"] += 1

    # Sprints (optional but nice)
    try:
        sprint_sessions = get_sessions(year, session_name="Sprint")
        for sess in sprint_sessions:
            skey = sess["session_key"]
            results = get_session_results(skey)
            results_sorted = sorted(results, key=lambda x: x.get("position", 9999))
            for idx, row in enumerate(results_sorted):
                driver_num = str(row.get("driver_number"))
                if not driver_num:
                    continue
                driver_name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
                team_name = row.get("team_name") or row.get("team") or ""
                pos = row.get("position")
                d = standings.setdefault(driver_num, {"name": driver_name, "team": team_name, "points": 0, "wins": 0, "podiums": 0})
                if isinstance(pos, int) and 1 <= pos <= len(FIA_SPRINT_POINTS):
                    d["points"] += FIA_SPRINT_POINTS[pos - 1]
    except Exception:
        # If 'Sprint' sessions not present, just ignore.
        pass

    return standings

def standings_as_sorted_list(standings: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for dnum, rec in standings.items():
        rows.append({
            "driver_number": dnum,
            "driver": rec["name"],
            "team": rec["team"],
            "points": rec["points"],
            "wins": rec["wins"],
            "podiums": rec["podiums"],
        })
    return sorted(rows, key=lambda r: (-r["points"], -r["wins"], r["driver"]))

# -----------------------------
# LLM INTENT PARSING (Tool routing)
# -----------------------------
INTENT_SYSTEM = """You are a router that returns ONLY compact JSON.
Decide the user's INTENT for Formula 1 questions and extract slots.

Schema (JSON):
{
  "intent": "driver_standings | team_standings | race_winner | session_list | driver_info | raw_answer",
  "year": 2024,
  "grand_prix": null,           // e.g. "Bahrain" (for race_winner)
  "entity": null,               // e.g. "Max Verstappen" (for driver_info)
  "notes": null
}

Rules:
- If the user asks "Who is the best driver in 2024?", map to driver_standings with latest year (default 2024/2025 if mentioned).
- If the question is about which driver won a GP, use intent=race_winner with 'grand_prix' set.
- If asking “what sessions are there in YEAR?”, use session_list.
- If it's general like 'Tell me about Verstappen 2024', use driver_info with entity set.
- If you cannot determine a toolable intent, use raw_answer.

Return ONLY the JSON, no prose.
"""

def llm_intent(question: str) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": question}
    ]
    resp = client.chat.complete(model=MISTRAL_MODEL, messages=msgs)
    txt = resp.choices[0].message.content.strip()
    # Extract JSON (be defensive)
    m = re.search(r"\{.*\}", txt, flags=re.S)
    payload = json.loads(m.group(0)) if m else {"intent": "raw_answer", "year": DEFAULT_YEAR}
    if "year" not in payload or payload["year"] is None:
        payload["year"] = DEFAULT_YEAR
    return payload

# -----------------------------
# ANSWER COMPOSER
# -----------------------------
def compose_answer(question: str, context_str: str) -> str:
    """Let Mistral answer with the retrieved context appended."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an F1 assistant. Use the provided context exactly and avoid fabrications. "
                "Cite drivers/teams and numbers only if present in the context."
            ),
        },
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_str}"}
    ]
    resp = client.chat.complete(model=MISTRAL_MODEL, messages=messages)
    return resp.choices[0].message.content

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("F1 AI Agent")

tab1, tab2 = st.tabs(["Data Analysis", "F1 AI Agent"])

with tab1:
    st.write("Your existing data analysis content")

with tab2:
    st.header("Ask the F1 AI Agent")
    year = st.number_input("Season year", min_value=2018, max_value=2030, value=DEFAULT_YEAR, step=1)
    user_q = st.text_input("Ask about F1 (e.g., 'Who is the best driver in 2024?', 'Who won Bahrain GP 2024?')", key="f1_ai_agent_input")

    if user_q:
        try:
            intent = llm_intent(user_q)
            # Use explicit UI year if present
            if year and isinstance(year, int):
                intent["year"] = year

            intent_name = intent.get("intent", "raw_answer")
            context = ""
            table = None

            if intent_name == "driver_standings":
                st.info(f"Fetching computed driver standings for {intent['year']} from OpenF1 Race/Sprint results…")
                standings = compute_season_driver_standings(intent["year"])
                rows = standings_as_sorted_list(standings)
                if rows:
                    table = rows
                    leader = rows[0]
                    context = json.dumps({"standings_top5": rows[:5]}, ensure_ascii=False)
                else:
                    context = "No standings computed (no sessions found)."

            elif intent_name == "race_winner":
                gp = intent.get("grand_prix")
                st.info(f"Searching Race sessions for {intent['year']}…")
                sessions = get_sessions(intent["year"], session_name="Race")
                # naive match on meeting_name
                cand = [s for s in sessions if gp and gp.lower() in (s.get("meeting_name","").lower())]
                if not cand:
                    # Fallback: try GP in session name
                    cand = [s for s in sessions if gp and gp.lower() in (s.get("session_name","").lower())]
                if cand:
                    skey = cand[0]["session_key"]
                    results = get_session_results(skey)
                    results_sorted = sorted(results, key=lambda x: x.get("position", 9999))
                    winner = results_sorted[0] if results_sorted else {}
                    context = json.dumps({"race": cand[0], "winner": winner}, ensure_ascii=False)
                    table = results_sorted[:10]
                else:
                    context = f"No race found for grand_prix={gp}."

            elif intent_name == "session_list":
                sessions = get_sessions(intent["year"])
                context = json.dumps({"sessions_count": len(sessions)}, ensure_ascii=False)
                table = sessions

            elif intent_name == "driver_info":
                drivers = get_drivers(intent["year"])
                want = (intent.get("entity") or "").lower()
                match = [d for d in drivers if want and (want in f"{d.get('first_name','')} {d.get('last_name','')}".lower())]
                context = json.dumps({"driver_info": match}, ensure_ascii=False) if match else "Driver not found."
                table = match

            else:
                # raw answer with no tool call
                context = "No tool call; answer directly."

            # Show an optional table for transparency
            if table:
                st.caption("Context preview")
                st.dataframe(table, use_container_width=True)

            answer = compose_answer(user_q, context)
            st.write("**Answer**")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {e}")
