from __future__ import annotations
from typing import Dict, Any
import pandas as pd

def results_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        year = int(race.get("season") or race.get("year") or 0)
        rnd  = int(race["round"])
        for res in race.get("Results", []):
            rows.append({
                "year": year,
                "round": rnd,
                "race_name": race["raceName"],
                "date": pd.to_datetime(race["date"], errors="coerce"),
                "circuit": race["Circuit"]["circuitName"],
                "position": int(res["position"]),
                "points": float(res.get("points", 0)),
                "driverId": res["Driver"]["driverId"],
                "driver": f"{res['Driver']['givenName']} {res['Driver']['familyName']}",
                "constructorId": res["Constructor"]["constructorId"],
                "team": res["Constructor"]["name"],
                "status": res.get("status",""),
            })
    df = pd.DataFrame(rows).sort_values(["year","round","position"]).reset_index(drop=True)
    return df

def sprint_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for r in races:
        season = int(r.get("season") or 0)
        round_ = int(r["round"])
        for s in r.get("SprintResults", []):
            rows.append({
                "year": season, "round": round_,
                "driverId": s["Driver"]["driverId"],
                "driver": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                "constructorId": s["Constructor"]["constructorId"],
                "team": s["Constructor"]["name"],
                "position": int(s["position"]),
                "points": float(s.get("points", 0)),
            })
    return pd.DataFrame(rows).sort_values(["year","round","position"]).reset_index(drop=True)

def qualifying_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for r in races:
        season = int(r.get("season") or 0)
        round_ = int(r["round"])
        for q in r.get("QualifyingResults", []):
            rows.append({
                "year": season,
                "round": round_,
                "driverId": q["Driver"]["driverId"],
                "driver": f"{q['Driver']['givenName']} {q['Driver']['familyName']}",
                "constructorId": q["Constructor"]["constructorId"],
                "team": q["Constructor"]["name"],
                "position": int(q["position"]),
                "Q1": q.get("Q1"), "Q2": q.get("Q2"), "Q3": q.get("Q3"),
            })
    return pd.DataFrame(rows).sort_values(["year","round","position"]).reset_index(drop=True)

def driver_standings_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    rows = []
    for sl in lists:
        year = int(sl.get("season") or 0)
        for pos in sl.get("DriverStandings", []):
            rows.append({
                "year": year,
                "position": int(pos["position"]),
                "points": float(pos.get("points", 0)),
                "wins": int(pos.get("wins", 0)),
                "driverId": pos["Driver"]["driverId"],
                "driver": f"{pos['Driver']['givenName']} {pos['Driver']['familyName']}",
                "constructorIds": [c["constructorId"] for c in pos.get("Constructors", [])],
            })
    return pd.DataFrame(rows).sort_values(["year","position"]).reset_index(drop=True)

def constructor_standings_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    rows = []
    for sl in lists:
        year = int(sl.get("season") or 0)
        for pos in sl.get("ConstructorStandings", []):
            rows.append({
                "year": year,
                "position": int(pos["position"]),
                "points": float(pos.get("points", 0)),
                "wins": int(pos.get("wins", 0)),
                "constructorId": pos["Constructor"]["constructorId"],
                "team": pos["Constructor"]["name"],
            })
    return pd.DataFrame(rows).sort_values(["year","position"]).reset_index(drop=True)
