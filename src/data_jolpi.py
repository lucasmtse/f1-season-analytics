from __future__ import annotations
import json, time, random
from pathlib import Path
from typing import Dict, Any, Optional
import requests

BASE = "https://api.jolpi.ca/ergast/f1"
RAW = Path("data/raw/jolpi")
RAW.mkdir(parents=True, exist_ok=True)

sess = requests.Session()
sess.headers.update({"User-Agent": "f1-season-analytics/1.0"})

def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def cache_path(name: str) -> Path:
    safe = name.strip('/').replace('/', '_').replace('?', '_').replace('&', '_').replace('=', '_')
    return RAW / f"{safe}.json"

def get_paged(endpoint: str, params: Optional[Dict[str, Any]] = None,
              limit: int = 500, sleep: float = 0.5,
              max_retries: int = 6, backoff_base: float = 1.8,
              use_cache: bool = True) -> Dict[str, Any]:
    """
    Jolpi/Ergast pagination with rate-limit handling (429) and retries.
    Returns merged MRData with concatenated table rows and caches to disk.
    """
    params = dict(params or {})
    params["limit"] = limit

    cpath = cache_path(f"{endpoint}?limit={limit}")
    if use_cache and cpath.exists():
        try:
            return load_json(cpath)
        except Exception:
            pass

    def _request_with_retry(offset: int) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                r = sess.get(f"{BASE}{endpoint}", params={**params, "offset": offset}, timeout=30)
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    wait_s = float(ra) if ra and ra.isdigit() else backoff_base ** attempt
                    wait_s += random.uniform(0, 0.5)
                    time.sleep(wait_s)
                    continue
                if 500 <= r.status_code < 600:
                    wait_s = backoff_base ** attempt + random.uniform(0, 0.5)
                    time.sleep(wait_s)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                wait_s = backoff_base ** attempt + random.uniform(0, 0.5)
                time.sleep(wait_s)
                continue
        r = sess.get(f"{BASE}{endpoint}", params={**params, "offset": offset}, timeout=30)
        r.raise_for_status()
        return r.json()

    data = _request_with_retry(0)
    mr = data.get("MRData", {})
    table_key = next((k for k in ("RaceTable","StandingsTable","CircuitTable","DriverTable",
                                  "ConstructorTable","StatusTable","LapTable") if k in mr), None)
    if not table_key:
        save_json(cpath, data)
        return data
    tbl = mr[table_key]
    rows_key = next((k for k in ("Races","StandingsLists","Circuits","Drivers",
                                 "Constructors","Status","Laps") if k in tbl), None)
    if not rows_key:
        save_json(cpath, data)
        return data

    merged = {"MRData": {**mr, table_key: {**tbl, rows_key: []}}}
    merged["MRData"][table_key][rows_key].extend(tbl.get(rows_key, []))

    total = int(mr.get("total", 0))
    lim   = int(mr.get("limit", limit))
    offset = lim

    while offset < total:
        time.sleep(sleep)
        page = _request_with_retry(offset)
        rows = page.get("MRData", {}).get(table_key, {}).get(rows_key, [])
        merged["MRData"][table_key][rows_key].extend(rows)
        offset += lim

    merged["MRData"]["total"] = str(len(merged["MRData"][table_key][rows_key]))
    merged["MRData"]["offset"] = "0"
    merged["MRData"]["limit"] = str(lim)

    save_json(cpath, merged)
    return merged
