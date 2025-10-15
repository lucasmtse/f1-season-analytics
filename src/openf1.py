# src/openf1.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
import requests

BASE = "https://api.openf1.org/v1"

_sess = requests.Session()
_sess.headers.update({"User-Agent": "f1-season-analytics/1.0"})

def q(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> List[dict]:
    r = _sess.get(f"{BASE}{path}", params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()
