"""evals/store.py

Persistent storage and statistics computation for call telemetry and evaluation scorecards.
Saves data into structured JSON files (logs/calls and logs/evals) and an indexed SQLite DB (logs/evals.db).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta

# IST = UTC+05:30
_IST = timezone(timedelta(hours=5, minutes=30))
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
CALLS_DIR = LOGS_DIR / "calls"
EVALS_DIR = LOGS_DIR / "evals"
DB_PATH = LOGS_DIR / "evals.db"


def _ensure_dirs():
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    EVALS_DIR.mkdir(parents=True, exist_ok=True)


def _init_db():
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS calls (
            call_id TEXT PRIMARY KEY,
            timestamp TEXT,
            duration_seconds REAL,
            total_turns INTEGER,
            avg_ttfa_ms REAL,
            min_ttfa_ms REAL,
            max_ttfa_ms REAL,
            p90_ttfa_ms REAL,
            avg_stt_ms REAL,
            avg_llm_ttft_ms REAL,
            avg_tts_ms REAL,
            total_prompt_tokens INTEGER,
            total_completion_tokens INTEGER,
            status TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id TEXT,
            turn_id INTEGER,
            user_query TEXT,
            agent_text TEXT,
            ttfa_ms REAL,
            stt_latency_ms REAL,
            llm_ttft_ms REAL,
            tts_latency_ms REAL,
            FOREIGN KEY(call_id) REFERENCES calls(call_id)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_call_id ON turns(call_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls(timestamp)")
    conn.commit()
    conn.close()


# Ensure DB initialized on module load
_init_db()


def compute_summary(turns: List[Dict[str, Any]], tokens: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Compute aggregated metrics (averages, min/max, P90) from a list of turn records."""
    if not turns:
        return {
            "total_turns": 0,
            "avg_ttfa_ms": 0.0,
            "min_ttfa_ms": 0.0,
            "max_ttfa_ms": 0.0,
            "p90_ttfa_ms": 0.0,
            "avg_stt_ms": 0.0,
            "avg_llm_ttft_ms": 0.0,
            "avg_tts_ms": 0.0,
            "total_prompt_tokens": tokens.get("prompt", 0) if tokens else 0,
            "total_completion_tokens": tokens.get("completion", 0) if tokens else 0,
        }

    ttfas = [t.get("ttfa_ms", 0.0) for t in turns if t.get("ttfa_ms", 0.0) > 0]
    stts = [t.get("stt_latency_ms", 0.0) for t in turns if t.get("stt_latency_ms", 0.0) > 0]
    ttfts = [t.get("llm_ttft_ms", 0.0) for t in turns if t.get("llm_ttft_ms", 0.0) > 0]
    ttss = [t.get("tts_latency_ms", 0.0) for t in turns if t.get("tts_latency_ms", 0.0) > 0]

    def _avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    def _min(lst):
        return round(min(lst), 1) if lst else 0.0

    def _max(lst):
        return round(max(lst), 1) if lst else 0.0

    def _p90(lst):
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        idx = int(len(sorted_lst) * 0.90)
        idx = min(idx, len(sorted_lst) - 1)
        return round(sorted_lst[idx], 1)

    return {
        "total_turns": len(turns),
        "avg_ttfa_ms": _avg(ttfas),
        "min_ttfa_ms": _min(ttfas),
        "max_ttfa_ms": _max(ttfas),
        "p90_ttfa_ms": _p90(ttfas),
        "avg_stt_ms": _avg(stts),
        "avg_llm_ttft_ms": _avg(ttfts),
        "avg_tts_ms": _avg(ttss),
        "total_prompt_tokens": tokens.get("prompt", 0) if tokens else 0,
        "total_completion_tokens": tokens.get("completion", 0) if tokens else 0,
    }


def save_call_metrics(
    call_id: str,
    turns: List[Dict[str, Any]],
    duration_seconds: float = 0.0,
    status: str = "completed",
    tokens: Optional[Dict[str, int]] = None,
    custom_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Save full turn telemetry and scorecard both to JSON files and SQLite."""
    _ensure_dirs()
    timestamp = custom_timestamp or datetime.now(_IST).isoformat()
    summary = compute_summary(turns, tokens)

    record = {
        "call_id": call_id,
        "timestamp": timestamp,
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
        "summary": summary,
        "turns": turns,
    }

    # Format timestamp for safe filename: YYYY-MM-DD_HH-MM-SS
    try:
        dt = datetime.fromisoformat(timestamp)
        file_ts = dt.strftime("%Y-%m-%d_%H-%M-%S")
    except Exception:
        file_ts = datetime.now(_IST).strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Save JSON files with timestamp prefix for chronological sorting
    call_file = CALLS_DIR / f"{file_ts}_{call_id}.json"
    scorecard_file = EVALS_DIR / f"{file_ts}_{call_id}_scorecard.json"

    with open(call_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    with open(scorecard_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # 2. Save into SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO calls (
                call_id, timestamp, duration_seconds, total_turns,
                avg_ttfa_ms, min_ttfa_ms, max_ttfa_ms, p90_ttfa_ms,
                avg_stt_ms, avg_llm_ttft_ms, avg_tts_ms,
                total_prompt_tokens, total_completion_tokens, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call_id,
                timestamp,
                record["duration_seconds"],
                summary["total_turns"],
                summary["avg_ttfa_ms"],
                summary["min_ttfa_ms"],
                summary["max_ttfa_ms"],
                summary["p90_ttfa_ms"],
                summary["avg_stt_ms"],
                summary["avg_llm_ttft_ms"],
                summary["avg_tts_ms"],
                summary["total_prompt_tokens"],
                summary["total_completion_tokens"],
                status,
            ),
        )

        cursor.execute("DELETE FROM turns WHERE call_id = ?", (call_id,))
        for t in turns:
            cursor.execute(
                """
                INSERT INTO turns (
                    call_id, turn_id, user_query, agent_text,
                    ttfa_ms, stt_latency_ms, llm_ttft_ms, tts_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    t.get("turn_id", 0),
                    t.get("user_query", ""),
                    t.get("agent_text", ""),
                    t.get("ttfa_ms", 0.0),
                    t.get("stt_latency_ms", 0.0),
                    t.get("llm_ttft_ms", 0.0),
                    t.get("tts_latency_ms", 0.0),
                ),
            )

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to save call eval to SQLite: {e}")

    return record


def get_latest_scorecard() -> Optional[Dict[str, Any]]:
    """Retrieve the most recent scorecard, preferring SQLite with JSON file fallback."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT call_id FROM calls ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            return get_scorecard(row[0])
    except Exception:
        pass

    # Fallback to filesystem
    if not EVALS_DIR.exists():
        return None
    files = sorted(
        [os.path.join(EVALS_DIR, f) for f in os.listdir(EVALS_DIR) if f.endswith("_scorecard.json")],
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def get_scorecard(call_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve scorecard for a specific call (supports timestamp prefix or exact match)."""
    scorecard_file = EVALS_DIR / f"{call_id}_scorecard.json"
    if scorecard_file.exists():
        with open(scorecard_file, "r", encoding="utf-8") as f:
            return json.load(f)
    matches = list(EVALS_DIR.glob(f"*{call_id}*_scorecard.json"))
    if matches:
        latest = max(matches, key=os.path.getmtime)
        with open(latest, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_telemetry(call_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve turn telemetry for a specific call (supports timestamp prefix or exact match)."""
    call_file = CALLS_DIR / f"{call_id}.json"
    if call_file.exists():
        with open(call_file, "r", encoding="utf-8") as f:
            return json.load(f)
    matches = [m for m in CALLS_DIR.glob(f"*{call_id}*.json") if m.name.endswith(".json")]
    if matches:
        latest = max(matches, key=os.path.getmtime)
        with open(latest, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_all_calls_summary(limit: int = 50) -> Dict[str, Any]:
    """Calculate aggregate stats across all stored calls."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_calls,
                AVG(avg_ttfa_ms) as global_avg_ttfa,
                AVG(avg_stt_ms) as global_avg_stt,
                AVG(avg_llm_ttft_ms) as global_avg_llm_ttft,
                AVG(avg_tts_ms) as global_avg_tts,
                SUM(total_turns) as global_total_turns,
                SUM(total_prompt_tokens) as global_prompt_tokens,
                SUM(total_completion_tokens) as global_completion_tokens
            FROM calls
            """
        )
        row = cursor.fetchone()

        cursor.execute(
            """
            SELECT call_id, timestamp, total_turns, avg_ttfa_ms, status
            FROM calls ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,),
        )
        recent_calls = [
            {
                "call_id": r[0],
                "timestamp": r[1],
                "total_turns": r[2],
                "avg_ttfa_ms": round(r[3], 1) if r[3] else 0.0,
                "status": r[4],
            }
            for r in cursor.fetchall()
        ]
        conn.close()

        return {
            "total_calls": row[0] or 0,
            "global_avg_ttfa_ms": round(row[1] or 0.0, 1),
            "global_avg_stt_ms": round(row[2] or 0.0, 1),
            "global_avg_llm_ttft_ms": round(row[3] or 0.0, 1),
            "global_avg_tts_ms": round(row[4] or 0.0, 1),
            "global_total_turns": row[5] or 0,
            "total_prompt_tokens": row[6] or 0,
            "total_completion_tokens": row[7] or 0,
            "recent_calls": recent_calls,
        }
    except Exception as e:
        return {"error": str(e), "total_calls": 0, "recent_calls": []}
