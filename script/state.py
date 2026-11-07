"""Fetch a compact, team-only competition status.

Repository layout:
    .env
    script/auth.py
    script/state.py   <-- this file

Usage:
    python script/state.py                   # API -> stdout + data/state_status.json
    python script/state.py --input state.json  # local snapshot, no login
    python script/state.py --stdout-only       # JSON only, suitable for jq
"""

import argparse
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_URL = "https://final.html2026.csie.org"
ROOT_DIR = Path(__file__).resolve().parent.parent
TAIPEI = ZoneInfo("Asia/Taipei")


def local_time(timestamp):
    if not timestamp:
        return None
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return dt.astimezone(TAIPEI).isoformat(timespec="seconds")


def fetch_state():
    from auth import login  # Uses the existing login + Session cookies

    # Older auth.py versions print a success message; keep stdout valid JSON.
    with redirect_stdout(sys.stderr):
        session = login()

    with session:
        response = session.get(
            f"{BASE_URL}/api/state",
            headers={"Referer": f"{BASE_URL}/"},
            timeout=30,
        )
        response.raise_for_status()
        if "application/json" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("/api/state did not return JSON (session expired?)")
        state = response.json()
        if not isinstance(state, dict) or "team" not in state:
            raise RuntimeError("Unexpected /api/state response")
        return state


def submission_summary(state, phase):
    """Count visible submission *attempts*, including archived attempts."""
    active = [s for s in (state.get("mySubmissions") or []) if s.get("phase") == phase]
    history = [s for s in (state.get("submissionHistory") or []) if s.get("phase") == phase]
    records = active + history
    latest = max(records, key=lambda s: s.get("submittedAt") or "", default=None)

    compact_latest = None
    if latest is not None:
        compact_latest = {
            "file": latest.get("fileName"),
            "tradingDate": latest.get("tradingDate"),
            "submittedAt": local_time(latest.get("submittedAt")),
            "score": latest.get("score"),
            "status": "scored" if latest.get("score") is not None else "pending",
        }

    scored = sum(s.get("score") is not None for s in records)
    return {
        "latest": compact_latest,
        "total": len(records),
        "scored": scored,
        "pending": len(records) - scored,
    }


def quota_remaining(quota):
    """0 limits are treated as unspecified, not as zero remaining."""
    if not isinstance(quota, dict):
        return None
    remaining = []
    for prefix in ("team", "user"):
        limit = quota.get(prefix + "Limit")
        count = quota.get(prefix + "Count")
        try:
            limit = int(limit)
            count = int(count)
        except (TypeError, ValueError):
            continue
        if limit > 0:
            remaining.append(max(limit - count, 0))
    return min(remaining) if remaining else None


def phase2_availability(state, now=None):
    now = now or datetime.now(timezone.utc)
    active = state.get("activePhase2Day")
    date = active.get("tradingDate") if isinstance(active, dict) else active
    if not isinstance(date, str) or not date:
        return {
            "isOpen": False,
            "tradingDate": None,
            "deadline": None,
            "remaining": None,
            "canSubmit": False,
        }

    # Try to find the schedule for the server-advertised active trading date.
    schedule = next(
        (d for d in (state.get("phase2Days") or []) if d.get("tradingDate") == date),
        {},
    )
    if isinstance(active, dict):
        schedule = {**schedule, **active}

    is_open = bool(schedule.get("enabled", True))
    for field, is_start in (("openAt", True), ("deadline", False)):
        raw = schedule.get(field)
        if raw:
            boundary = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if is_start and now < boundary:
                is_open = False
            if not is_start and now >= boundary:
                is_open = False

    quota_data = state.get("uploadQuota") or {}
    quota = (quota_data.get("phase2ByDate") or {}).get(date)
    if quota is None:
        quota = quota_data.get("phase2")
    remaining = quota_remaining(quota)

    return {
        "isOpen": is_open,
        "tradingDate": date,
        "deadline": local_time(schedule.get("deadline")),
        "remaining": remaining,
        # Unknown quota means that the API has not confirmed remaining capacity.
        "canSubmit": False if not is_open else (remaining > 0 if remaining is not None else None),
    }


def team_ranks(state):
    team_id = state["team"]["id"]

    phase1_rows = state.get("phase1Leaderboard") or []
    phase1_sorted = sorted(
        phase1_rows,
        key=lambda r: r.get("score") if r.get("score") is not None else float("-inf"),
        reverse=True,
    )
    phase1_position = next(
        (i for i, r in enumerate(phase1_sorted, 1) if r.get("teamId") == team_id),
        None,
    )

    cumulative = state.get("phase2Cumulative") or []
    own_cumulative = next((r for r in cumulative if r.get("teamId") == team_id), None)

    mock_rows = (state.get("mockFinals") or {}).get("rows") or []
    own_mock = next((r for r in mock_rows if r.get("teamId") == team_id), None)

    return {
        "phase1": {
            "rank": phase1_position,
            "totalTeams": len(phase1_rows),
        },
        "phase2": {
            "mockFinalRank": own_mock.get("rank") if own_mock else None,
            "totalTeams": len(mock_rows),
            "cumulativeAverageRank": own_cumulative.get("averageRank") if own_cumulative else None,
            "cumulativeTotalRank": own_cumulative.get("totalRank") if own_cumulative else None,
        },
    }


def summarize(state, now=None):
    if not isinstance(state.get("team"), dict) or not state["team"].get("id"):
        raise ValueError("Missing team.id in state")
    return {
        "submissions": {
            "phase1": submission_summary(state, 1),
            "phase2": submission_summary(state, 2),
        },
        "phase2": phase2_availability(state, now=now),
        "ranks": team_ranks(state),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Use a saved state.json instead of the API")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "stat_data" / "state_status.json")
    parser.add_argument("--stdout-only", action="store_true", help="Print JSON but do not save a file")
    args = parser.parse_args()

    state = json.loads(args.input.read_text(encoding="utf-8")) if args.input else fetch_state()
    result = summarize(state)
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if not args.stdout_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_name(args.output.name + ".tmp")
        temp.write_text(output, encoding="utf-8")
        temp.replace(args.output)
    print(output, end="")


if __name__ == "__main__":
    main()
