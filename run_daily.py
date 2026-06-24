"""
Daily prediction writer -- football-data.org version.

Same job as before: fit the Dixon-Coles model on finished matches, predict the
upcoming ones, and UPSERT rows into your Supabase `predictions` table. The only
thing that changed vs the API-Football version is the data provider, so the
fetch + parse functions below are rewritten for football-data.org's v4 shape.

predict.py and your Supabase tables are UNCHANGED.

ENV VARS (set as GitHub Actions secrets / workflow env):
  FOOTBALL_DATA_TOKEN        your token from football-data.org/client/register
  SUPABASE_URL               https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY  the SERVICE-ROLE (secret) key -- bypasses RLS to write
  COMPETITION                competition code, e.g. 'PL' (Premier League)
  HISTORY_SEASONS            e.g. '2023,2024,2025'  (season = starting year)
  DAYS_AHEAD                 e.g. '3'  (use a big number like 60 for a first test)

Free plan = 10 requests/minute, top competitions free. We make only a few
requests per run and sleep between them to stay under the limit.
"""

from __future__ import annotations
import os
import re
import time
import datetime as dt

import pandas as pd
import requests

from predict import fit_model  # reuse the tested model

API_BASE = "https://api.football-data.org/v4"


# ---------------------------------------------------------------------------
# Pure helpers (no network) -- unit-tested at the bottom of this file
# ---------------------------------------------------------------------------
def parse_finished(api_json) -> pd.DataFrame:
    """football-data.org /matches response -> dataframe the model expects."""
    rows = []
    for m in api_json.get("matches", []):
        ft = (m.get("score") or {}).get("fullTime") or {}
        if ft.get("home") is None or ft.get("away") is None:
            continue  # not actually finished / no score
        rows.append({
            "Date": m["utcDate"],
            "HomeTeam": m["homeTeam"]["name"],
            "AwayTeam": m["awayTeam"]["name"],
            "FTHG": ft["home"],
            "FTAG": ft["away"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    return df


def parse_upcoming(api_json) -> list[dict]:
    """football-data.org /matches response -> list of upcoming matches."""
    league_name = (api_json.get("competition") or {}).get("name")
    out = []
    for m in api_json.get("matches", []):
        out.append({
            "fixture_id": m["id"],
            "kickoff": m["utcDate"],
            "league": league_name or (m.get("competition") or {}).get("name"),
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
        })
    return out


def slugify(home: str, away: str, kickoff_iso: str) -> str:
    date = kickoff_iso[:10]  # YYYY-MM-DD
    base = f"{home}-vs-{away}-{date}"
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


def prediction_row(model, fx: dict) -> dict | None:
    """One upcoming fixture -> one row for the `predictions` table.
    Returns None if a team is unknown to the model (e.g. just promoted)."""
    if fx["home"] not in model.attack or fx["away"] not in model.attack:
        return None
    p = model.predict(fx["home"], fx["away"])
    return {
        "fixture_id": fx["fixture_id"],
        "kickoff": fx["kickoff"],
        "league": fx["league"],
        "home": fx["home"],
        "away": fx["away"],
        "p_home": round(p["home"], 4),
        "p_draw": round(p["draw"], 4),
        "p_away": round(p["away"], 4),
        "over25": round(p["over25"], 4),
        "likely_home": p["likely_score"][0],
        "likely_away": p["likely_score"][1],
        "is_premium": False,
        "analysis": None,
        "slug": slugify(fx["home"], fx["away"], fx["kickoff"]),
    }


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def _get(params, token, competition):
    url = f"{API_BASE}/competitions/{competition}/matches"
    r = requests.get(url, params=params,
                     headers={"X-Auth-Token": token}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_history(competition, seasons, token) -> pd.DataFrame:
    frames = []
    for i, season in enumerate(seasons):
        if i:
            time.sleep(7)  # stay under 10 requests/min on the free plan
        try:
            data = _get({"season": season, "status": "FINISHED"}, token, competition)
            df = parse_finished(data)
            print(f"  season {season}: {len(df)} finished matches")
            frames.append(df)
        except requests.HTTPError as e:
            print(f"  season {season}: skipped ({e.response.status_code})")
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No history fetched. Try fewer/newer HISTORY_SEASONS.")
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def fetch_upcoming(competition, days_ahead, token) -> list[dict]:
    today = dt.date.today()
    data = _get({
        "status": "SCHEDULED",
        "dateFrom": today.isoformat(),
        "dateTo": (today + dt.timedelta(days=days_ahead)).isoformat(),
    }, token, competition)
    return parse_upcoming(data)


def upsert(rows: list[dict]):
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    sb.table("predictions").upsert(rows, on_conflict="fixture_id").execute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    token = os.environ["FOOTBALL_DATA_TOKEN"]
    competition = os.environ.get("COMPETITION", "PL")
    seasons = [int(s) for s in os.environ.get("HISTORY_SEASONS", "2023,2024,2025").split(",")]
    days_ahead = int(os.environ.get("DAYS_AHEAD", "3"))

    print(f"Competition: {competition}")
    print("Fetching history to fit the model...")
    hist = fetch_history(competition, seasons, token)
    print(f"Total finished matches: {len(hist)}")

    print("Fitting model...")
    model = fit_model(hist)

    print("Fetching upcoming fixtures...")
    time.sleep(7)
    fixtures = fetch_upcoming(competition, days_ahead, token)
    print(f"  {len(fixtures)} upcoming")

    rows = [r for fx in fixtures if (r := prediction_row(model, fx)) is not None]
    print(f"Writing {len(rows)} predictions to Supabase...")
    if rows:
        upsert(rows)
    print("Done.")


if __name__ == "__main__":
    main()
