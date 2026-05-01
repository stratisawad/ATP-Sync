"""
Standalone script to load odds and rankings into Supabase.
Matches and stats are already loaded — this only does odds + rankings.

Two key matching fixes vs the original:
1. FUZZY NAME MATCHING: tennis-data.co.uk uses "Last F." while Sackmann uses
   "First Last". We convert all DB names to "Last F." format and use rapidfuzz
   to handle spelling variants (accents, hyphens, apostrophes, multi-part names).

2. DATE-RANGE MATCH LOOKUP: Sackmann stores tournament-start dates, not actual
   match dates. tennis-data.co.uk has actual dates. We match when the actual
   date falls within [tourney_start, tourney_start + 14 days].
"""
import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import requests
from io import BytesIO
from dotenv import load_dotenv
from supabase import create_client, Client
from rapidfuzz import process as fz_process, fuzz

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
DATA_DIR     = "./tennis_atp"
BATCH_SIZE   = 200
YEARS        = list(range(2015, 2025))
FUZZY_CUTOFF = 85   # minimum similarity score (0-100) to accept a fuzzy match

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BOOKMAKERS = {
    "B365W": ("Bet365",   "winner"),
    "B365L": ("Bet365",   "loser"),
    "PSW":   ("Pinnacle", "winner"),
    "PSL":   ("Pinnacle", "loser"),
}


def si(v):
    try:
        return int(v) if pd.notna(v) else None
    except Exception:
        return None


def paginate_all(table, select, order=None):
    rows, offset = [], 0
    while True:
        q = supabase.table(table).select(select)
        if order:
            q = q.order(order)
        result = q.range(offset, offset + 999).execute()
        rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    return rows


def batch_upsert(table, rows, on_conflict):
    if not rows:
        return
    seen, deduped = set(), []
    for r in rows:
        key = tuple(r[k] for k in on_conflict.split(","))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    for i in range(0, len(deduped), BATCH_SIZE):
        supabase.table(table).upsert(deduped[i:i + BATCH_SIZE], on_conflict=on_conflict).execute()


def batch_insert(table, rows):
    if not rows:
        return
    for i in range(0, len(rows), BATCH_SIZE):
        supabase.table(table).insert(rows[i:i + BATCH_SIZE]).execute()


def normalize(s: str) -> str:
    """Lowercase, strip accents, remove punctuation for comparison."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return s.strip()


def to_short(fullname: str) -> str:
    """Convert 'First [Middle] Last' → 'Last F.' matching tennis-data format."""
    parts = fullname.strip().split()
    if len(parts) == 1:
        return parts[0]
    # First token = first name, rest = last name (handles "Alex De Minaur" → "De Minaur A.")
    first_initial = parts[0][0].upper() + "."
    last = " ".join(parts[1:])
    return f"{last} {first_initial}"


# ── Load player map ───────────────────────────────────────────────────────────
print("Loading players from DB ...")
all_players = paginate_all("players", "player_id,name")
player_map  = {r["name"]: r["player_id"] for r in all_players}
print(f"  {len(player_map)} players")

# Build two lookup structures:
# 1. short_to_pid: "De Minaur A." -> player_id  (for exact/near-exact lookup)
# 2. norm_short_list: [(normalized_short, player_id)] for rapidfuzz
short_to_pid: dict[str, int]  = {}
norm_short_list: list[tuple]  = []   # (normalized_short, player_id, original_short)

for fullname, pid in player_map.items():
    short = to_short(fullname)
    short_to_pid[short] = pid
    norm_short_list.append((normalize(short), pid, short))

norm_short_keys = [x[0] for x in norm_short_list]  # for rapidfuzz choices


def resolve_player(odds_name: str):
    odds_name = odds_name.strip()
    if not odds_name or odds_name == "nan":
        return None

    # 1. Exact match on "Last F." form
    if odds_name in short_to_pid:
        return short_to_pid[odds_name]

    # 2. Exact match after normalization
    norm_odds = normalize(odds_name)
    for norm, pid, _ in norm_short_list:
        if norm == norm_odds:
            return pid

    # 3. rapidfuzz fuzzy match on normalized form
    result = fz_process.extractOne(
        norm_odds, norm_short_keys,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=FUZZY_CUTOFF,
    )
    if result:
        idx = norm_short_keys.index(result[0])
        return norm_short_list[idx][1]

    return None


# ── Load match lookup: (winner_id, loser_id) → [(match_id, tourney_start)] ───
# Sackmann tourney_date = Monday tournament start, not actual match date.
# tennis-data.co.uk has actual dates. We accept when:
#   tourney_start <= actual_date <= tourney_start + 14
print("Loading matches from DB ...")
all_matches = paginate_all("matches", "match_id,match_date,winner_id,loser_id")
match_lookup_by_pair: dict[tuple, list] = defaultdict(list)
for m in all_matches:
    if m["winner_id"] and m["loser_id"] and m["match_date"]:
        key = (m["winner_id"], m["loser_id"])
        match_lookup_by_pair[key].append((m["match_id"], m["match_date"]))
print(f"  {len(all_matches)} matches, {len(match_lookup_by_pair)} unique pairs")


def find_match(w_id: int, l_id: int, actual_date_str: str):
    candidates = match_lookup_by_pair.get((w_id, l_id), [])
    if not candidates:
        return None
    actual_dt = datetime.strptime(actual_date_str, "%Y-%m-%d")
    for match_id, tourney_start in candidates:
        start_dt = datetime.strptime(tourney_start, "%Y-%m-%d")
        if timedelta(0) <= (actual_dt - start_dt) <= timedelta(days=14):
            return match_id
    return None


# ── Odds ──────────────────────────────────────────────────────────────────────
print("\nDownloading odds ...")
odds_rows = []

import sys
test_year = int(sys.argv[1]) if len(sys.argv) > 1 else None
years_to_run = [test_year] if test_year else YEARS

for year in years_to_run:
    url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        odf = pd.read_excel(BytesIO(resp.content), engine="openpyxl")
    except Exception as e:
        print(f"  {year}: skipped ({e})")
        continue

    odf["Date"] = pd.to_datetime(odf["Date"], errors="coerce")
    matched = name_miss = date_miss = 0

    for _, row in odf.iterrows():
        raw_date = row.get("Date")
        if pd.isna(raw_date):
            continue
        date_str = raw_date.strftime("%Y-%m-%d")

        w_id = resolve_player(str(row.get("Winner", "")))
        l_id = resolve_player(str(row.get("Loser",  "")))
        if not w_id or not l_id:
            name_miss += 1
            continue

        match_id = find_match(w_id, l_id, date_str)
        if not match_id:
            date_miss += 1
            continue

        matched += 1
        for col, (bookmaker, side) in BOOKMAKERS.items():
            if col not in odf.columns:
                continue
            try:
                odds_val = float(row[col])
            except Exception:
                continue
            if pd.isna(odds_val) or odds_val <= 0:
                continue
            pid = w_id if side == "winner" else l_id
            odds_rows.append({
                "match_id":     match_id,
                "bookmaker":    bookmaker,
                "player_id":    pid,
                "closing_odds": round(odds_val, 4),
                "implied_prob": round(1.0 / odds_val, 4),
            })

    total = matched + name_miss + date_miss
    print(f"  {year}: {total} rows  matched={matched}  name_miss={name_miss}  date_miss={date_miss}  "
          f"match_rate={matched/total*100:.1f}%")

print(f"\nTotal odds rows to insert: {len(odds_rows)}")

# If test run, don't wipe and reload all years — just upsert what we got
if test_year:
    print(f"Test mode: upserting {year} odds only (will deduplicate on match_id+bookmaker+player_id) ...")
    # Delete existing odds for matches from this year first
    print("  Clearing existing odds for this year's matches ...")
    year_match_ids = [
        m["match_id"] for m in all_matches
        if m.get("match_date", "")[:4] == str(test_year)
    ]
    for i in range(0, len(year_match_ids), 200):
        chunk = year_match_ids[i:i+200]
        if chunk:
            supabase.table("odds").delete().in_("match_id", chunk).execute()
    batch_insert("odds", odds_rows)
    print(f"  Done. {len(odds_rows)} odds rows for {test_year}.")
else:
    print("Inserting all odds rows ...")
    batch_insert("odds", odds_rows)
    print(f"Done. {len(odds_rows)} total odds rows loaded.")


# ── Rankings (skipped in test mode) ──────────────────────────────────────────
if not test_year:
    print("\nLoading rankings ...")
    RANKING_FILES = [
        os.path.join(DATA_DIR, "atp_rankings_10s.csv"),
        os.path.join(DATA_DIR, "atp_rankings_20s.csv"),
        os.path.join(DATA_DIR, "atp_rankings_current.csv"),
    ]
    ranking_frames = []
    for path in RANKING_FILES:
        if os.path.exists(path):
            rdf = pd.read_csv(path, header=None,
                              names=["rank_date", "atp_rank", "player_sackmann_id", "atp_points"],
                              low_memory=False)
            rdf = rdf[rdf["rank_date"] != "ranking_date"]
            ranking_frames.append(rdf)
            print(f"  {os.path.basename(path)}: {len(rdf)} rows")

    if ranking_frames:
        rdf_all = pd.concat(ranking_frames, ignore_index=True)
        rdf_all["rank_date"] = rdf_all["rank_date"].astype(str)
        rdf_all = rdf_all[rdf_all["rank_date"].str[:4] >= "2015"]

        players_path = os.path.join(DATA_DIR, "atp_players.csv")
        pdf = pd.read_csv(players_path, low_memory=False)
        pdf["full_name"] = (pdf["name_first"].fillna("") + " " + pdf["name_last"].fillna("")).str.strip()
        sid_to_name = dict(zip(pdf["player_id"].astype(str), pdf["full_name"]))

        rdf_all["player_sackmann_id"] = rdf_all["player_sackmann_id"].astype(str)
        rdf_all["full_name"] = rdf_all["player_sackmann_id"].map(sid_to_name)
        rdf_all["player_id"] = rdf_all["full_name"].map(player_map)
        rdf_all = rdf_all.dropna(subset=["player_id"])
        rdf_all["rank_date_fmt"] = rdf_all["rank_date"].apply(
            lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}" if len(str(x)) == 8 else None
        )
        rdf_all = rdf_all.dropna(subset=["rank_date_fmt"])

        rank_rows = [
            {"player_id": int(r["player_id"]), "rank_date": r["rank_date_fmt"],
             "atp_rank": si(r["atp_rank"]), "atp_points": si(r["atp_points"])}
            for _, r in rdf_all.iterrows()
        ]
        print(f"  Inserting {len(rank_rows)} ranking rows ...")
        batch_insert("rankings", rank_rows)
        print(f"  Done.")

print("\nload_odds_rankings.py complete.")
