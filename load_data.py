import os
import subprocess
import pandas as pd
import requests
from io import BytesIO
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
DATA_DIR     = "./tennis_atp"
BATCH_SIZE   = 200
YEARS        = list(range(2015, 2025))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SURFACE_MAP = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Carpet"}
CSI_MAP     = {"Hard": 42.0, "Clay": 28.0, "Grass": 55.0, "Carpet": 50.0}
TIER_MAP    = {
    "G": "Grand Slam", "M": "Masters 1000",
    "A": "ATP 500", "D": "ATP 250",
    "250": "ATP 250", "500": "ATP 500",
    "C": "Challenger", "F": "ITF"
}


def batch_upsert(table, rows, on_conflict):
    if not rows:
        return
    # Deduplicate within each batch on the conflict key (Postgres rejects duplicate keys in one upsert)
    seen, deduped = set(), []
    for r in rows:
        key = tuple(r[k] for k in on_conflict.split(","))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    for i in range(0, len(deduped), BATCH_SIZE):
        supabase.table(table).upsert(deduped[i:i+BATCH_SIZE], on_conflict=on_conflict).execute()


def batch_insert(table, rows):
    if not rows:
        return
    for i in range(0, len(rows), BATCH_SIZE):
        supabase.table(table).insert(rows[i:i+BATCH_SIZE]).execute()


def paginate_all(table, select="*"):
    rows, offset = [], 0
    while True:
        result = supabase.table(table).select(select).range(offset, offset + 999).execute()
        rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    return rows


def si(v):
    try:
        return int(v) if pd.notna(v) else None
    except Exception:
        return None


def sf(num, den):
    try:
        n, d = float(num), float(den)
        return round(n / d * 100, 2) if d > 0 else None
    except Exception:
        return None


# ── Step 1: Clone or update tennis_atp repo ──────────────────────────────────
if not os.path.exists(DATA_DIR):
    print("Cloning JeffSackmann/tennis_atp ...")
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/JeffSackmann/tennis_atp.git", DATA_DIR],
        check=True
    )
else:
    print("Pulling latest tennis_atp ...")
    subprocess.run(["git", "-C", DATA_DIR, "pull"], check=True)


# ── Step 2: Load CSV data ─────────────────────────────────────────────────────
print("Loading match CSVs ...")
frames = []
for year in YEARS:
    path = os.path.join(DATA_DIR, f"atp_matches_{year}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        df["year"] = year
        frames.append(df)
    else:
        print(f"  Missing: {path}")

df = pd.concat(frames, ignore_index=True)
print(f"  Loaded {len(df)} total match rows")


# ── Step 3: Players ───────────────────────────────────────────────────────────
print("Upserting players ...")
winners = df[["winner_name", "winner_hand", "winner_ht", "winner_ioc"]].rename(
    columns={"winner_name": "name", "winner_hand": "hand",
             "winner_ht": "height_cm", "winner_ioc": "nationality"})
losers = df[["loser_name", "loser_hand", "loser_ht", "loser_ioc"]].rename(
    columns={"loser_name": "name", "loser_hand": "hand",
             "loser_ht": "height_cm", "loser_ioc": "nationality"})
players_df = pd.concat([winners, losers]).drop_duplicates(subset=["name"]).dropna(subset=["name"])

player_rows = []
for _, p in players_df.iterrows():
    player_rows.append({
        "name":        str(p["name"]),
        "hand":        p["hand"] if p["hand"] in ("R", "L", "U") else "U",
        "height_cm":   si(p["height_cm"]),
        "nationality": str(p["nationality"]) if pd.notna(p["nationality"]) else None,
    })
batch_upsert("players", player_rows, "name")
all_players = paginate_all("players", "player_id,name")
player_map  = {r["name"]: r["player_id"] for r in all_players}
print(f"  {len(player_map)} players in DB")


# ── Step 4: Tournaments ───────────────────────────────────────────────────────
print("Upserting tournaments ...")
tours = df[["tourney_id", "tourney_name", "surface", "tourney_level", "draw_size"]].drop_duplicates(subset=["tourney_id"])
tour_rows = []
for _, t in tours.iterrows():
    surface = SURFACE_MAP.get(str(t["surface"]), "Hard")
    tour_rows.append({
        "name":              str(t["tourney_name"]),
        "surface":           surface,
        "court_speed_index": CSI_MAP.get(surface),
        "draw_size":         si(t["draw_size"]),
        "tier":              TIER_MAP.get(str(t["tourney_level"]), "ATP 250"),
    })
batch_upsert("tournaments", tour_rows, "name")
all_tours   = paginate_all("tournaments", "tournament_id,name")
name_to_id  = {r["name"]: r["tournament_id"] for r in all_tours}
tourney_map = {t["tourney_id"]: name_to_id.get(str(t["tourney_name"])) for _, t in tours.iterrows()}
print(f"  {len(tourney_map)} tournaments in DB")


# ── Step 5: Matches + match_stats ─────────────────────────────────────────────
print("Inserting matches and stats ...")
match_rows = []
valid_idx  = []

for idx, row in df.iterrows():
    w_id = player_map.get(str(row.get("winner_name")))
    l_id = player_map.get(str(row.get("loser_name")))
    t_id = tourney_map.get(row.get("tourney_id"))
    if not w_id or not l_id or not t_id:
        continue
    raw = str(row.get("tourney_date", ""))
    match_rows.append({
        "tournament_id": t_id,
        "round":         str(row.get("round", "")),
        "match_date":    f"{raw[:4]}-{raw[4:6]}-{raw[6:]}" if len(raw) == 8 else None,
        "winner_id":     w_id,
        "loser_id":      l_id,
        "score":         str(row.get("score")) if pd.notna(row.get("score")) else None,
        "retirement":    str(row.get("score", "")).endswith("RET"),
        "best_of":       int(row["best_of"]) if pd.notna(row.get("best_of")) else 3,
    })
    valid_idx.append(idx)

df_valid  = df.loc[valid_idx].reset_index(drop=True)
total_matches = len(match_rows)
stat_rows = []

for i in range(0, total_matches, BATCH_SIZE):
    batch_m  = match_rows[i:i+BATCH_SIZE]
    result   = supabase.table("matches").insert(batch_m).execute()
    inserted = result.data
    batch_df = df_valid.iloc[i:i+BATCH_SIZE].reset_index(drop=True)

    for k, match in enumerate(inserted):
        if k >= len(batch_df):
            break
        row      = batch_df.iloc[k]
        match_id = match["match_id"]
        w_id     = player_map.get(str(row.get("winner_name")))
        l_id     = player_map.get(str(row.get("loser_name")))
        svpt_w   = row.get("w_svpt"); in_w = row.get("w_1stIn")
        svpt_l   = row.get("l_svpt"); in_l = row.get("l_1stIn")

        stat_rows += [
            {
                "match_id": match_id, "player_id": w_id,
                "aces": si(row.get("w_ace")), "double_faults": si(row.get("w_df")),
                "first_serve_pct":      sf(in_w, svpt_w),
                "first_serve_won_pct":  sf(row.get("w_1stWon"), in_w),
                "second_serve_won_pct": sf(row.get("w_2ndWon"),
                                          (si(svpt_w) or 0) - (si(in_w) or 0)),
                "bp_faced": si(row.get("w_bpFaced")), "bp_saved": si(row.get("w_bpSaved")),
                "service_games_played": si(row.get("w_SvGms")),
            },
            {
                "match_id": match_id, "player_id": l_id,
                "aces": si(row.get("l_ace")), "double_faults": si(row.get("l_df")),
                "first_serve_pct":      sf(in_l, svpt_l),
                "first_serve_won_pct":  sf(row.get("l_1stWon"), in_l),
                "second_serve_won_pct": sf(row.get("l_2ndWon"),
                                          (si(svpt_l) or 0) - (si(in_l) or 0)),
                "bp_faced": si(row.get("l_bpFaced")), "bp_saved": si(row.get("l_bpSaved")),
                "service_games_played": si(row.get("l_SvGms")),
            },
        ]

    if (i // BATCH_SIZE + 1) % 10 == 0 or i + BATCH_SIZE >= total_matches:
        print(f"  Matches: {min(i+BATCH_SIZE, total_matches)}/{total_matches}", flush=True)

print("Inserting match stats ...")
for i in range(0, len(stat_rows), BATCH_SIZE):
    supabase.table("match_stats").insert(stat_rows[i:i+BATCH_SIZE]).execute()
    if (i // BATCH_SIZE + 1) % 50 == 0:
        print(f"  Stats: {min(i+BATCH_SIZE, len(stat_rows))}/{len(stat_rows)}", flush=True)

print(f"  Inserted {total_matches} matches, {len(stat_rows)} stat rows")


# ── Step 6: Odds from tennis-data.co.uk ──────────────────────────────────────
# tennis-data.co.uk uses "Last F." name format; Sackmann uses "First Last".
# We match by building a last-name index from our DB and resolving ambiguity
# using ATP rank (WRank/LRank columns).
print("Downloading odds from tennis-data.co.uk ...")
BOOKMAKERS = {
    "B365W": ("Bet365",   "winner"),
    "B365L": ("Bet365",   "loser"),
    "PSW":   ("Pinnacle", "winner"),
    "PSL":   ("Pinnacle", "loser"),
}

all_players_db = paginate_all("players", "player_id,name")
player_map_db  = {r["name"]: r["player_id"] for r in all_players_db}

# Build last-name -> list of (player_id, full_name) for quick lookup
from collections import defaultdict as _dd
lastname_index: dict = _dd(list)
for fullname, pid in player_map_db.items():
    parts = fullname.strip().split()
    if parts:
        lastname_index[parts[-1].lower()].append((pid, fullname))


def resolve_player(odds_name: str) -> int | None:
    """Convert 'Last F.' (tennis-data) to a player_id via last-name index."""
    odds_name = odds_name.strip()
    if not odds_name:
        return None
    parts = odds_name.split()
    if not parts:
        return None
    # Last name is the first token in "Last F." format
    last = parts[0].rstrip(".").lower()
    candidates = lastname_index.get(last, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]
    # Multiple players share last name — match on first initial too
    if len(parts) > 1:
        initial = parts[1].replace(".", "").lower()
        for pid, fullname in candidates:
            fname_parts = fullname.strip().split()
            if fname_parts and fname_parts[0].lower().startswith(initial):
                return pid
    # Fall back to first candidate
    return candidates[0][0]


all_matches_db = paginate_all("matches", "match_id,match_date,winner_id,loser_id")
# Build lookup: (date, winner_id, loser_id) -> match_id
match_lookup = {}
for m in all_matches_db:
    key = (m["match_date"], m["winner_id"], m["loser_id"])
    match_lookup[key] = m["match_id"]

odds_rows = []
for year in YEARS:
    url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        odf = pd.read_excel(BytesIO(resp.content), engine="openpyxl")
        print(f"  {year}: {len(odf)} rows from tennis-data.co.uk")
    except Exception as e:
        print(f"  {year}: skipped ({e})")
        continue

    odf["Date"] = pd.to_datetime(odf["Date"], errors="coerce")
    matched = 0
    for _, row in odf.iterrows():
        raw_date = row.get("Date")
        if pd.isna(raw_date):
            continue
        date_str = raw_date.strftime("%Y-%m-%d")

        w_id = resolve_player(str(row.get("Winner", "")))
        l_id = resolve_player(str(row.get("Loser",  "")))
        if not w_id or not l_id:
            continue

        match_id = match_lookup.get((date_str, w_id, l_id))
        if not match_id:
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
    print(f"    -> {matched} matches linked")

print(f"  Inserting {len(odds_rows)} odds rows ...")
batch_insert("odds", odds_rows)
print(f"  Done. {len(odds_rows)} odds rows loaded.")


# ── Step 7: Rankings ──────────────────────────────────────────────────────────
# Sackmann stores rankings in decade files: atp_rankings_10s.csv, _20s.csv, _current.csv
print("Loading rankings ...")
ranking_frames = []
RANKING_FILES = [
    os.path.join(DATA_DIR, "atp_rankings_10s.csv"),
    os.path.join(DATA_DIR, "atp_rankings_20s.csv"),
    os.path.join(DATA_DIR, "atp_rankings_current.csv"),
]
for path in RANKING_FILES:
    if os.path.exists(path):
        rdf = pd.read_csv(path, header=None,
                          names=["rank_date", "atp_rank", "player_sackmann_id", "atp_points"],
                          low_memory=False)
        # Drop repeated header rows
        rdf = rdf[rdf["rank_date"] != "ranking_date"]
        ranking_frames.append(rdf)
        print(f"  Loaded {len(rdf)} rows from {os.path.basename(path)}")

if ranking_frames:
    rdf_all = pd.concat(ranking_frames, ignore_index=True)
    # Load Sackmann player ID -> name mapping
    # atp_players.csv has header: player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id
    players_path = os.path.join(DATA_DIR, "atp_players.csv")
    if os.path.exists(players_path):
        pdf = pd.read_csv(players_path, low_memory=False)
        pdf["name"] = (pdf["name_first"].fillna("") + " " + pdf["name_last"].fillna("")).str.strip()
        sid_to_name = dict(zip(pdf["player_id"].astype(str), pdf["name"]))

        rdf_all["name"] = rdf_all["player_sackmann_id"].map(sid_to_name)
        rdf_all["player_id"] = rdf_all["name"].map(player_map_db)
        rdf_all = rdf_all.dropna(subset=["player_id"])
        rdf_all["rank_date"] = rdf_all["rank_date"].astype(str).apply(
            lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}" if len(str(x)) == 8 else None
        )
        rdf_all = rdf_all.dropna(subset=["rank_date"])

        rank_rows = []
        for _, r in rdf_all.iterrows():
            rank_rows.append({
                "player_id":  int(r["player_id"]),
                "rank_date":  r["rank_date"],
                "atp_rank":   si(r["atp_rank"]),
                "atp_points": si(r["atp_points"]),
            })
        batch_insert("rankings", rank_rows)
        print(f"  {len(rank_rows)} ranking rows loaded")
    else:
        print("  atp_players.csv not found, skipping rankings")
else:
    print("  No ranking files found")

print("\nload_data.py complete.")
