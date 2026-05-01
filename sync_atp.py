import os
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
DATA_DIR     = "./tennis_atp"
BATCH_SIZE   = 200

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SURFACE_MAP = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Carpet"}
CSI_MAP     = {"Hard": 42.0, "Clay": 28.0, "Grass": 55.0, "Carpet": 50.0}
TIER_MAP    = {
    "G": "Grand Slam", "M": "Masters 1000",
    "A": "ATP 500",    "D": "ATP 250",
    "250": "ATP 250",  "500": "ATP 500",
    "C": "Challenger", "F": "ITF",
}
START_ELO = 1500.0
TIER_K    = {
    "Grand Slam": 50, "Masters 1000": 50,
    "ATP 500": 40,    "ATP 250": 32,
    "Challenger": 32, "ITF": 32,
}


def get_k(tier):
    return TIER_K.get(tier, 32)


def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


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


# ── Update repo ───────────────────────────────────────────────────────────────
if not os.path.exists(DATA_DIR):
    print("Cloning tennis_atp ...")
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/JeffSackmann/tennis_atp.git", DATA_DIR],
        check=True
    )
else:
    print("Pulling latest tennis_atp ...")
    subprocess.run(["git", "-C", DATA_DIR, "pull"], check=True)

# ── Determine cutoff date ─────────────────────────────────────────────────────
print("Checking last sync date ...")
result = (supabase.table("matches")
          .select("match_date")
          .order("match_date", desc=True)
          .limit(1)
          .execute())

if result.data and result.data[0]["match_date"]:
    last_date   = datetime.strptime(result.data[0]["match_date"], "%Y-%m-%d")
    cutoff      = (last_date - timedelta(days=7)).strftime("%Y%m%d")
    cutoff_year = last_date.year
    print("Last match in DB: " + str(last_date.date()))
else:
    cutoff      = "20150101"
    cutoff_year = 2015
    print("No matches found - doing full load from 2015")

current_year = datetime.now().year
frames = []
for year in range(max(cutoff_year, current_year - 1), current_year + 1):
    path = os.path.join(DATA_DIR, "atp_matches_" + str(year) + ".csv")
    if os.path.exists(path):
        df_year = pd.read_csv(path, low_memory=False)
        df_year["year"] = year
        frames.append(df_year)

if not frames:
    print("No new CSV data found. Exiting.")
    raise SystemExit(0)

df = pd.concat(frames, ignore_index=True)
df = df[df["tourney_date"].astype(str) >= cutoff]
print(str(len(df)) + " new match rows to process")

if len(df) == 0:
    print("Database is already up to date.")
    raise SystemExit(0)

# ── Players ───────────────────────────────────────────────────────────────────
print("Upserting players ...")
winners = df[["winner_name", "winner_hand", "winner_ht", "winner_ioc"]].rename(
    columns={"winner_name": "name", "winner_hand": "hand",
             "winner_ht": "height_cm", "winner_ioc": "nationality"})
losers  = df[["loser_name", "loser_hand", "loser_ht", "loser_ioc"]].rename(
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
print(str(len(player_map)) + " total players in DB")

# ── Tournaments ───────────────────────────────────────────────────────────────
print("Upserting tournaments ...")
tours = df[["tourney_id", "tourney_name", "surface", "tourney_level", "draw_size"]].drop_duplicates(
    subset=["tourney_id"])
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
print(str(len(tourney_map)) + " total tournaments in DB")

# ── Matches + stats ───────────────────────────────────────────────────────────
print("Inserting new matches and stats ...")
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
        "match_date":    raw[:4] + "-" + raw[4:6] + "-" + raw[6:] if len(raw) == 8 else None,
        "winner_id":     w_id,
        "loser_id":      l_id,
        "score":         str(row.get("score")) if pd.notna(row.get("score")) else None,
        "retirement":    str(row.get("score", "")).endswith("RET"),
        "best_of":       int(row["best_of"]) if pd.notna(row.get("best_of")) else 3,
    })
    valid_idx.append(idx)

df_valid  = df.loc[valid_idx].reset_index(drop=True)
stat_rows = []
new_match_ids = []

for i in range(0, len(match_rows), BATCH_SIZE):
    result   = supabase.table("matches").insert(match_rows[i:i + BATCH_SIZE]).execute()
    inserted = result.data
    batch_df = df_valid.iloc[i:i + BATCH_SIZE].reset_index(drop=True)

    for k, match in enumerate(inserted):
        if k >= len(batch_df):
            break
        row      = batch_df.iloc[k]
        match_id = match["match_id"]
        new_match_ids.append(match_id)
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

for i in range(0, len(stat_rows), BATCH_SIZE):
    supabase.table("match_stats").insert(stat_rows[i:i + BATCH_SIZE]).execute()

print("Sync complete! New matches: " + str(len(match_rows)) + "  New stats: " + str(len(stat_rows)))

# ── Incremental Elo update for new matches only ───────────────────────────────
if not new_match_ids:
    print("No new matches to recompute Elo for.")
    raise SystemExit(0)

print("Recomputing Elo for " + str(len(new_match_ids)) + " new matches ...")

# Load current Elo state
elo_raw = paginate_all("elo_ratings", "player_id,surface,elo,matches_played")
elo_state: dict = {}
SURFACES = ["Overall", "Hard", "Clay", "Grass"]
for r in elo_raw:
    pid = r["player_id"]
    if pid not in elo_state:
        elo_state[pid] = {s: START_ELO for s in SURFACES}
        elo_state[pid]["matches_played"] = 0
    elo_state[pid][r["surface"]] = float(r["elo"])
    if r["surface"] == "Overall":
        elo_state[pid]["matches_played"] = r.get("matches_played") or 0

# Load tournament info for new matches
new_matches_result = (supabase.table("matches")
                      .select("match_id,match_date,winner_id,loser_id,tournament_id")
                      .in_("match_id", new_match_ids[:100])
                      .order("match_date")
                      .execute())
new_matches = new_matches_result.data

all_tours_raw = paginate_all("tournaments", "tournament_id,surface,tier")
tour_info     = {t["tournament_id"]: t for t in all_tours_raw}


def get_elo_val(pid, surface):
    if pid not in elo_state:
        elo_state[pid] = {s: START_ELO for s in SURFACES}
        elo_state[pid]["matches_played"] = 0
    return elo_state[pid][surface]


for m in sorted(new_matches, key=lambda x: x.get("match_date") or ""):
    w_id = m["winner_id"]
    l_id = m["loser_id"]
    t_id = m["tournament_id"]
    if not w_id or not l_id or not t_id:
        continue
    tinfo   = tour_info.get(t_id, {})
    surface = tinfo.get("surface", "Hard")
    if surface not in ("Hard", "Clay", "Grass"):
        surface = "Hard"
    tier = tinfo.get("tier", "ATP 250")
    k    = get_k(tier)

    w_o = get_elo_val(w_id, "Overall")
    l_o = get_elo_val(l_id, "Overall")
    w_s = get_elo_val(w_id, surface)
    l_s = get_elo_val(l_id, surface)

    e_o = expected_score(w_o, l_o)
    elo_state[w_id]["Overall"] = w_o + k * (1 - e_o)
    elo_state[l_id]["Overall"] = l_o + k * (0 - (1 - e_o))

    e_s = expected_score(w_s, l_s)
    elo_state[w_id][surface] = w_s + k * (1 - e_s)
    elo_state[l_id][surface] = l_s + k * (0 - (1 - e_s))

    elo_state[w_id]["matches_played"] = elo_state[w_id].get("matches_played", 0) + 1
    elo_state[l_id]["matches_played"] = elo_state[l_id].get("matches_played", 0) + 1

from datetime import date as date_cls
today      = date_cls.today().isoformat()
elo_rows   = []
player_ids = set(m["winner_id"] for m in new_matches) | set(m["loser_id"] for m in new_matches)
for pid in player_ids:
    if pid not in elo_state:
        continue
    for surf in SURFACES:
        elo_rows.append({
            "player_id":      pid,
            "surface":        surf,
            "elo":            round(elo_state[pid][surf], 2),
            "matches_played": elo_state[pid].get("matches_played", 0),
            "updated_date":   today,
        })

batch_upsert("elo_ratings", elo_rows, "player_id,surface")
print("Elo updated for " + str(len(player_ids)) + " players")

# ── Commit and push to GitHub ─────────────────────────────────────────────────
print("Committing and pushing to GitHub ...")
subprocess.run(["git", "add", "-A"], check=True)
result = subprocess.run(["git", "diff", "--cached", "--quiet"])
if result.returncode == 0:
    print("Nothing to commit.")
else:
    today_str = date_cls.today().isoformat()
    subprocess.run(
        ["git", "commit", "-m", "sync: daily ATP data update " + today_str],
        check=True
    )
    subprocess.run(["git", "push"], check=True)
    print("Pushed to GitHub.")
