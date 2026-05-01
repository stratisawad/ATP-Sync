import os
import csv
from datetime import date
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
BATCH_SIZE   = 200

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SURFACES  = ["Overall", "Hard", "Clay", "Grass"]
START_ELO = 1500.0

TIER_K = {
    "Grand Slam":    50,
    "Masters 1000":  50,
    "ATP 500":       40,
    "ATP 250":       32,
    "Challenger":    32,
    "ITF":           32,
}


def get_k(tier):
    return TIER_K.get(tier, 32)


def expected(ra, rb):
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
    for i in range(0, len(rows), BATCH_SIZE):
        supabase.table(table).upsert(rows[i:i + BATCH_SIZE], on_conflict=on_conflict).execute()


# ── Load all matches with tournament surface/tier ────────────────────────────
print("Loading matches ...")
matches = paginate_all(
    "matches",
    "match_id,match_date,winner_id,loser_id,tournament_id",
    order="match_date"
)
print(f"  {len(matches)} matches loaded")

print("Loading tournaments ...")
tours_raw = paginate_all("tournaments", "tournament_id,surface,tier")
tour_info = {t["tournament_id"]: t for t in tours_raw}

# Sort matches by date (already ordered, but ensure it)
matches.sort(key=lambda m: m["match_date"] or "")

# ── Elo state: player_id -> {surface: elo, ..., matches_played} ──────────────
elo_state: dict[int, dict] = {}


def get_elo(pid, surface):
    if pid not in elo_state:
        elo_state[pid] = {s: START_ELO for s in SURFACES}
        elo_state[pid]["matches_played"] = 0
    return elo_state[pid][surface]


def set_elo(pid, surface, val):
    elo_state[pid][surface] = val


# ── Process matches and write elo_history.csv ─────────────────────────────────
print("Computing Elo ...")
history_rows = []

for m in matches:
    w_id = m["winner_id"]
    l_id = m["loser_id"]
    t_id = m["tournament_id"]
    if not w_id or not l_id or not t_id:
        continue

    tinfo   = tour_info.get(t_id, {})
    surface = tinfo.get("surface", "Hard")
    if surface not in ("Hard", "Clay", "Grass"):
        surface = "Hard"
    tier    = tinfo.get("tier", "ATP 250")
    k       = get_k(tier)

    # Snapshot before-match values for history
    w_overall_before = get_elo(w_id, "Overall")
    w_surface_before = get_elo(w_id, surface)
    l_overall_before = get_elo(l_id, "Overall")
    l_surface_before = get_elo(l_id, surface)

    history_rows.append({
        "match_id":             m["match_id"],
        "player_id":            w_id,
        "elo_overall_before":   round(w_overall_before, 2),
        "elo_surface_before":   round(w_surface_before, 2),
        "surface":              surface,
        "match_date":           m["match_date"],
        "result":               "win",
    })
    history_rows.append({
        "match_id":             m["match_id"],
        "player_id":            l_id,
        "elo_overall_before":   round(l_overall_before, 2),
        "elo_surface_before":   round(l_surface_before, 2),
        "surface":              surface,
        "match_date":           m["match_date"],
        "result":               "loss",
    })

    # Update Overall
    e_w_o = expected(w_overall_before, l_overall_before)
    new_w_o = w_overall_before + k * (1 - e_w_o)
    new_l_o = l_overall_before + k * (0 - (1 - e_w_o))
    set_elo(w_id, "Overall", new_w_o)
    set_elo(l_id, "Overall", new_l_o)

    # Update surface-specific
    e_w_s = expected(w_surface_before, l_surface_before)
    new_w_s = w_surface_before + k * (1 - e_w_s)
    new_l_s = l_surface_before + k * (0 - (1 - e_w_s))
    set_elo(w_id, surface, new_w_s)
    set_elo(l_id, surface, new_l_s)

    # Track matches played
    if w_id not in elo_state:
        elo_state[w_id] = {s: START_ELO for s in SURFACES}
        elo_state[w_id]["matches_played"] = 0
    if l_id not in elo_state:
        elo_state[l_id] = {s: START_ELO for s in SURFACES}
        elo_state[l_id]["matches_played"] = 0
    elo_state[w_id]["matches_played"] += 1
    elo_state[l_id]["matches_played"] += 1

print(f"  Computed Elo for {len(elo_state)} players, {len(history_rows)} history rows")

# ── Save elo_history.csv ──────────────────────────────────────────────────────
csv_path = "./elo_history.csv"
fieldnames = ["match_id", "player_id", "elo_overall_before",
              "elo_surface_before", "surface", "match_date", "result"]
print(f"Writing {csv_path} ...")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(history_rows)
print(f"  Wrote {len(history_rows)} rows to {csv_path}")

# ── Upsert final Elo ratings into elo_ratings table ───────────────────────────
print("Upserting final Elo ratings ...")
today = date.today().isoformat()
elo_rows = []
for pid, state in elo_state.items():
    for surface in SURFACES:
        elo_rows.append({
            "player_id":      pid,
            "surface":        surface,
            "elo":            round(state[surface], 2),
            "matches_played": state["matches_played"],
            "updated_date":   today,
        })

batch_upsert("elo_ratings", elo_rows, "player_id,surface")
print(f"  Upserted {len(elo_rows)} elo_ratings rows")
print("\ncompute_elo.py complete.")
