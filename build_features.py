"""
build_features.py - builds a no-leakage feature matrix from Supabase data.
All features are computed using only data available BEFORE each match.
Output: features.csv
"""
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
BATCH_SIZE   = 200

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ROUND_MAP = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7, "RR": 4, "BR": 5,
}

TIER_MAP = {
    "Grand Slam": 5, "Masters 1000": 4, "ATP 500": 3,
    "ATP 250": 2, "Challenger": 1, "ITF": 0,
}


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


# ── Load everything we need ───────────────────────────────────────────────────
print("Loading data from Supabase ...")

matches_raw = paginate_all(
    "matches",
    "match_id,match_date,winner_id,loser_id,tournament_id,round,best_of,retirement",
    order="match_date"
)
matches_raw.sort(key=lambda m: m["match_date"] or "")
print(f"  {len(matches_raw)} matches")

tours_raw = paginate_all("tournaments", "tournament_id,surface,tier,court_speed_index")
tour_info = {t["tournament_id"]: t for t in tours_raw}

stats_raw = paginate_all(
    "match_stats",
    "match_id,player_id,aces,double_faults,first_serve_pct,first_serve_won_pct,"
    "second_serve_won_pct,bp_faced,bp_saved,service_games_played"
)
stats_by_match = defaultdict(dict)
for s in stats_raw:
    stats_by_match[s["match_id"]][s["player_id"]] = s
print(f"  {len(stats_raw)} stat rows")

rankings_raw = paginate_all("rankings", "player_id,rank_date,atp_rank", order="rank_date")
rankings_by_player: dict[int, list] = defaultdict(list)
for r in rankings_raw:
    rankings_by_player[r["player_id"]].append(r)
print(f"  {len(rankings_raw)} ranking rows")

elo_history_rows = []
try:
    with open("./elo_history.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elo_history_rows.append(row)
    print(f"  {len(elo_history_rows)} elo history rows from CSV")
except FileNotFoundError:
    print("  elo_history.csv not found - Elo features will be 1500")

# Build elo lookup: (match_id, player_id) -> {elo_overall_before, elo_surface_before}
elo_lookup: dict[tuple, dict] = {}
for row in elo_history_rows:
    key = (int(row["match_id"]), int(row["player_id"]))
    elo_lookup[key] = {
        "elo_overall": float(row["elo_overall_before"]),
        "elo_surface": float(row["elo_surface_before"]),
    }


def get_elo(match_id, player_id, default=1500.0):
    key = (match_id, player_id)
    d   = elo_lookup.get(key, {})
    return d.get("elo_overall", default), d.get("elo_surface", default)


def get_rank_before(player_id, before_date_str):
    if not rankings_by_player.get(player_id):
        return None
    best = None
    for r in rankings_by_player[player_id]:
        if r["rank_date"] <= before_date_str:
            best = r
        else:
            break
    return best["atp_rank"] if best else None


# ── Per-match rolling state ───────────────────────────────────────────────────
# Indexed by player_id
player_match_history: dict[int, list] = defaultdict(list)  # list of (date_str, surface, won, stats)


def _fl(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def compute_player_features(player_id, before_date_str, surface, match_id):
    history = player_match_history[player_id]
    before_dt = datetime.strptime(before_date_str, "%Y-%m-%d")

    # Filter to matches strictly before this one
    past = [(d, s, w, st) for (d, s, w, st) in history if d < before_date_str]

    # surface_win_rate_1y
    one_yr_ago = (before_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    surf_1y = [(d, s, w, st) for (d, s, w, st) in past if d >= one_yr_ago and s == surface]
    surf_win_rate = (sum(w for _, _, w, _ in surf_1y) / len(surf_1y)) if surf_1y else None

    # recent_form_10
    last10 = past[-10:]
    recent_form = (sum(w for _, _, w, _ in last10) / len(last10)) if last10 else None

    # serve_rating 90d rolling: average of (1st_won_pct + 2nd_won_pct) / 2
    ninety_ago = (before_dt - timedelta(days=90)).strftime("%Y-%m-%d")
    recent90 = [st for (d, _, _, st) in past if d >= ninety_ago and st]

    def _avg(vals):
        v = [x for x in vals if x is not None]
        return sum(v) / len(v) if v else None

    serve_rating   = _avg([(_fl(s.get("first_serve_won_pct")) or 0) * 0.6 +
                           (_fl(s.get("second_serve_won_pct")) or 0) * 0.4
                           for s in recent90]) if recent90 else None
    return_rating  = _avg([100 - ((_fl(s.get("first_serve_won_pct")) or 50) * 0.6 +
                                  (_fl(s.get("second_serve_won_pct")) or 50) * 0.4)
                           for s in recent90]) if recent90 else None
    ace_rate       = _avg([_fl(s.get("aces")) for s in recent90]) if recent90 else None
    bp_save_rate   = None
    bp_vals        = [(_fl(s.get("bp_saved")), _fl(s.get("bp_faced"))) for s in recent90
                      if s.get("bp_faced") and _fl(s.get("bp_faced")) and _fl(s.get("bp_faced")) > 0]
    if bp_vals:
        bp_save_rate = sum(sv / fc for sv, fc in bp_vals) / len(bp_vals) * 100

    # days_rest and matches_last_14d
    if past:
        last_match_date = datetime.strptime(past[-1][0], "%Y-%m-%d")
        days_rest = (before_dt - last_match_date).days
    else:
        days_rest = None

    two_weeks_ago = (before_dt - timedelta(days=14)).strftime("%Y-%m-%d")
    matches_14d   = len([d for (d, _, _, _) in past if d >= two_weeks_ago])

    # Elo
    elo_overall, elo_surface = get_elo(match_id, player_id)

    return {
        "elo_overall":      elo_overall,
        "elo_surface":      elo_surface,
        "surface_win_rate": surf_win_rate,
        "recent_form_10":   recent_form,
        "serve_rating":     serve_rating,
        "return_rating":    return_rating,
        "ace_rate":         ace_rate,
        "bp_save_rate":     bp_save_rate,
        "days_rest":        days_rest,
        "matches_last_14d": matches_14d,
    }


def compute_h2h(w_id, l_id, before_date_str, surface):
    w_history = player_match_history[w_id]
    h2h_matches = [
        (d, s, won) for (d, s, won, _) in w_history
        if d < before_date_str
    ]
    # All past h2h involving l_id: scan l_id history for matches against w_id
    l_history = player_match_history[l_id]

    # Build h2h counts from w perspective
    # A win for w means they appear as winner in a match where l is loser
    # We'll track this by storing opponent_id in history
    # -- re-derive from the broader match list (already processed)
    # We build a separate dict for this
    return None, None  # Filled below using h2h_state


# ── Build H2H state separately ────────────────────────────────────────────────
# h2h_state[(a, b)] = {"overall": [1=a_won, ...], "surface": {surf: [...]}}
h2h_state: dict[tuple, dict] = defaultdict(lambda: {"overall": [], "surface": defaultdict(list)})


# ── Main feature building loop ───────────────────────────────────────────────
print("Building features ...")
feature_rows = []

SURFACES_ONEHOT = ["Hard", "Clay", "Grass"]
TIERS_ONEHOT    = ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "Challenger"]

for idx, m in enumerate(matches_raw):
    w_id = m["winner_id"]
    l_id = m["loser_id"]
    t_id = m["tournament_id"]
    match_id   = m["match_id"]
    match_date = m["match_date"]
    if not w_id or not l_id or not t_id or not match_date:
        continue

    tinfo   = tour_info.get(t_id, {})
    surface = tinfo.get("surface", "Hard")
    if surface not in ("Hard", "Clay", "Grass"):
        surface = "Hard"
    tier    = tinfo.get("tier", "ATP 250")
    csi     = tinfo.get("court_speed_index") or 42.0

    w_feats = compute_player_features(w_id, match_date, surface, match_id)
    l_feats = compute_player_features(l_id, match_date, surface, match_id)

    w_rank = get_rank_before(w_id, match_date)
    l_rank = get_rank_before(l_id, match_date)

    # H2H
    key_ab   = (min(w_id, l_id), max(w_id, l_id))
    h2h      = h2h_state[key_ab]
    three_yr = (datetime.strptime(match_date, "%Y-%m-%d") - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    h2h_overall_w = sum(1 for (pid, _) in h2h["overall"] if pid == w_id)
    h2h_overall_l = sum(1 for (pid, _) in h2h["overall"] if pid == l_id)
    h2h_total     = h2h_overall_w + h2h_overall_l
    h2h_overall   = (h2h_overall_w / h2h_total) if h2h_total > 0 else 0.5

    surf_h2h  = h2h["surface"].get(surface, [])
    h2h_s_w   = sum(1 for (pid, _) in surf_h2h if pid == w_id)
    h2h_s_l   = sum(1 for (pid, _) in surf_h2h if pid == l_id)
    h2h_s_t   = h2h_s_w + h2h_s_l
    h2h_surf  = (h2h_s_w / h2h_s_t) if h2h_s_t > 0 else 0.5

    # H2H on same surface — last 3 years only
    surf_h2h_3y = [(pid, d) for (pid, d) in surf_h2h if d >= three_yr]
    h2h_3y_w    = sum(1 for (pid, _) in surf_h2h_3y if pid == w_id)
    h2h_3y_l    = sum(1 for (pid, _) in surf_h2h_3y if pid == l_id)
    h2h_3y_t    = h2h_3y_w + h2h_3y_l
    h2h_surf_3y = (h2h_3y_w / h2h_3y_t) if h2h_3y_t > 0 else 0.5

    def diff(a, b, default=0.0):
        if a is None and b is None:
            return default
        a = a if a is not None else 1500.0
        b = b if b is not None else 1500.0
        return a - b

    context = {
        "match_id":          match_id,
        "match_date":        match_date,
        "winner_id":         w_id,
        "loser_id":          l_id,
        "court_speed_index": float(csi),
        "round_numeric":     ROUND_MAP.get(str(m.get("round", "")), 3),
        "best_of":           int(m.get("best_of") or 3),
        "tier_numeric":      TIER_MAP.get(tier, 2),
    }
    for s in SURFACES_ONEHOT:
        context[f"surface_{s}"] = 1 if surface == s else 0
    for t in TIERS_ONEHOT:
        context[f"tier_{t.replace(' ', '_')}"] = 1 if tier == t else 0

    # Generate both perspectives so the model has 50/50 positive/negative labels.
    # P1=winner, target=1 and P1=loser, target=0 — differences simply flip sign.
    for p1_is_winner in (True, False):
        p1f, p2f     = (w_feats, l_feats) if p1_is_winner else (l_feats, w_feats)
        p1_rank, p2_rank = (w_rank, l_rank) if p1_is_winner else (l_rank, w_rank)
        h2h_p1       = h2h_overall  if p1_is_winner else (1 - h2h_overall)
        h2h_p1s      = h2h_surf     if p1_is_winner else (1 - h2h_surf)
        h2h_p1s_3y   = h2h_surf_3y  if p1_is_winner else (1 - h2h_surf_3y)

        # Serve-vs-return matchup: P1's serve power against P2's return ability.
        # Positive = P1 serve dominates P2 return; captures court dynamics better
        # than simple serve or return differences alone.
        def _safe(v, fallback=50.0):
            return v if v is not None else fallback

        serve_vs_return = _safe(p1f["serve_rating"]) - _safe(p2f["return_rating"])

        row = {**context,
            "p1_is_winner":        int(p1_is_winner),
            "d_elo_overall":       diff(p1f["elo_overall"], p2f["elo_overall"]),
            "d_elo_surface":       diff(p1f["elo_surface"], p2f["elo_surface"]),
            "d_atp_rank":          diff(p2_rank, p1_rank),  # lower rank = better
            "d_surface_win_rate":  diff(p1f["surface_win_rate"], p2f["surface_win_rate"]),
            "d_recent_form_10":    diff(p1f["recent_form_10"], p2f["recent_form_10"]),
            "d_serve_rating":      diff(p1f["serve_rating"], p2f["serve_rating"]),
            "d_return_rating":     diff(p1f["return_rating"], p2f["return_rating"]),
            "serve_vs_return":     serve_vs_return,           # NEW: P1 serve - P2 return
            "d_ace_rate":          diff(p1f["ace_rate"], p2f["ace_rate"]),
            "d_bp_save_rate":      diff(p1f["bp_save_rate"], p2f["bp_save_rate"]),
            "d_days_rest":         diff(p1f["days_rest"], p2f["days_rest"]),
            "d_matches_last_14d":  diff(p1f["matches_last_14d"], p2f["matches_last_14d"]),
            "p1_matches_14d":      p1f["matches_last_14d"] or 0,  # NEW: absolute fatigue P1
            "p2_matches_14d":      p2f["matches_last_14d"] or 0,  # NEW: absolute fatigue P2
            "h2h_overall":         h2h_p1,
            "h2h_surface":         h2h_p1s,
            "h2h_surface_3y":      h2h_p1s_3y,               # NEW: 3-year surface H2H
            "winner_won":          int(p1_is_winner),
        }
        feature_rows.append(row)

    # ── Update rolling state AFTER computing features (no leakage) ────────────
    w_stats = stats_by_match.get(match_id, {}).get(w_id)
    l_stats = stats_by_match.get(match_id, {}).get(l_id)

    player_match_history[w_id].append((match_date, surface, 1, w_stats or {}))
    player_match_history[l_id].append((match_date, surface, 0, l_stats or {}))

    h2h_state[key_ab]["overall"].append((w_id, match_date))
    h2h_state[key_ab]["surface"][surface].append((w_id, match_date))

    if (idx + 1) % 5000 == 0:
        print(f"  Processed {idx + 1}/{len(matches_raw)} matches", flush=True)

print(f"  Built {len(feature_rows)} feature rows")

# ── Save to CSV ───────────────────────────────────────────────────────────────
csv_path = "./features.csv"
if feature_rows:
    fieldnames = list(feature_rows[0].keys())
    print(f"Writing {csv_path} ...")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"  Wrote {len(feature_rows)} rows, {len(fieldnames)} columns")

print("\nbuild_features.py complete.")
