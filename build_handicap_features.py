"""
build_handicap_features.py - Feature matrix for Games Handicap model.

Target: game_diff = winner_total_games - loser_total_games
        (always positive — winner scored more games)

Features (per-player rolling, no leakage):
  p1/p2 serve hold rate on surface (90d)
  p1/p2 break rate (approximated as opponent hold failure rate) on surface (90d)
  p1/p2 ace rate (90d)
  p1/p2 first serve % (90d)
  p1/p2 avg games played per match (90d, proxy for dominance/stamina)
  p1/p2 avg total games per match on surface (90d, proxy for match length)
  d_elo_surface
  court_speed_index
  fatigue_games_7d for each player
  best_of
  surface one-hots, tier_numeric
  expected_game_diff: Elo-based win-prob → rough game diff estimate
"""
import os, re
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ROLL_DAYS   = 90
FATIGUE_D   = 7
TIER_MAP    = {"Grand Slam": 4, "Masters 1000": 3, "ATP 500": 2, "ATP 250": 1, "Challenger": 0}
SURFACES    = ["Hard", "Clay", "Grass"]


def paginate_all(table, select, filters=None, order=None):
    rows, offset = [], 0
    while True:
        q = supabase.table(table).select(select)
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)
        if order:
            q = q.order(order)
        result = q.range(offset, offset + 999).execute()
        rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    return rows


def parse_games(score):
    if not score or str(score).upper() in ("W/O", "RET", "DEF", "N/A", ""):
        return (0, 0)
    wg = lg = 0
    for s in str(score).split():
        s = re.sub(r"\(\d+\)", "", s)
        parts = s.split("-")
        if len(parts) == 2:
            try:
                wg += int(parts[0])
                lg += int(parts[1])
            except ValueError:
                pass
    return (wg, lg)


def safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def elo_wp(e1, e2):
    return 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))


# Load data
print("Loading matches ...")
matches_raw = paginate_all("matches",
    "match_id,match_date,winner_id,loser_id,tournament_id,score,retirement,best_of",
    order="match_date")
matches_df = pd.DataFrame(matches_raw)
matches_df["match_date"] = pd.to_datetime(matches_df["match_date"]).dt.date
matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

print("Loading tournaments ...")
tours_raw = paginate_all("tournaments", "tournament_id,name,surface,court_speed_index,tier")
tours = {r["tournament_id"]: r for r in tours_raw}

print("Loading match_stats ...")
stats_raw = paginate_all("match_stats",
    "match_id,player_id,aces,first_serve_pct,first_serve_won_pct,bp_faced,bp_saved,service_games_played")
stats_by_match = defaultdict(dict)
for r in stats_raw:
    stats_by_match[r["match_id"]][r["player_id"]] = r

print("Loading elo_history.csv ...")
elo_df = pd.read_csv("./elo_history.csv", low_memory=False)
elo_df["match_date"] = pd.to_datetime(elo_df["match_date"]).dt.date
elo_lookup = {(r["match_id"], r["player_id"]): r for r in elo_df.to_dict("records")}


class HcapState:
    __slots__ = ["matches", "surf_stats"]

    def __init__(self):
        self.matches    = deque()   # (date, wg, lg, won)
        self.surf_stats = defaultdict(deque)
        # surf_stats[s] entries: (date, bp_faced, bp_saved, svc_gms, aces, first_pct)


hstate = defaultdict(HcapState)


def trim(dq, cutoff):
    while dq and dq[0][0] < cutoff:
        dq.popleft()


def get_surf_hold(state, surf, on_date):
    dq = state.surf_stats[surf]
    cutoff = on_date - timedelta(days=ROLL_DAYS)
    trim(dq, cutoff)
    entries = [(bp_f, bp_s, svc_g) for _, bp_f, bp_s, svc_g, *_ in dq
               if bp_f is not None and svc_g is not None and float(svc_g) > 0]
    if not entries:
        return None
    rates = [max(0.0, 1.0 - (float(bpf) - float(bps)) / float(svcg))
             for bpf, bps, svcg in entries]
    return float(np.mean(rates))


def get_ace_rate(state, surf, on_date):
    dq = state.surf_stats[surf]
    cutoff = on_date - timedelta(days=ROLL_DAYS)
    trim(dq, cutoff)
    ta = tsvpts = 0
    for _, bp_f, bp_s, svc_g, aces, first_pct in dq:
        if first_pct is not None:
            tsvpts += 1
        if aces is not None:
            ta += safe(aces)
    return (ta / tsvpts) if tsvpts > 0 else None


def get_first_serve_pct(state, surf, on_date):
    dq = state.surf_stats[surf]
    cutoff = on_date - timedelta(days=ROLL_DAYS)
    trim(dq, cutoff)
    vals = [safe(fp) for _, *_, fp in dq if fp is not None]
    return float(np.mean(vals)) if vals else None


def get_avg_games(state, surf, on_date):
    """Average games played (wg+lg) per match on this surface in last 90d."""
    cutoff = on_date - timedelta(days=ROLL_DAYS)
    games = [m[1] + m[2] for m in state.matches
             if m[0] >= cutoff and m[4] == surf and (m[1] + m[2]) > 0]
    return float(np.mean(games)) if games else None


def get_fatigue(state, on_date):
    cutoff = on_date - timedelta(days=FATIGUE_D)
    return sum(m[1] + m[2] for m in state.matches if m[0] >= cutoff)


print("Building handicap feature rows ...")
rows = []
n_total = len(matches_df)

for idx, match in matches_df.iterrows():
    if idx % 5000 == 0:
        print(f"  {idx}/{n_total} ...", flush=True)

    mid       = match["match_id"]
    mdate     = match["match_date"]
    winner_id = match["winner_id"]
    loser_id  = match["loser_id"]
    tid       = match["tournament_id"]
    score_str = match.get("score", "")
    best_of   = safe(match.get("best_of", 3), 3)
    retirement= bool(match.get("retirement", False))

    wg, lg    = parse_games(score_str)
    total_g   = wg + lg
    game_diff = wg - lg

    # Skip retirements and walkovers — scores are incomplete
    if retirement or total_g == 0:
        # Still update state
        pass

    tour      = tours.get(tid, {})
    surface   = tour.get("surface", "Hard")
    csi       = safe(tour.get("court_speed_index", 42.0), 42.0)
    tier_name = tour.get("tier", "ATP 250")
    tier_num  = TIER_MAP.get(tier_name, 1)

    elo_w = elo_lookup.get((mid, winner_id), {})
    elo_l = elo_lookup.get((mid, loser_id), {})
    elo_w_surf = safe(elo_w.get("elo_surface_before", 1500), 1500)
    elo_l_surf = safe(elo_l.get("elo_surface_before", 1500), 1500)
    d_elo_surf = elo_w_surf - elo_l_surf
    win_prob   = elo_wp(elo_w_surf, elo_l_surf)

    sw = hstate[winner_id]
    sl = hstate[loser_id]

    # Per-player features (winner perspective)
    hold_w   = get_surf_hold(sw, surface, mdate)
    hold_l   = get_surf_hold(sl, surface, mdate)
    ace_w    = get_ace_rate(sw, surface, mdate)
    ace_l    = get_ace_rate(sl, surface, mdate)
    fs_pct_w = get_first_serve_pct(sw, surface, mdate)
    fs_pct_l = get_first_serve_pct(sl, surface, mdate)
    avg_g_w  = get_avg_games(sw, surface, mdate)
    avg_g_l  = get_avg_games(sl, surface, mdate)
    fat_w    = get_fatigue(sw, mdate)
    fat_l    = get_fatigue(sl, mdate)

    surf_hard  = int(surface == "Hard")
    surf_clay  = int(surface == "Clay")
    surf_grass = int(surface == "Grass")

    if total_g > 0 and not retirement:
        rows.append({
            "match_id":     mid,
            "match_date":   str(mdate),
            "winner_id":    winner_id,
            "loser_id":     loser_id,
            "game_diff":    game_diff,
            "total_games":  total_g,
            "best_of":      int(best_of),
            "d_elo_surface":d_elo_surf,
            "elo_win_prob": win_prob,
            "court_speed_index": csi,
            "tier_numeric": tier_num,
            "surface_Hard": surf_hard,
            "surface_Clay": surf_clay,
            "surface_Grass":surf_grass,
            "p1_hold_surface":  hold_w,
            "p2_hold_surface":  hold_l,
            "d_hold_surface":   ((hold_w - hold_l)
                                 if hold_w is not None and hold_l is not None else None),
            "p1_ace_rate":      ace_w,
            "p2_ace_rate":      ace_l,
            "d_ace_rate":       ((ace_w - ace_l)
                                 if ace_w is not None and ace_l is not None else None),
            "p1_first_serve_pct": fs_pct_w,
            "p2_first_serve_pct": fs_pct_l,
            "d_first_serve_pct":  ((fs_pct_w - fs_pct_l)
                                   if fs_pct_w is not None and fs_pct_l is not None else None),
            "p1_avg_games_surface": avg_g_w,
            "p2_avg_games_surface": avg_g_l,
            "p1_fatigue_games_7d":  fat_w,
            "p2_fatigue_games_7d":  fat_l,
            "d_fatigue_games_7d":   fat_w - fat_l,
        })

    # Update state
    stats_w = stats_by_match.get(mid, {}).get(winner_id, {})
    stats_l = stats_by_match.get(mid, {}).get(loser_id, {})

    for pid, state, wg_p, lg_p, won, stats in [
        (winner_id, sw, wg, lg, 1, stats_w),
        (loser_id,  sl, lg, wg, 0, stats_l),
    ]:
        state.matches.append((mdate, wg_p, lg_p, won, surface))
        if stats:
            bpf   = stats.get("bp_faced")
            bps   = stats.get("bp_saved")
            svcg  = stats.get("service_games_played")
            aces  = stats.get("aces")
            fspct = stats.get("first_serve_pct")
            state.surf_stats[surface].append(
                (mdate, bpf, bps, svcg, aces, fspct)
            )


hcap_df = pd.DataFrame(rows)
hcap_df.to_csv("./handicap_features.csv", index=False)
print(f"\nSaved handicap_features.csv  ({len(hcap_df)} rows × {len(hcap_df.columns)} cols)")
print(f"  game_diff range: {hcap_df['game_diff'].min():.0f} – {hcap_df['game_diff'].max():.0f}")
print(f"  mean game_diff: {hcap_df['game_diff'].mean():.2f}  std: {hcap_df['game_diff'].std():.2f}")
print("build_handicap_features.py complete.")
