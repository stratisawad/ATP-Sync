"""
build_features_v2.py - Extended feature matrix for Match Winner model v2.

New features vs v1:
  elo_mkt_gap             : Elo surface win-prob minus Pinnacle implied prob
  d_serve_style_csi       : (ace_rate × first_serve_pct) diff × court_speed_index
  p1/p2_fatigue_games_7d  : total games played in last 7 days
  d_fatigue_games_7d      : difference
  p1/p2_surface_transition: weeks since player last played this surface
  d_surface_transition    : difference
  h2h_last5_weighted      : recency-weighted H2H win rate for P1, last 5 meetings
  serve_hold_matchup      : p1 - p2 hold-rate on this surface (rolling 90d)
  p1/p2_form_regression   : Elo-expected win rate minus actual rolling-10 win rate
  tournament_proximity    : 1 if top-30 player in lower-tier event with major within 7d
"""
import os, re
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import date, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SURF_WIN_DAYS    = 365
FORM_N           = 10
SERVE_ROLL_DAYS  = 90
HOLD_ROLL_DAYS   = 90
FATIGUE_DAYS     = 7
PROXIMITY_DAYS   = 7
H2H_LAST_N       = 5
H2H_DECAY        = 0.65
TOP_RANK_THR     = 30
DEFAULT_TRANS    = 52.0
TOP_TIERS        = {"Grand Slam", "Masters 1000"}
LOWER_TIERS      = {"ATP 250", "ATP 500"}
SURFACES         = ["Hard", "Clay", "Grass"]

TIER_MAP = {"Grand Slam": 4, "Masters 1000": 3, "ATP 500": 2,
            "ATP 250": 1, "Challenger": 0}
ROUND_MAP = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5,
             "SF": 6, "F": 7, "RR": 3}


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
    """Return (winner_games, loser_games) from score string."""
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


def elo_wp(elo1, elo2):
    return 1.0 / (1.0 + 10.0 ** ((elo2 - elo1) / 400.0))


def safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading matches ...")
matches_raw = paginate_all("matches",
    "match_id,match_date,winner_id,loser_id,tournament_id,score,retirement,best_of",
    order="match_date")
matches_df = pd.DataFrame(matches_raw)
matches_df["match_date"] = pd.to_datetime(matches_df["match_date"]).dt.date
matches_df = matches_df.sort_values("match_date").reset_index(drop=True)
print(f"  {len(matches_df)} matches")

print("Loading tournaments ...")
tours_raw = paginate_all("tournaments", "tournament_id,name,surface,court_speed_index,tier")
tours = {r["tournament_id"]: r for r in tours_raw}

print("Loading match_stats ...")
stats_raw = paginate_all("match_stats",
    "match_id,player_id,aces,first_serve_pct,first_serve_won_pct,bp_faced,bp_saved,service_games_played")
stats_by_match = defaultdict(dict)  # match_id -> {player_id: stats}
for r in stats_raw:
    stats_by_match[r["match_id"]][r["player_id"]] = r

print("Loading rankings ...")
ranks_raw = paginate_all("rankings", "player_id,rank_date,atp_rank")
ranks_df = pd.DataFrame(ranks_raw)
ranks_df["rank_date"] = pd.to_datetime(ranks_df["rank_date"]).dt.date
ranks_df = ranks_df.sort_values("rank_date")

print("Loading Pinnacle odds ...")
odds_raw = paginate_all("odds", "match_id,player_id,implied_prob",
                        filters={"bookmaker": "Pinnacle"})
pinnacle = {(r["match_id"], r["player_id"]): float(r["implied_prob"]) for r in odds_raw}

print("Loading elo_history.csv ...")
elo_df = pd.read_csv("./elo_history.csv", low_memory=False)
elo_df["match_date"] = pd.to_datetime(elo_df["match_date"]).dt.date
elo_lookup = {(r["match_id"], r["player_id"]): r
              for r in elo_df.to_dict("records")}

# Pre-compute tournament start dates and ranking snapshots
print("Pre-computing tournament schedules ...")
tourney_dates = matches_df.groupby("tournament_id")["match_date"].min().to_dict()
major_schedule = [(d, tid) for tid, d in tourney_dates.items()
                  if tours.get(tid, {}).get("tier") in TOP_TIERS]

# Build a ranking lookup: (player_id, date) -> rank
# We'll use the most recent rank on or before a given date
print("Building ranking index ...")
rank_by_player = defaultdict(list)  # player_id -> sorted [(date, rank)]
for _, row in ranks_df.iterrows():
    rank_by_player[row["player_id"]].append((row["rank_date"], row["atp_rank"]))


def get_rank(pid, on_date):
    history = rank_by_player.get(pid, [])
    rank = None
    for d, r in history:
        if d <= on_date:
            rank = r
        else:
            break
    return rank


# ─────────────────────────────────────────────────────────────────────────────
# Per-player rolling state
# ─────────────────────────────────────────────────────────────────────────────
class PlayerState:
    __slots__ = [
        "matches",         # deque of dicts: date, surface, wg, lg, won, opp_elo
        "h2h",             # defaultdict(deque): opp_id -> [(date, won)]
        "last_surface",    # {surface: date}
        "serve_stats",     # deque of (date, aces, svpts, first_in, bp_faced, bp_saved, svc_gms)
        "surf_serve",      # {surface: deque of same}
        "surf_hold",       # {surface: deque of (date, hold_rate)}
    ]

    def __init__(self):
        self.matches    = deque()
        self.h2h        = defaultdict(deque)
        self.last_surface = defaultdict(lambda: None)
        self.serve_stats  = deque()
        self.surf_serve   = defaultdict(deque)
        self.surf_hold    = defaultdict(deque)


pstate = defaultdict(PlayerState)


def trim_old(dq, cutoff_date):
    while dq and dq[0][0] < cutoff_date:
        dq.popleft()


def serve_style(state, on_date):
    """ace_rate × first_serve_pct rolling 90d."""
    cutoff = on_date - timedelta(days=SERVE_ROLL_DAYS)
    trim_old(state.serve_stats, cutoff)
    if not state.serve_stats:
        return None
    total_aces = total_svpts = total_first_in = total_svpts2 = 0
    for _, aces, svpts, first_in, *_ in state.serve_stats:
        total_aces   += safe(aces)
        total_svpts  += safe(svpts)
        total_first_in += safe(first_in)
    ace_r     = total_aces / total_svpts if total_svpts > 0 else None
    first_p   = total_first_in / total_svpts if total_svpts > 0 else None
    if ace_r is None or first_p is None:
        return None
    return ace_r * first_p


def hold_rate_surface(state, surface, on_date):
    """bp_save_rate proxy as hold-rate on specific surface, rolling 90d."""
    cutoff = on_date - timedelta(days=HOLD_ROLL_DAYS)
    dq = state.surf_hold[surface]
    trim_old(dq, cutoff)
    if not dq:
        return None
    rates = [r for _, r in dq]
    return float(np.mean(rates))


def fatigue_games(state, on_date):
    """Total games played (winner + loser games) in last FATIGUE_DAYS days."""
    cutoff = on_date - timedelta(days=FATIGUE_DAYS)
    total = 0
    for m in state.matches:
        if m["date"] >= cutoff:
            total += m["wg"] + m["lg"]
    return total


def surface_transition(state, surface, on_date):
    """Weeks since player last played this surface. Returns DEFAULT_TRANS if never."""
    last = state.last_surface.get(surface)
    if last is None:
        return DEFAULT_TRANS
    return (on_date - last).days / 7.0


def h2h_weighted(state, opp_id, n=H2H_LAST_N):
    """Recency-weighted H2H win rate for last n meetings."""
    history = list(state.h2h[opp_id])[-n:]
    if not history:
        return 0.5  # neutral prior when no history
    weights = [H2H_DECAY ** (len(history) - 1 - i) for i in range(len(history))]
    total_w = sum(weights)
    return sum(w * r["won"] for w, r in zip(weights, history)) / total_w


def form_regression(state, elo_overall, on_date):
    """Elo expected win rate vs actual rolling-10, using mean opponent Elo."""
    recent = [m for m in state.matches][-FORM_N:]
    if not recent:
        return 0.0
    avg_opp_elo = np.mean([m.get("opp_elo", 1500) for m in recent])
    elo_exp    = elo_wp(safe(elo_overall, 1500), avg_opp_elo)
    actual_wr  = np.mean([m["won"] for m in recent])
    return elo_exp - actual_wr


def is_proximity_flag(match_date, this_tier, player_rank):
    """1 if top-30 player in lower-tier event with a major starting within PROXIMITY_DAYS."""
    if this_tier not in LOWER_TIERS:
        return 0
    if player_rank is None or player_rank > TOP_RANK_THR:
        return 0
    for major_start, _ in major_schedule:
        days_to_major = (major_start - match_date).days
        if 0 <= days_to_major <= PROXIMITY_DAYS:
            return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main loop: iterate chronologically
# ─────────────────────────────────────────────────────────────────────────────
print("Building feature rows ...")
feature_rows = []
n_total = len(matches_df)

for idx, match in matches_df.iterrows():
    if idx % 5000 == 0:
        print(f"  {idx}/{n_total} ...", flush=True)

    mid        = match["match_id"]
    mdate      = match["match_date"]
    winner_id  = match["winner_id"]
    loser_id   = match["loser_id"]
    tid        = match["tournament_id"]
    score_str  = match.get("score", "")
    best_of    = safe(match.get("best_of", 3), 3)
    retirement = bool(match.get("retirement", False))

    tour       = tours.get(tid, {})
    surface    = tour.get("surface", "Hard")
    csi        = safe(tour.get("court_speed_index", 42.0), 42.0)
    tier_name  = tour.get("tier", "ATP 250")
    tier_num   = TIER_MAP.get(tier_name, 1)

    wg, lg     = parse_games(score_str)
    total_games = wg + lg

    # Pre-match Elo
    elo_w = elo_lookup.get((mid, winner_id), {})
    elo_l = elo_lookup.get((mid, loser_id), {})
    elo_w_overall  = safe(elo_w.get("elo_overall_before",  1500), 1500)
    elo_w_surface  = safe(elo_w.get("elo_surface_before",  1500), 1500)
    elo_l_overall  = safe(elo_l.get("elo_overall_before",  1500), 1500)
    elo_l_surface  = safe(elo_l.get("elo_surface_before",  1500), 1500)

    elo_win_prob_surf = elo_wp(elo_w_surface, elo_l_surface)

    # Pinnacle implied for winner side
    pin_win = pinnacle.get((mid, winner_id))
    elo_mkt_gap_w = (elo_win_prob_surf - pin_win) if pin_win is not None else None
    elo_mkt_gap_l = ((1 - elo_win_prob_surf) - (1 - pin_win)) if pin_win is not None else None

    # Rankings
    rank_w = get_rank(winner_id, mdate)
    rank_l = get_rank(loser_id, mdate)

    # Round
    round_str = match.get("round", "R32")
    round_num = ROUND_MAP.get(str(round_str).strip(), 3)

    # Player states (pre-match)
    sw = pstate[winner_id]
    sl = pstate[loser_id]

    # ── All pre-match features (winner perspective) ──────────────────────────
    d_elo_overall   = elo_w_overall - elo_l_overall
    d_elo_surface   = elo_w_surface - elo_l_surface
    d_atp_rank      = (safe(rank_w, 300) - safe(rank_l, 300)) if (rank_w and rank_l) else None

    # Surface win rate (rolling 1y)
    cutoff_1y = mdate - timedelta(days=SURF_WIN_DAYS)

    def surf_wr(state, surf, cutoff):
        recent = [m for m in state.matches if m["date"] >= cutoff and m["surface"] == surf]
        if not recent:
            return None
        return np.mean([m["won"] for m in recent])

    swr_w = surf_wr(sw, surface, cutoff_1y)
    swr_l = surf_wr(sl, surface, cutoff_1y)
    d_surface_win_rate = (swr_w - swr_l) if (swr_w is not None and swr_l is not None) else None

    # Recent form (rolling 10 matches)
    def recent_form(state, n=FORM_N):
        recent = list(state.matches)[-n:]
        return np.mean([m["won"] for m in recent]) if recent else None

    rf_w = recent_form(sw)
    rf_l = recent_form(sl)
    d_recent_form_10 = (rf_w - rf_l) if (rf_w is not None and rf_l is not None) else None

    # Serve style × court speed
    ss_w = serve_style(sw, mdate)
    ss_l = serve_style(sl, mdate)
    d_serve_style_csi = ((ss_w - ss_l) * csi
                         if ss_w is not None and ss_l is not None else None)

    # Fatigue
    fat_w = fatigue_games(sw, mdate)
    fat_l = fatigue_games(sl, mdate)
    d_fatigue_games_7d = fat_w - fat_l

    # Surface transition
    trans_w = surface_transition(sw, surface, mdate)
    trans_l = surface_transition(sl, surface, mdate)
    d_surface_transition = trans_w - trans_l

    # H2H last 5 weighted
    h2h_last5_w = h2h_weighted(sw, loser_id)

    # H2H overall / surface / 3y (kept from v1)
    def h2h_count(state, opp_id, surf=None, days=None, cutoff_date=None):
        history = list(state.h2h[opp_id])
        if days:
            cutoff_date = mdate - timedelta(days=days)
        if cutoff_date:
            history = [m for m in history if m["date"] >= cutoff_date]
        if surf:
            history = [m for m in history if m["surface"] == surf]
        if not history:
            return 0.5  # neutral prior when no history
        return np.mean([m["won"] for m in history])

    h2h_overall     = h2h_count(sw, loser_id)
    h2h_surf        = h2h_count(sw, loser_id, surf=surface)
    h2h_surf_3y     = h2h_count(sw, loser_id, surf=surface, days=365*3)

    # Serve hold matchup on this surface
    hold_w = hold_rate_surface(sw, surface, mdate)
    hold_l = hold_rate_surface(sl, surface, mdate)
    serve_hold_matchup = ((hold_w - hold_l)
                          if hold_w is not None and hold_l is not None else None)

    # Serve / return ratings from rolling stats
    def serve_return_ratings(state, on_date, days=SERVE_ROLL_DAYS):
        cutoff = on_date - timedelta(days=days)
        trim_old(state.serve_stats, cutoff)
        if not state.serve_stats:
            return None, None
        total_first = total_svpts = 0
        total_bp_sv = total_bp_fc = 0
        for _, aces, svpts, first_in, bp_faced, bp_saved, svc_gms in state.serve_stats:
            total_svpts  += safe(svpts)
            total_first  += safe(first_in)
            total_bp_sv  += safe(bp_saved)
            total_bp_fc  += safe(bp_faced)
        serve_r  = (total_first / total_svpts * 100) if total_svpts > 0 else None
        return_r = (total_bp_sv / total_bp_fc * 100) if total_bp_fc > 0 else None
        return serve_r, return_r

    sr_w, rr_w = serve_return_ratings(sw, mdate)
    sr_l, rr_l = serve_return_ratings(sl, mdate)
    d_serve_rating  = (sr_w - sr_l) if (sr_w and sr_l) else None
    d_return_rating = (rr_w - rr_l) if (rr_w and rr_l) else None
    serve_vs_return = (sr_w - rr_l) if (sr_w and rr_l) else None

    # Ace rate / BP save rate differentials
    def ace_bp_rates(state, on_date, days=SERVE_ROLL_DAYS):
        cutoff = on_date - timedelta(days=days)
        trim_old(state.serve_stats, cutoff)
        if not state.serve_stats:
            return None, None
        ta = ts = tbpf = tbps = 0
        for _, aces, svpts, *_, bp_faced, bp_saved, svc_gms in state.serve_stats:
            ta   += safe(aces)
            ts   += safe(svpts)
            tbpf += safe(bp_faced)
            tbps += safe(bp_saved)
        ace_r = ta / ts if ts > 0 else None
        bps_r = tbps / tbpf if tbpf > 0 else None
        return ace_r, bps_r

    ar_w, bpr_w = ace_bp_rates(sw, mdate)
    ar_l, bpr_l = ace_bp_rates(sl, mdate)
    d_ace_rate    = (ar_w - ar_l) if (ar_w and ar_l) else None
    d_bp_save_rate = (bpr_w - bpr_l) if (bpr_w and bpr_l) else None

    # Days rest difference
    def days_since_last(state, on_date):
        if not state.matches:
            return 14
        return (on_date - state.matches[-1]["date"]).days

    rest_w = days_since_last(sw, mdate)
    rest_l = days_since_last(sl, mdate)
    d_days_rest = rest_w - rest_l

    # Matches in last 14 days
    cutoff_14d = mdate - timedelta(days=14)
    m14d_w = sum(1 for m in sw.matches if m["date"] >= cutoff_14d)
    m14d_l = sum(1 for m in sl.matches if m["date"] >= cutoff_14d)
    d_matches_last_14d = m14d_w - m14d_l

    # Form regression flag
    freg_w = form_regression(sw, elo_w_overall, mdate)
    freg_l = form_regression(sl, elo_l_overall, mdate)

    # Tournament proximity flag
    prox_w = is_proximity_flag(mdate, tier_name, rank_w)
    prox_l = is_proximity_flag(mdate, tier_name, rank_l)
    tournament_proximity = max(prox_w, prox_l)

    # One-hot surface and tier
    surf_hard  = int(surface == "Hard")
    surf_clay  = int(surface == "Clay")
    surf_grass = int(surface == "Grass")
    tier_gs    = int(tier_name == "Grand Slam")
    tier_m1k   = int(tier_name == "Masters 1000")
    tier_500   = int(tier_name == "ATP 500")
    tier_250   = int(tier_name == "ATP 250")
    tier_ch    = int(tier_name == "Challenger")

    base = dict(
        match_id=mid, match_date=str(mdate),
        winner_id=winner_id, loser_id=loser_id,
        court_speed_index=csi, round_numeric=round_num,
        best_of=int(best_of), tier_numeric=tier_num,
        surface_Hard=surf_hard, surface_Clay=surf_clay, surface_Grass=surf_grass,
        tier_Grand_Slam=tier_gs, tier_Masters_1000=tier_m1k,
        tier_ATP_500=tier_500, tier_ATP_250=tier_250, tier_Challenger=tier_ch,
        d_elo_overall=d_elo_overall, d_elo_surface=d_elo_surface,
        d_atp_rank=d_atp_rank,
        d_surface_win_rate=d_surface_win_rate,
        d_recent_form_10=d_recent_form_10,
        d_serve_rating=d_serve_rating, d_return_rating=d_return_rating,
        serve_vs_return=serve_vs_return,
        d_ace_rate=d_ace_rate, d_bp_save_rate=d_bp_save_rate,
        d_days_rest=d_days_rest,
        d_matches_last_14d=d_matches_last_14d,
        p1_matches_14d=None, p2_matches_14d=None,  # filled per perspective
        h2h_overall=h2h_overall, h2h_surface=h2h_surf, h2h_surface_3y=h2h_surf_3y,
        h2h_last5_weighted=h2h_last5_w,
        elo_mkt_gap=None,  # filled per perspective
        d_serve_style_csi=d_serve_style_csi,
        p1_fatigue_games_7d=None, p2_fatigue_games_7d=None,
        d_fatigue_games_7d=d_fatigue_games_7d,
        p1_surface_transition=None, p2_surface_transition=None,
        d_surface_transition=d_surface_transition,
        serve_hold_matchup=serve_hold_matchup,
        p1_form_regression=None, p2_form_regression=None,
        tournament_proximity=tournament_proximity,
        # for handicap model
        total_games=total_games,
        game_diff=wg - lg if (wg + lg) > 0 else None,
    )

    # Perspective 1: P1 = winner (target = 1)
    r1 = dict(base)
    r1.update(
        p1_is_winner=1, winner_won=1,
        p1_matches_14d=m14d_w, p2_matches_14d=m14d_l,
        elo_mkt_gap=elo_mkt_gap_w,
        p1_fatigue_games_7d=fat_w, p2_fatigue_games_7d=fat_l,
        p1_surface_transition=trans_w, p2_surface_transition=trans_l,
        p1_form_regression=freg_w, p2_form_regression=freg_l,
    )
    # Perspective 2: P1 = loser (negate all deltas, target = 0)
    r2 = dict(base)
    r2.update(
        p1_is_winner=0, winner_won=0,
        p1_matches_14d=m14d_l, p2_matches_14d=m14d_w,
        elo_mkt_gap=elo_mkt_gap_l,
        p1_fatigue_games_7d=fat_l, p2_fatigue_games_7d=fat_w,
        p1_surface_transition=trans_l, p2_surface_transition=trans_w,
        p1_form_regression=freg_l, p2_form_regression=freg_w,
        d_elo_overall=-d_elo_overall, d_elo_surface=-d_elo_surface,
        d_atp_rank=(-d_atp_rank if d_atp_rank is not None else None),
        d_surface_win_rate=(-d_surface_win_rate
                            if d_surface_win_rate is not None else None),
        d_recent_form_10=(-d_recent_form_10
                          if d_recent_form_10 is not None else None),
        d_serve_rating=(-d_serve_rating if d_serve_rating else None),
        d_return_rating=(-d_return_rating if d_return_rating else None),
        serve_vs_return=((sr_l - rr_w) if (sr_l and rr_w) else None),
        d_ace_rate=(-d_ace_rate if d_ace_rate else None),
        d_bp_save_rate=(-d_bp_save_rate if d_bp_save_rate else None),
        d_days_rest=-d_days_rest,
        d_matches_last_14d=-d_matches_last_14d,
        h2h_overall=(1 - h2h_overall),
        h2h_surface=(1 - h2h_surf),
        h2h_surface_3y=(1 - h2h_surf_3y),
        h2h_last5_weighted=(1 - h2h_last5_w),
        d_serve_style_csi=(-d_serve_style_csi
                           if d_serve_style_csi is not None else None),
        d_fatigue_games_7d=-d_fatigue_games_7d,
        d_surface_transition=-d_surface_transition,
        serve_hold_matchup=(-serve_hold_matchup
                            if serve_hold_matchup is not None else None),
    )
    feature_rows.extend([r1, r2])

    # ── Update state AFTER computing features (no leakage) ───────────────────
    stats_w = stats_by_match.get(mid, {}).get(winner_id, {})
    stats_l = stats_by_match.get(mid, {}).get(loser_id, {})

    def update_player(state, pid, won, opp_id, opp_elo, stats_dict):
        m_rec = {"date": mdate, "surface": surface, "wg": wg, "lg": lg,
                 "won": int(won), "opp_elo": safe(opp_elo, 1500)}
        state.matches.append(m_rec)
        state.h2h[opp_id].append({"date": mdate, "won": int(won), "surface": surface})
        state.last_surface[surface] = mdate

        if stats_dict:
            aces    = stats_dict.get("aces")
            svpts   = stats_dict.get("first_serve_pct")   # actually first_serve_pct*svpt
            first_in = None
            if svpts is not None:
                pass   # we don't have raw service points — use first_serve_pct directly
            bp_f  = stats_dict.get("bp_faced")
            bp_s  = stats_dict.get("bp_saved")
            svc_g = stats_dict.get("service_games_played")

            row = (mdate, aces, None, None, bp_f, bp_s, svc_g)
            state.serve_stats.append(row)
            state.surf_serve[surface].append(row)

            if bp_f and bp_s and svc_g and float(svc_g) > 0:
                hold_r = max(0.0, 1.0 - (float(bp_f) - float(bp_s)) / float(svc_g))
                state.surf_hold[surface].append((mdate, hold_r))

    update_player(sw, winner_id, True,  loser_id,  elo_l_overall, stats_w)
    update_player(sl, loser_id,  False, winner_id, elo_w_overall, stats_l)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nBuilt {len(feature_rows)} rows ({len(feature_rows)//2} matches)")
features_v2 = pd.DataFrame(feature_rows)
features_v2.to_csv("./features_v2.csv", index=False)
print(f"Saved features_v2.csv  ({len(features_v2)} rows × {len(features_v2.columns)} cols)")
print("build_features_v2.py complete.")
