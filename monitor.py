"""
monitor.py - ATP betting alert system.

Every run (called from cron or manually):
  1. Pull upcoming ATP match odds from The Odds API
  2. Resolve players against Supabase DB (fuzzy name matching)
  3. Compute v2 features for each match from live DB state
  4. Run Model 1 (match winner) and Model 2 (games handicap)
  5. Flag matches where:
       match winner edge > 4%  OR  handicap edge > 3%
  6. For flagged matches: format structured alert with key factors
  7. Send email via smtplib
  8. Log alerts to Supabase alerts table

Credentials in .env:
  ODDS_API_KEY, EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD,
  SMTP_HOST (default: smtp.gmail.com), SMTP_PORT (default: 587)
"""
import os
import re
import json
import smtplib
import joblib
import requests
import unicodedata
import numpy as np
from datetime import date, datetime, timedelta
from collections import defaultdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from supabase import create_client, Client
from rapidfuzz import process as fz_process, fuzz

load_dotenv()

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
ODDS_API_KEY   = os.environ.get("ODDS_API_KEY", "")
EMAIL_FROM     = os.environ.get("EMAIL_FROM", "")
EMAIL_TO       = os.environ.get("EMAIL_TO", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
SMTP_HOST      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.environ.get("SMTP_PORT", "587"))

WINNER_EDGE_THR  = 0.04   # 4%
HANDICAP_EDGE_THR = 0.03  # 3%
KELLY_FRACTION   = 0.25   # quarter-Kelly as base suggestion
MAX_STAKE        = 3.0    # cap suggestion at 3 units
FUZZY_CUTOFF     = 80
ODDS_API_SPORT   = "tennis_atp"
ROLL_DAYS        = 90
FATIGUE_D        = 7
SURF_WIN_DAYS    = 365
FORM_N           = 10
PROXIMITY_DAYS   = 7
TOP_RANK_THR     = 30
LOWER_TIERS      = {"ATP 250", "ATP 500"}
TOP_TIERS        = {"Grand Slam", "Masters 1000"}
H2H_LAST_N       = 5
H2H_DECAY        = 0.65

TIER_MAP  = {"Grand Slam": 4, "Masters 1000": 3, "ATP 500": 2, "ATP 250": 1, "Challenger": 0}
ROUND_MAP = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 3}
CSI_DEFAULT = {"Hard": 42.0, "Clay": 28.0, "Grass": 55.0}

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
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


def normalize(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def elo_wp(e1, e2):
    return 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))


def kelly_stake(edge, net_odds, fraction=KELLY_FRACTION, cap=MAX_STAKE):
    if net_odds <= 0 or edge <= 0:
        return 0.0
    return round(min(fraction * edge / net_odds, cap), 3)


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


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
print("Loading models ...")
try:
    winner_payload  = joblib.load("./atp_model_v2.pkl")
    winner_model    = winner_payload["model"]
    winner_features = winner_payload["feature_cols"]
    winner_medians  = winner_payload["medians"]
    print("  atp_model_v2.pkl loaded")
except FileNotFoundError:
    print("  WARNING: atp_model_v2.pkl not found — trying atp_model.pkl")
    winner_payload  = joblib.load("./atp_model.pkl")
    winner_model    = winner_payload["model"]
    winner_features = winner_payload["feature_cols"]
    winner_medians  = winner_payload["medians"]

try:
    hcap_payload   = joblib.load("./atp_handicap_model.pkl")
    hcap_model     = hcap_payload["model"]
    hcap_features  = hcap_payload["feature_cols"]
    hcap_medians   = hcap_payload["medians"]
    hcap_res_std   = hcap_payload["residual_std"]
    print("  atp_handicap_model.pkl loaded")
except FileNotFoundError:
    hcap_model = None
    print("  WARNING: atp_handicap_model.pkl not found — handicap alerts disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Load DB state (player history for feature computation)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading DB state ...")
all_players = paginate_all("players", "player_id,name")
pid_to_name = {r["player_id"]: r["name"] for r in all_players}
name_to_pid = {r["name"]: r["player_id"] for r in all_players}

# Build fuzzy name index
norm_names = [(normalize(name), pid, name) for name, pid in name_to_pid.items()]
norm_keys  = [x[0] for x in norm_names]

def resolve_player(raw_name):
    raw_name = raw_name.strip()
    if raw_name in name_to_pid:
        return name_to_pid[raw_name]
    norm = normalize(raw_name)
    for nk, pid, _ in norm_names:
        if nk == norm:
            return pid
    result = fz_process.extractOne(norm, norm_keys, scorer=fuzz.token_sort_ratio,
                                   score_cutoff=FUZZY_CUTOFF)
    if result:
        idx = norm_keys.index(result[0])
        return norm_names[idx][1]
    return None


# Load match history (last 18 months for feature computation)
cutoff_load = date.today() - timedelta(days=540)
all_matches = paginate_all("matches",
    "match_id,match_date,winner_id,loser_id,tournament_id,score,retirement,best_of",
    order="match_date")
all_matches = [m for m in all_matches
               if m.get("match_date", "") >= str(cutoff_load)]

# Load stats for those matches
all_match_ids = [m["match_id"] for m in all_matches]
all_stats_raw = paginate_all("match_stats",
    "match_id,player_id,aces,first_serve_pct,bp_faced,bp_saved,service_games_played")
stats_by_match = defaultdict(dict)
for r in all_stats_raw:
    if r["match_id"] in set(all_match_ids):
        stats_by_match[r["match_id"]][r["player_id"]] = r

# Load current Elo
elo_raw = paginate_all("elo_ratings", "player_id,surface,elo")
elo_by_player = defaultdict(dict)
for r in elo_raw:
    elo_by_player[r["player_id"]][r["surface"]] = safe(r["elo"], 1500)

# Load recent rankings
ranks_raw = paginate_all("rankings", "player_id,rank_date,atp_rank", order="rank_date")
rank_by_player = defaultdict(list)
for r in ranks_raw:
    rank_by_player[r["player_id"]].append((r["rank_date"], r["atp_rank"]))


def get_rank(pid):
    history = rank_by_player.get(pid, [])
    return history[-1][1] if history else None


# Load tournament info
tours_raw = paginate_all("tournaments", "tournament_id,name,surface,court_speed_index,tier")
tours_by_id   = {r["tournament_id"]: r for r in tours_raw}
tours_by_name = {normalize(r["name"]): r for r in tours_raw}

# Precompute major schedule
major_starts = []
if all_matches:
    tour_first = defaultdict(lambda: date(2099, 1, 1))
    for m in all_matches:
        tid = m.get("tournament_id")
        md  = m.get("match_date", "")
        if tid and md:
            d = date.fromisoformat(md)
            if d < tour_first[tid]:
                tour_first[tid] = d
    for tid, d in tour_first.items():
        tr = tours_by_id.get(tid, {})
        if tr.get("tier") in TOP_TIERS:
            major_starts.append(d)

print(f"  {len(all_matches)} recent matches loaded for feature computation")


# ─────────────────────────────────────────────────────────────────────────────
# Build per-player rolling state from DB history
# ─────────────────────────────────────────────────────────────────────────────
class LiveState:
    def __init__(self):
        self.matches   = []
        self.h2h       = defaultdict(list)
        self.surf_hold = defaultdict(list)
        self.surf_serve = defaultdict(list)

live_state = defaultdict(LiveState)

for m in all_matches:
    mid       = m["match_id"]
    mdate     = date.fromisoformat(m["match_date"])
    winner_id = m.get("winner_id")
    loser_id  = m.get("loser_id")
    tid       = m.get("tournament_id")
    score_str = m.get("score", "")
    retirement = bool(m.get("retirement", False))

    tour    = tours_by_id.get(tid, {})
    surface = tour.get("surface", "Hard")
    wg, lg  = parse_games(score_str)

    for pid, opp_id, won, pg, og in [
        (winner_id, loser_id,  1, wg, lg),
        (loser_id,  winner_id, 0, lg, wg),
    ]:
        if not pid:
            continue
        st = live_state[pid]
        st.matches.append({"date": mdate, "surface": surface, "won": won,
                           "wg": pg, "lg": og, "opp_id": opp_id})
        st.h2h[opp_id].append({"date": mdate, "won": won, "surface": surface})

        stats = stats_by_match.get(mid, {}).get(pid, {})
        if stats:
            bpf  = stats.get("bp_faced")
            bps  = stats.get("bp_saved")
            svcg = stats.get("service_games_played")
            aces = stats.get("aces")
            fsp  = stats.get("first_serve_pct")
            if bpf is not None and svcg is not None and safe(svcg) > 0:
                hold_r = max(0.0, 1.0 - (safe(bpf) - safe(bps or 0)) / safe(svcg))
                st.surf_hold[surface].append((mdate, hold_r))
            st.surf_serve[surface].append((mdate, aces, fsp))


def compute_features_for_match(p1_id, p2_id, surface, csi, tier_name,
                                 match_date, best_of, round_num,
                                 p1_mkt_imp, p2_mkt_imp):
    """
    Compute feature vector for a live upcoming match.
    Returns (winner_feature_vec, handicap_feature_vec, factor_dict)
    """
    today   = match_date
    cutoff_r= today - timedelta(days=ROLL_DAYS)
    cutoff_f= today - timedelta(days=FATIGUE_D)
    cutoff_1y = today - timedelta(days=SURF_WIN_DAYS)
    cutoff_14d= today - timedelta(days=14)

    elo_p1 = {s: elo_by_player[p1_id].get(s, 1500) for s in ["Overall","Hard","Clay","Grass"]}
    elo_p2 = {s: elo_by_player[p2_id].get(s, 1500) for s in ["Overall","Hard","Clay","Grass"]}
    elo_p1_s = elo_p1.get(surface, elo_p1["Overall"])
    elo_p2_s = elo_p2.get(surface, elo_p2["Overall"])

    win_prob = elo_wp(elo_p1_s, elo_p2_s)
    elo_mkt_gap = (win_prob - p1_mkt_imp) if p1_mkt_imp else None

    rank_p1 = get_rank(p1_id)
    rank_p2 = get_rank(p2_id)

    st1 = live_state[p1_id]
    st2 = live_state[p2_id]

    def surf_wr(st, surf, cutoff):
        recent = [m for m in st.matches if m["date"] >= cutoff and m["surface"] == surf]
        return np.mean([m["won"] for m in recent]) if recent else None

    def recent_form(st, n=FORM_N):
        recent = st.matches[-n:]
        return np.mean([m["won"] for m in recent]) if recent else None

    def hold_rate(st, surf, cutoff):
        recent = [(d, r) for d, r in st.surf_hold[surf] if d >= cutoff]
        return float(np.mean([r for _, r in recent])) if recent else None

    def serve_style(st, surf, cutoff):
        recent = [(aces, fsp) for d, aces, fsp in st.surf_serve[surf] if d >= cutoff]
        if not recent:
            return None
        aces_m = np.nanmean([safe(a) for a, _ in recent if a is not None])
        fsp_m  = np.nanmean([safe(f) for _, f in recent if f is not None])
        return aces_m * fsp_m if np.isfinite(aces_m) and np.isfinite(fsp_m) else None

    def fatigue(st, cutoff):
        return sum(m["wg"] + m["lg"] for m in st.matches if m["date"] >= cutoff)

    def surface_transition(st, surf):
        dates = [m["date"] for m in st.matches if m["surface"] == surf]
        if not dates:
            return 52.0
        return (today - max(dates)).days / 7.0

    def h2h_weighted(st, opp_id):
        history = st.h2h[opp_id][-H2H_LAST_N:]
        if not history:
            return 0.0
        weights = [H2H_DECAY ** (len(history) - 1 - i) for i in range(len(history))]
        return sum(w * m["won"] for w, m in zip(weights, history)) / sum(weights)

    def days_rest(st):
        if not st.matches:
            return 14
        return (today - st.matches[-1]["date"]).days

    def form_regression(st, elo_overall):
        recent = st.matches[-FORM_N:]
        if not recent:
            return 0.0
        avg_opp_elo = np.mean([
            elo_by_player[m["opp_id"]].get("Overall", 1500) for m in recent
        ])
        elo_exp   = elo_wp(elo_overall, avg_opp_elo)
        actual_wr = np.mean([m["won"] for m in recent])
        return elo_exp - actual_wr

    def proximity_flag(rank, tier):
        if tier not in LOWER_TIERS or not rank or rank > TOP_RANK_THR:
            return 0
        for ms in major_starts:
            days = (ms - today).days
            if 0 <= days <= PROXIMITY_DAYS:
                return 1
        return 0

    swr_p1 = surf_wr(st1, surface, cutoff_1y)
    swr_p2 = surf_wr(st2, surface, cutoff_1y)
    rf_p1  = recent_form(st1)
    rf_p2  = recent_form(st2)
    hold_p1 = hold_rate(st1, surface, cutoff_r)
    hold_p2 = hold_rate(st2, surface, cutoff_r)
    ss_p1  = serve_style(st1, surface, cutoff_r)
    ss_p2  = serve_style(st2, surface, cutoff_r)
    fat_p1 = fatigue(st1, cutoff_f)
    fat_p2 = fatigue(st2, cutoff_f)
    tran_p1= surface_transition(st1, surface)
    tran_p2= surface_transition(st2, surface)
    h2h_w  = h2h_weighted(st1, p2_id)
    h2h_h2h_overall = h2h_w
    rest_p1 = days_rest(st1)
    rest_p2 = days_rest(st2)
    m14d_p1 = sum(1 for m in st1.matches if m["date"] >= cutoff_14d)
    m14d_p2 = sum(1 for m in st2.matches if m["date"] >= cutoff_14d)
    freg_p1 = form_regression(st1, elo_p1["Overall"])
    freg_p2 = form_regression(st2, elo_p2["Overall"])
    prox    = max(proximity_flag(rank_p1, tier_name),
                  proximity_flag(rank_p2, tier_name))

    surf_hard  = int(surface == "Hard")
    surf_clay  = int(surface == "Clay")
    surf_grass = int(surface == "Grass")
    tier_num   = TIER_MAP.get(tier_name, 1)
    tier_gs    = int(tier_name == "Grand Slam")
    tier_m1k   = int(tier_name == "Masters 1000")
    tier_500   = int(tier_name == "ATP 500")
    tier_250   = int(tier_name == "ATP 250")
    tier_ch    = int(tier_name == "Challenger")

    d_elo_overall = elo_p1["Overall"] - elo_p2["Overall"]
    d_elo_surface = elo_p1_s - elo_p2_s
    d_atp_rank    = (safe(rank_p1, 300) - safe(rank_p2, 300)) if (rank_p1 and rank_p2) else 0
    d_swr         = (swr_p1 - swr_p2) if (swr_p1 and swr_p2) else 0
    d_rf          = (rf_p1 - rf_p2) if (rf_p1 and rf_p2) else 0
    d_hold        = ((hold_p1 - hold_p2) if (hold_p1 and hold_p2) else None)
    d_ss_csi      = ((ss_p1 - ss_p2) * csi if (ss_p1 and ss_p2) else None)
    d_fat         = fat_p1 - fat_p2
    d_tran        = tran_p1 - tran_p2
    d_rest        = rest_p1 - rest_p2
    d_m14d        = m14d_p1 - m14d_p2

    def _v(x): return x if x is not None else 0.0

    # Winner model feature vector
    wf = {
        "d_elo_overall":           d_elo_overall,
        "d_elo_surface":           d_elo_surface,
        "d_atp_rank":              d_atp_rank,
        "d_surface_win_rate":      _v(d_swr),
        "d_recent_form_10":        _v(d_rf),
        "d_serve_rating":          0.0,
        "d_return_rating":         0.0,
        "serve_vs_return":         0.0,
        "d_ace_rate":              0.0,
        "d_bp_save_rate":          0.0,
        "d_days_rest":             d_rest,
        "d_matches_last_14d":      d_m14d,
        "p1_matches_14d":          m14d_p1,
        "p2_matches_14d":          m14d_p2,
        "p1_fatigue_games_7d":     fat_p1,
        "p2_fatigue_games_7d":     fat_p2,
        "d_fatigue_games_7d":      d_fat,
        "h2h_overall":             h2h_h2h_overall,
        "h2h_surface":             h2h_weighted(st1, p2_id),
        "h2h_surface_3y":          h2h_weighted(st1, p2_id),
        "h2h_last5_weighted":      h2h_w,
        "elo_mkt_gap":             _v(elo_mkt_gap),
        "d_serve_style_csi":       _v(d_ss_csi),
        "p1_surface_transition":   tran_p1,
        "p2_surface_transition":   tran_p2,
        "d_surface_transition":    d_tran,
        "serve_hold_matchup":      _v(d_hold),
        "p1_form_regression":      freg_p1,
        "p2_form_regression":      freg_p2,
        "tournament_proximity":    prox,
        "court_speed_index":       csi,
        "round_numeric":           round_num,
        "best_of":                 int(best_of),
        "tier_numeric":            tier_num,
        "surface_Hard":            surf_hard,
        "surface_Clay":            surf_clay,
        "surface_Grass":           surf_grass,
        "tier_Grand_Slam":         tier_gs,
        "tier_Masters_1000":       tier_m1k,
        "tier_ATP_500":            tier_500,
        "tier_ATP_250":            tier_250,
        "tier_Challenger":         tier_ch,
    }

    # Handicap model feature vector
    hf = {
        "d_elo_surface":           d_elo_surface,
        "elo_win_prob":            win_prob,
        "court_speed_index":       csi,
        "tier_numeric":            tier_num,
        "best_of":                 int(best_of),
        "surface_Hard":            surf_hard,
        "surface_Clay":            surf_clay,
        "surface_Grass":           surf_grass,
        "p1_hold_surface":         _v(hold_p1),
        "p2_hold_surface":         _v(hold_p2),
        "d_hold_surface":          _v(d_hold),
        "p1_ace_rate":             0.0,
        "p2_ace_rate":             0.0,
        "d_ace_rate":              0.0,
        "p1_first_serve_pct":      0.0,
        "p2_first_serve_pct":      0.0,
        "d_first_serve_pct":       0.0,
        "p1_avg_games_surface":    0.0,
        "p2_avg_games_surface":    0.0,
        "p1_fatigue_games_7d":     fat_p1,
        "p2_fatigue_games_7d":     fat_p2,
        "d_fatigue_games_7d":      d_fat,
    }

    factors = {
        "elo_gap":          round(d_elo_surface, 1),
        "elo_mkt_gap":      round(safe(elo_mkt_gap) * 100, 2) if elo_mkt_gap else None,
        "win_prob_elo":     round(win_prob * 100, 1),
        "surface_wr_diff":  round(safe(d_swr) * 100, 1),
        "form_diff":        round(safe(d_rf) * 100, 1),
        "hold_matchup":     round(safe(d_hold) * 100, 1) if d_hold else None,
        "serve_style_csi":  round(safe(d_ss_csi), 3) if d_ss_csi else None,
        "fatigue_p1_games": fat_p1,
        "fatigue_p2_games": fat_p2,
        "surf_transition_p1_wks": round(tran_p1, 1),
        "surf_transition_p2_wks": round(tran_p2, 1),
        "form_regression_p1":     round(freg_p1 * 100, 1),
        "form_regression_p2":     round(freg_p2 * 100, 1),
        "proximity_flag":         prox,
        "h2h_last5_weighted":     round(h2h_w, 3),
        "rank_p1":                rank_p1,
        "rank_p2":                rank_p2,
        "days_rest_p1":           rest_p1,
        "days_rest_p2":           rest_p2,
    }

    return wf, hf, factors


# ─────────────────────────────────────────────────────────────────────────────
# Pull odds from The Odds API
# ─────────────────────────────────────────────────────────────────────────────
def fetch_odds():
    if not ODDS_API_KEY:
        print("  WARNING: ODDS_API_KEY not set — no odds data")
        return []
    url = (f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/odds/"
           f"?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return []


def infer_surface_from_tournament(name):
    """Heuristic surface inference from tournament name."""
    n = normalize(name)
    if any(w in n for w in ["clay", "roland", "monte", "madrid", "rome",
                              "barcelona", "hamburg", "rio", "argentina"]):
        return "Clay"
    if any(w in n for w in ["wimbledon", "grass", "queens", "eastbourne",
                              "halle", "hertogenbosch"]):
        return "Grass"
    return "Hard"


def infer_tier(name):
    n = normalize(name)
    if "grand slam" in n or any(w in n for w in ["australian", "roland", "wimbledon", "us open"]):
        return "Grand Slam"
    if any(w in n for w in ["masters", "montreal", "toronto", "indian wells",
                              "miami", "madrid", "rome", "cincinnati", "canada",
                              "shanghai", "paris", "monte carlo", "hamburg"]):
        return "Masters 1000"
    if any(w in n for w in ["500", "barcelona", "dubai", "acapulco", "rotterdam",
                              "halle", "queens", "vienna", "tokyo", "beijing"]):
        return "ATP 500"
    return "ATP 250"


# ─────────────────────────────────────────────────────────────────────────────
# Format alert
# ─────────────────────────────────────────────────────────────────────────────
def format_alert(p1_name, p2_name, surface, tier, match_date,
                  p1_odds, p2_odds, model_prob_p1,
                  winner_edge, winner_bet_on,
                  hcap_pred, hcap_line, hcap_edge,
                  factors, stake_winner, stake_hcap):
    lines = []
    lines.append(f"=== ATP ALERT: {p1_name} vs {p2_name} ===")
    lines.append(f"Date: {match_date}  |  Surface: {surface}  |  Tier: {tier}")
    lines.append("")

    # Match winner section
    lines.append("── MATCH WINNER ──────────────────────────────")
    lines.append(f"  Market odds:     {p1_name} {p1_odds:.2f}  /  {p2_name} {p2_odds:.2f}")
    lines.append(f"  Model estimate:  {p1_name} win prob = {model_prob_p1*100:.1f}%")
    lines.append(f"  Market implied:  {p1_name} = {100/p1_odds:.1f}%  |  {p2_name} = {100/p2_odds:.1f}%")
    if winner_edge and abs(winner_edge) >= WINNER_EDGE_THR:
        bet_odds = p1_odds if winner_bet_on == p1_name else p2_odds
        ev = (bet_odds - 1) * abs(winner_edge) - (1 - abs(winner_edge))
        lines.append(f"  EDGE:            {winner_bet_on} +{abs(winner_edge)*100:.1f}%  "
                     f"(EV per 1u: {ev:+.3f}u)")
        lines.append(f"  Suggested stake: {stake_winner}u  "
                     f"[quarter-Kelly baseline — adjust manually]")
    else:
        lines.append(f"  Edge: {winner_edge*100:+.1f}% — below threshold, no bet flagged")

    # Handicap section
    lines.append("")
    lines.append("── GAMES HANDICAP ────────────────────────────")
    if hcap_pred is not None:
        lines.append(f"  Model game spread: {p1_name} by {hcap_pred:+.1f} games (expected)")
        if hcap_line is not None and hcap_edge is not None and abs(hcap_edge) >= HANDICAP_EDGE_THR:
            lines.append(f"  Line:              {hcap_line:+.1f}  |  Edge: {hcap_edge*100:+.1f}%")
            lines.append(f"  Suggested stake:   {stake_hcap}u  "
                         f"[quarter-Kelly baseline — adjust manually]")
        elif hcap_line is not None:
            lines.append(f"  Line: {hcap_line:+.1f}  |  Edge {hcap_edge*100:+.1f}% — below threshold")
    else:
        lines.append("  Handicap model not available")

    # Key factors
    lines.append("")
    lines.append("── KEY FACTORS ───────────────────────────────")
    factor_labels = [
        ("elo_gap",             "Elo surface gap (P1−P2)",         lambda v: f"{v:+.0f} pts"),
        ("elo_mkt_gap",         "Elo vs market gap",               lambda v: f"{v:+.1f}%"),
        ("win_prob_elo",        "Elo win probability",             lambda v: f"{v:.1f}%"),
        ("surface_wr_diff",     "Surface win rate diff (P1−P2)",   lambda v: f"{v:+.1f}%"),
        ("form_diff",           "Recent form diff (10 matches)",   lambda v: f"{v:+.1f}%"),
        ("hold_matchup",        "Serve hold matchup (P1−P2)",      lambda v: f"{v:+.1f}%"),
        ("serve_style_csi",     "Serve style × court speed",       lambda v: f"{v:+.4f}"),
        ("fatigue_p1_games",    f"Fatigue: {p1_name} (games 7d)",  lambda v: f"{v}"),
        ("fatigue_p2_games",    f"Fatigue: {p2_name} (games 7d)",  lambda v: f"{v}"),
        ("surf_transition_p1_wks", f"Surface transition: {p1_name}", lambda v: f"{v:.1f} weeks"),
        ("surf_transition_p2_wks", f"Surface transition: {p2_name}", lambda v: f"{v:.1f} weeks"),
        ("form_regression_p1",  f"Form regression: {p1_name}",    lambda v: f"{v:+.1f}%"),
        ("form_regression_p2",  f"Form regression: {p2_name}",    lambda v: f"{v:+.1f}%"),
        ("proximity_flag",      "Tournament proximity flag",        lambda v: "YES" if v else "no"),
        ("h2h_last5_weighted",  "H2H last-5 weighted (P1 wins)",   lambda v: f"{v:.2f}"),
        ("rank_p1",             f"ATP rank: {p1_name}",            lambda v: f"#{v}" if v else "N/A"),
        ("rank_p2",             f"ATP rank: {p2_name}",            lambda v: f"#{v}" if v else "N/A"),
        ("days_rest_p1",        f"Days rest: {p1_name}",           lambda v: f"{v}d"),
        ("days_rest_p2",        f"Days rest: {p2_name}",           lambda v: f"{v}d"),
    ]
    for key, label, fmt in factor_labels:
        val = factors.get(key)
        if val is not None:
            lines.append(f"  • {label:<42} {fmt(val)}")

    lines.append("")
    lines.append("⚠  Unit suggestions are quarter-Kelly estimates, not instructions.")
    lines.append("   Review manually before placing any bet.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Send email
# ─────────────────────────────────────────────────────────────────────────────
def send_email(subject, body):
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD]):
        print("  Email credentials not set — skipping send")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        return True
    except Exception as e:
        print(f"  Email error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Log to Supabase
# ─────────────────────────────────────────────────────────────────────────────
def log_alert(p1_name, p2_name, surface, tier, match_date_str,
               alert_type, p1_odds, p2_odds, model_prob_p1,
               hcap_pred, hcap_line, edge_pct, ev_1unit,
               unit_suggestion, factors, full_text):
    try:
        supabase.table("alerts").insert({
            "match_date":    match_date_str,
            "tournament":    tier,
            "surface":       surface,
            "player1_name":  p1_name,
            "player2_name":  p2_name,
            "alert_type":    alert_type,
            "market_odds_p1":float(p1_odds) if p1_odds else None,
            "market_odds_p2":float(p2_odds) if p2_odds else None,
            "model_prob_p1": float(model_prob_p1) if model_prob_p1 else None,
            "handicap_line": float(hcap_line) if hcap_line else None,
            "model_spread":  float(hcap_pred) if hcap_pred else None,
            "edge_pct":      float(edge_pct) if edge_pct else None,
            "ev_1unit":      float(ev_1unit) if ev_1unit else None,
            "unit_suggestion": float(unit_suggestion) if unit_suggestion else None,
            "key_factors":   json.dumps(factors),
            "full_payload":  json.dumps({"alert_text": full_text}),
            "alert_sent":    True,
        }).execute()
    except Exception as e:
        print(f"  DB log error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"ATP Monitor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    events = fetch_odds()
    print(f"Fetched {len(events)} events from The Odds API")

    alerts_sent = 0

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        try:
            match_dt   = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            match_date = match_dt.date()
        except Exception:
            match_date = date.today()

        # Only process future matches
        if match_date < date.today():
            continue

        # Resolve player IDs
        p1_id = resolve_player(home)
        p2_id = resolve_player(away)
        if not p1_id or not p2_id:
            print(f"  Skipping {home} vs {away}: player not found in DB")
            continue

        # Extract best available H2H odds
        p1_odds = p2_odds = None
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    if home in outcomes and away in outcomes:
                        p1_odds = outcomes[home]
                        p2_odds = outcomes[away]
                        break
            if p1_odds:
                break

        if not p1_odds or not p2_odds:
            continue

        p1_imp = 1.0 / p1_odds
        p2_imp = 1.0 / p2_odds

        # Infer tournament context from event title
        sport_title = event.get("sport_title", "")
        surface  = infer_surface_from_tournament(sport_title)
        tier     = infer_tier(sport_title)
        csi      = CSI_DEFAULT.get(surface, 42.0)

        # Try to match tournament in DB for better context
        norm_title = normalize(sport_title)
        if norm_title in tours_by_name:
            tr       = tours_by_name[norm_title]
            surface  = tr.get("surface", surface)
            csi      = safe(tr.get("court_speed_index", csi), csi)
            tier     = tr.get("tier", tier)

        # Compute features
        wf, hf, factors = compute_features_for_match(
            p1_id, p2_id, surface, csi, tier,
            match_date, 3, 3,
            p1_imp, p2_imp,
        )

        # Run match winner model
        wf_vec = np.array([[wf.get(c, winner_medians.get(c, 0.0))
                            for c in winner_features]])
        model_prob_p1 = float(winner_model.predict_proba(wf_vec)[:, 1][0])
        winner_edge   = model_prob_p1 - p1_imp

        # Run handicap model
        hcap_pred = hcap_line = hcap_edge = None
        if hcap_model is not None:
            hf_vec = np.array([[hf.get(c, hcap_medians.get(c, 0.0))
                                for c in hcap_features]])
            hcap_pred  = float(hcap_model.predict(hf_vec)[0])
            # Use half-game lines around the predicted spread
            hcap_line  = round(hcap_pred) - 0.5 if hcap_pred >= 0 else round(hcap_pred) + 0.5
            from scipy.stats import norm as scipy_norm
            prob_over  = 1 - scipy_norm.cdf((hcap_line - hcap_pred) / max(hcap_res_std, 0.1))
            # For handicap edge we need a market line — use model spread vs 50/50 as proxy
            hcap_edge  = prob_over - 0.5   # positive = model favours the over

        # Flag?
        winner_flagged = abs(winner_edge) >= WINNER_EDGE_THR
        hcap_flagged   = (hcap_edge is not None and abs(hcap_edge) >= HANDICAP_EDGE_THR)

        if not winner_flagged and not hcap_flagged:
            continue

        # Compute stakes
        net_odds_p1 = p1_odds - 1
        net_odds_p2 = p2_odds - 1
        bet_on = home if winner_edge > 0 else away
        bet_odds = p1_odds if winner_edge > 0 else p2_odds
        stake_winner = kelly_stake(abs(winner_edge), bet_odds - 1) if winner_flagged else 0
        stake_hcap   = kelly_stake(abs(hcap_edge), 0.9, cap=2.0) if hcap_flagged else 0

        # Compute EV
        ev = (bet_odds - 1) * abs(winner_edge) - (1 - abs(winner_edge)) if winner_flagged else None

        alert_type = ("both" if winner_flagged and hcap_flagged
                      else "match_winner" if winner_flagged else "handicap")

        alert_text = format_alert(
            home, away, surface, tier, str(match_date),
            p1_odds, p2_odds, model_prob_p1,
            winner_edge, bet_on,
            hcap_pred, hcap_line, hcap_edge,
            factors, stake_winner, stake_hcap,
        )

        print(f"\n{'─'*60}")
        print(alert_text)

        # Email
        subject = (f"ATP Alert [{alert_type.upper()}]: {home} vs {away} "
                   f"— edge {abs(winner_edge)*100:.1f}%")
        sent = send_email(subject, alert_text)
        if sent:
            print(f"  Email sent to {EMAIL_TO}")

        # Log to DB
        log_alert(
            home, away, surface, tier, str(match_date),
            alert_type, p1_odds, p2_odds, model_prob_p1,
            hcap_pred, hcap_line,
            float(max(abs(winner_edge), abs(hcap_edge or 0))),
            float(ev) if ev else None,
            float(max(stake_winner, stake_hcap)),
            factors, alert_text,
        )
        alerts_sent += 1

    print(f"\n{'='*60}")
    print(f"Run complete. {alerts_sent} alert(s) generated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
