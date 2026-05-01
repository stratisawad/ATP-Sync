"""
predict.py - given two player names and a surface, outputs win probabilities
and value odds for each player.

Usage:
    python predict.py "Novak Djokovic" "Carlos Alcaraz" "Clay"
"""
import sys
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import joblib
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SURFACES_ONEHOT = ["Hard", "Clay", "Grass"]
TIERS_ONEHOT    = ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "Challenger"]
ROUND_MAP = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
}
TIER_MAP = {
    "Grand Slam": 5, "Masters 1000": 4, "ATP 500": 3,
    "ATP 250": 2, "Challenger": 1, "ITF": 0,
}
CSI_MAP = {"Hard": 42.0, "Clay": 28.0, "Grass": 55.0}


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


def _fl(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def get_rank(player_id):
    result = (supabase.table("rankings")
              .select("atp_rank")
              .eq("player_id", player_id)
              .order("rank_date", desc=True)
              .limit(1)
              .execute())
    if result.data:
        return result.data[0]["atp_rank"]
    return None


def get_elo(player_id, surface):
    result = (supabase.table("elo_ratings")
              .select("elo")
              .eq("player_id", player_id)
              .in_("surface", [surface, "Overall"])
              .execute())
    overall = surface_elo = 1500.0
    for r in result.data:
        pass  # handled below
    result_o = (supabase.table("elo_ratings")
                .select("elo,surface")
                .eq("player_id", player_id)
                .execute())
    for r in result_o.data:
        if r["surface"] == "Overall":
            overall = float(r["elo"])
        if r["surface"] == surface:
            surface_elo = float(r["elo"])
    return overall, surface_elo


def get_recent_stats(player_id, days=90):
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    matches = (supabase.table("matches")
               .select("match_id,match_date,winner_id")
               .gte("match_date", cutoff)
               .or_(f"winner_id.eq.{player_id},loser_id.eq.{player_id}")
               .execute())
    if not matches.data:
        return {}, []
    match_ids = [m["match_id"] for m in matches.data]
    # Chunk into groups of 50 for IN query
    stat_rows = []
    for i in range(0, len(match_ids), 50):
        chunk = match_ids[i:i+50]
        res = (supabase.table("match_stats")
               .select("match_id,aces,double_faults,first_serve_won_pct,"
                       "second_serve_won_pct,bp_faced,bp_saved")
               .eq("player_id", player_id)
               .in_("match_id", chunk)
               .execute())
        stat_rows.extend(res.data)
    return matches.data, stat_rows


def compute_features_for_player(player_id, surface):
    elo_overall, elo_surface = get_elo(player_id, surface)
    rank = get_rank(player_id)

    matches_data, stat_rows = get_recent_stats(player_id, days=90)

    def _avg(vals):
        v = [x for x in vals if x is not None]
        return sum(v) / len(v) if v else None

    serve_rating  = _avg([(_fl(s.get("first_serve_won_pct")) or 0) * 0.6 +
                           (_fl(s.get("second_serve_won_pct")) or 0) * 0.4
                           for s in stat_rows]) if stat_rows else None
    return_rating = _avg([100 - ((_fl(s.get("first_serve_won_pct")) or 50) * 0.6 +
                                  (_fl(s.get("second_serve_won_pct")) or 50) * 0.4)
                          for s in stat_rows]) if stat_rows else None
    ace_rate      = _avg([_fl(s.get("aces")) for s in stat_rows]) if stat_rows else None

    bp_vals = [(_fl(s.get("bp_saved")), _fl(s.get("bp_faced"))) for s in stat_rows
               if s.get("bp_faced") and _fl(s.get("bp_faced")) and _fl(s.get("bp_faced")) > 0]
    bp_save_rate = (sum(sv / fc for sv, fc in bp_vals) / len(bp_vals) * 100) if bp_vals else None

    # Recent form: last 10 matches overall
    all_matches_90d = paginate_all(
        "matches",
        "match_id,match_date,winner_id",
    )
    # Filter client-side for this player's last 10 matches
    player_matches = [m for m in all_matches_90d
                      if m.get("winner_id") == player_id or
                      str(m.get("winner_id")) == str(player_id)]
    # Actually query directly
    recent_result = (supabase.table("matches")
                     .select("match_id,winner_id,match_date")
                     .or_(f"winner_id.eq.{player_id},loser_id.eq.{player_id}")
                     .order("match_date", desc=True)
                     .limit(10)
                     .execute())
    form_matches  = recent_result.data if recent_result.data else []
    recent_form   = (sum(1 for m in form_matches if m["winner_id"] == player_id) /
                     len(form_matches)) if form_matches else None

    return {
        "elo_overall":      elo_overall,
        "elo_surface":      elo_surface,
        "rank":             rank,
        "serve_rating":     serve_rating,
        "return_rating":    return_rating,
        "ace_rate":         ace_rate,
        "bp_save_rate":     bp_save_rate,
        "recent_form_10":   recent_form,
        "surface_win_rate": None,  # would need longer query; use None
        "days_rest":        None,
        "matches_last_14d": None,
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: python predict.py <Player1> <Player2> <Surface>")
        print('Example: python predict.py "Novak Djokovic" "Carlos Alcaraz" "Clay"')
        sys.exit(1)

    p1_name = sys.argv[1].strip()
    p2_name = sys.argv[2].strip()
    surface = sys.argv[3].strip().capitalize()

    if surface not in ("Hard", "Clay", "Grass"):
        print(f"Surface must be Hard, Clay, or Grass. Got: {surface}")
        sys.exit(1)

    print(f"\nLooking up players ...")
    r1 = supabase.table("players").select("player_id,name").ilike("name", f"%{p1_name}%").execute()
    r2 = supabase.table("players").select("player_id,name").ilike("name", f"%{p2_name}%").execute()

    if not r1.data:
        print(f"Player not found: {p1_name}")
        sys.exit(1)
    if not r2.data:
        print(f"Player not found: {p2_name}")
        sys.exit(1)

    p1 = r1.data[0]
    p2 = r2.data[0]
    print(f"  P1: {p1['name']} (id={p1['player_id']})")
    print(f"  P2: {p2['name']} (id={p2['player_id']})")
    print(f"  Surface: {surface}")

    print("\nComputing features ...")
    f1 = compute_features_for_player(p1["player_id"], surface)
    f2 = compute_features_for_player(p2["player_id"], surface)

    def diff(a, b, default=0.0):
        a = a if a is not None else default
        b = b if b is not None else default
        return a - b

    feature_vec = {
        "d_elo_overall":      diff(f1["elo_overall"], f2["elo_overall"]),
        "d_elo_surface":      diff(f1["elo_surface"], f2["elo_surface"]),
        "d_atp_rank":         diff(f2["rank"], f1["rank"]) if f1["rank"] and f2["rank"] else 0.0,
        "d_surface_win_rate": 0.0,
        "d_recent_form_10":   diff(f1["recent_form_10"], f2["recent_form_10"]),
        "d_serve_rating":     diff(f1["serve_rating"], f2["serve_rating"]),
        "d_return_rating":    diff(f1["return_rating"], f2["return_rating"]),
        "d_ace_rate":         diff(f1["ace_rate"], f2["ace_rate"]),
        "d_bp_save_rate":     diff(f1["bp_save_rate"], f2["bp_save_rate"]),
        "d_days_rest":        0.0,
        "d_matches_last_14d": 0.0,
        "h2h_overall":        0.5,
        "h2h_surface":        0.5,
        "court_speed_index":  CSI_MAP.get(surface, 42.0),
        "round_numeric":      6,   # assume SF as default
        "best_of":            3,
        "tier_numeric":       3,   # assume ATP 500 default
    }

    for s in SURFACES_ONEHOT:
        feature_vec[f"surface_{s}"] = 1 if surface == s else 0
    for t in TIERS_ONEHOT:
        feature_vec[f"tier_{t.replace(' ', '_')}"] = 0

    print("Loading model ...")
    payload       = joblib.load("./atp_model.pkl")
    model         = payload["model"]
    feature_cols  = payload["feature_cols"]
    medians       = payload["medians"]

    X = np.array([[feature_vec.get(c, medians.get(c, 0.0)) for c in feature_cols]])

    prob_p1 = float(model.predict_proba(X)[0, 1])
    prob_p2 = 1.0 - prob_p1

    fair_odds_p1 = round(1.0 / prob_p1, 3) if prob_p1 > 0 else None
    fair_odds_p2 = round(1.0 / prob_p2, 3) if prob_p2 > 0 else None

    print(f"\n{'='*55}")
    print(f"  {p1['name']:<28}  {prob_p1*100:5.1f}%  fair odds: {fair_odds_p1}")
    print(f"  {p2['name']:<28}  {prob_p2*100:5.1f}%  fair odds: {fair_odds_p2}")
    print(f"{'='*55}")
    print(f"\n  Value betting guide:")
    print(f"  Bet {p1['name']} if market odds > {fair_odds_p1}")
    print(f"  Bet {p2['name']} if market odds > {fair_odds_p2}")

    print(f"\n  Key features ({p1['name']} vs {p2['name']}):")
    print(f"    Elo overall diff:   {feature_vec['d_elo_overall']:+.1f}")
    print(f"    Elo {surface} diff: {feature_vec['d_elo_surface']:+.1f}")
    if f1["rank"] and f2["rank"]:
        print(f"    ATP rank:           {p1['name']} #{f1['rank']} vs {p2['name']} #{f2['rank']}")


if __name__ == "__main__":
    main()
