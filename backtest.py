"""
backtest.py - joins 2024 model predictions with Pinnacle closing odds,
simulates flat-stake betting where edge = model_prob - pinnacle_implied > 0.03.
Outputs backtest_results.csv and a summary report.
"""
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
STAKE        = 1.0   # flat stake per bet
EDGE_THRESH  = 0.03

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


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


# ── Load predictions ──────────────────────────────────────────────────────────
# test_predictions.csv has one row per match (P1=actual winner), model_win_prob = P(winner wins)
print("Loading test_predictions.csv ...")
preds = pd.read_csv("./test_predictions.csv", low_memory=False)
print(f"  {len(preds)} prediction rows (2024)")

# ── Load Pinnacle odds from DB ────────────────────────────────────────────────
print("Loading Pinnacle odds from Supabase ...")
odds_raw = paginate_all("odds", "match_id,player_id,closing_odds,implied_prob",
                         filters={"bookmaker": "Pinnacle"})
odds_df  = pd.DataFrame(odds_raw)
print(f"  {len(odds_df)} Pinnacle odds rows")

if odds_df.empty:
    print("No Pinnacle odds found in DB. Exiting.")
    exit(0)

# ── Load tournament context ───────────────────────────────────────────────────
matches_raw = paginate_all("matches",
                            "match_id,match_date,winner_id,loser_id,tournament_id",
                            order="match_date")
matches_df  = pd.DataFrame(matches_raw)
matches_df  = matches_df[matches_df["match_date"].str.startswith("2024", na=False)]

tours_raw = paginate_all("tournaments", "tournament_id,surface,tier")
tours_df  = pd.DataFrame(tours_raw)
tours_df.rename(columns={"surface": "surface_name", "tier": "tier_name"}, inplace=True)
matches_df = matches_df.merge(tours_df, on="tournament_id", how="left")

# ── Build bet candidates: one row per (match, player) ────────────────────────
# For each match we have model_win_prob = P(actual winner wins).
# We generate two candidate bets: backing the winner and backing the loser.
# True outcome: winner side always wins (won=1), loser side always loses (won=0).
bet_rows = []
odds_by_match_player = {(r["match_id"], r["player_id"]): r for r in odds_raw}

for _, pred in preds.iterrows():
    mid    = pred["match_id"]
    w_id   = pred["winner_id"]
    l_id   = pred["loser_id"]
    p_win  = pred["model_win_prob"]   # prob that winner wins
    p_lose = 1.0 - p_win              # prob that loser wins (per model)

    # Context
    ctx = matches_df[matches_df["match_id"] == mid]
    surface_name = ctx["surface_name"].values[0] if len(ctx) else None
    tier_name    = ctx["tier_name"].values[0]    if len(ctx) else None

    for player_id, model_prob, actual_won in [
        (w_id, p_win,  1),
        (l_id, p_lose, 0),
    ]:
        key  = (mid, player_id)
        odds = odds_by_match_player.get(key)
        if not odds:
            continue
        closing  = float(odds["closing_odds"])
        imp_prob = float(odds["implied_prob"])
        edge     = model_prob - imp_prob
        bet_rows.append({
            "match_id":      mid,
            "match_date":    pred["match_date"],
            "player_id":     player_id,
            "model_prob":    round(model_prob, 4),
            "pinnacle_odds": closing,
            "pinnacle_implied": imp_prob,
            "edge":          round(edge, 4),
            "actual_won":    actual_won,
            "surface_name":  surface_name,
            "tier_name":     tier_name,
        })

joined = pd.DataFrame(bet_rows)
print(f"  {len(joined)} bet candidates (two per match with odds)")

if joined.empty:
    print("No matching rows. Check that match_ids and player_ids align.")
    exit(0)

# ── Simulate flat-stake betting on edge > threshold ───────────────────────────
joined["bet"] = joined["edge"] > EDGE_THRESH
joined["pnl"] = np.where(
    joined["bet"],
    np.where(joined["actual_won"] == 1,
             STAKE * (joined["pinnacle_odds"] - 1),
             -STAKE),
    0.0
)

bets = joined[joined["bet"]].copy()
print(f"\n--- Backtest Results (edge threshold: {EDGE_THRESH}) ---")
print(f"Total opportunities: {len(joined)}")
print(f"Total bets placed:   {len(bets)}")

if len(bets) == 0:
    print("No bets placed at this edge threshold.")
else:
    win_rate  = bets["actual_won"].mean()
    total_pnl = bets["pnl"].sum()
    roi       = total_pnl / (len(bets) * STAKE) * 100
    avg_odds  = bets["pinnacle_odds"].mean()
    clv       = (bets["model_prob"] - bets["pinnacle_implied"]).mean() * 100

    print(f"Win rate:            {win_rate:.3f}")
    print(f"Total PnL:           {total_pnl:+.2f} units")
    print(f"ROI:                 {roi:+.2f}%")
    print(f"Avg closing odds:    {avg_odds:.3f}")
    print(f"Avg CLV:             {clv:+.3f}%")

    bets["month"] = pd.to_datetime(bets["match_date"]).dt.to_period("M")
    monthly = bets.groupby("month")["pnl"].agg(["sum", "count"]).reset_index()
    monthly.columns = ["month", "pnl", "bets"]
    print("\nMonthly PnL:")
    print(monthly.to_string(index=False))

    if "surface_name" in bets.columns:
        surf_summary = bets.groupby("surface_name").agg(
            bets_count=("pnl", "count"),
            pnl=("pnl", "sum"),
            win_rate=("actual_won", "mean"),
            avg_edge=("edge", "mean"),
        ).reset_index()
        print("\nBy surface:")
        print(surf_summary.to_string(index=False))

    if "tier_name" in bets.columns:
        tier_summary = bets.groupby("tier_name").agg(
            bets_count=("pnl", "count"),
            pnl=("pnl", "sum"),
            win_rate=("actual_won", "mean"),
            avg_edge=("edge", "mean"),
        ).reset_index()
        print("\nBy tier:")
        print(tier_summary.to_string(index=False))

# ── Save results ──────────────────────────────────────────────────────────────
out_cols = ["match_id", "match_date", "player_id", "model_prob",
            "pinnacle_odds", "pinnacle_implied", "edge", "bet", "actual_won", "pnl",
            "surface_name", "tier_name"]
out_cols = [c for c in out_cols if c in joined.columns]
joined[out_cols].to_csv("./backtest_results.csv", index=False)
print(f"\nSaved backtest_results.csv ({len(joined)} rows)")
print("\nbacktest.py complete.")
