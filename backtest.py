"""
backtest.py - joins 2024 model predictions with Pinnacle closing odds.

Filters:
  - edge >= EDGE_THRESH (5%) for all bets
  - edge >= EDGE_THRESH_FAV (7%) when closing_odds < FAVORITE_ODDS_CUTOFF (1.5)
  - closing_odds <= MAX_ODDS (3.0)
  - Clay surface excluded

Staking: half-Kelly   stake = 0.5 * edge / (closing_odds - 1)
"""
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_KEY         = os.environ["SUPABASE_KEY"]
import json

CFG_PATH = "./betting_config.json"
with open(CFG_PATH) as _f:
    CFG = json.load(_f)

EDGE_THRESH          = CFG["edge_min"]
EDGE_THRESH_FAV      = CFG["edge_min_favorite"]
FAVORITE_ODDS_CUTOFF = CFG["favorite_odds_cutoff"]
MIN_ODDS             = CFG["min_odds"]
MAX_ODDS             = CFG["max_odds"]
KELLY_FRACTION       = CFG["kelly_fraction"]
MAX_STAKE            = CFG["max_stake_units"]
ALLOWED_SURFACES     = set(CFG["allowed_surfaces"])
ALLOWED_TIERS        = set(CFG["allowed_tiers"])

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
matches_df = pd.DataFrame(matches_raw)
matches_df = matches_df[matches_df["match_date"].str.startswith("2024", na=False)]

tours_raw = paginate_all("tournaments", "tournament_id,surface,tier")
tours_df  = pd.DataFrame(tours_raw)
tours_df.rename(columns={"surface": "surface_name", "tier": "tier_name"}, inplace=True)
matches_df = matches_df.merge(tours_df, on="tournament_id", how="left")

ctx_lookup = matches_df.set_index("match_id")[["surface_name", "tier_name"]].to_dict("index")

# ── Build bet candidates ──────────────────────────────────────────────────────
odds_by_match_player = {(r["match_id"], r["player_id"]): r for r in odds_raw}

bet_rows = []
for _, pred in preds.iterrows():
    mid   = pred["match_id"]
    w_id  = pred["winner_id"]
    l_id  = pred["loser_id"]
    p_win = pred["model_win_prob"]

    ctx          = ctx_lookup.get(mid, {})
    surface_name = ctx.get("surface_name")
    tier_name    = ctx.get("tier_name")

    for player_id, model_prob, actual_won in [
        (w_id, p_win,       1),
        (l_id, 1 - p_win,   0),
    ]:
        key  = (mid, player_id)
        odds = odds_by_match_player.get(key)
        if not odds:
            continue
        closing  = float(odds["closing_odds"])
        imp_prob = float(odds["implied_prob"])
        edge     = model_prob - imp_prob
        bet_rows.append({
            "match_id":         mid,
            "match_date":       pred["match_date"],
            "player_id":        player_id,
            "model_prob":       round(model_prob, 4),
            "pinnacle_odds":    closing,
            "pinnacle_implied": imp_prob,
            "edge":             round(edge, 4),
            "actual_won":       actual_won,
            "surface_name":     surface_name,
            "tier_name":        tier_name,
        })

joined = pd.DataFrame(bet_rows)
print(f"  {len(joined)} raw bet candidates")

# ── Apply filters ─────────────────────────────────────────────────────────────
# 1. Minimum edge — higher bar for heavy favorites
edge_ok = (
      (joined["pinnacle_odds"] >= FAVORITE_ODDS_CUTOFF) & (joined["edge"] > EDGE_THRESH)
    | (joined["pinnacle_odds"] <  FAVORITE_ODDS_CUTOFF) & (joined["edge"] > EDGE_THRESH_FAV)
)
# 2. Odds window
odds_ok = (joined["pinnacle_odds"] >= MIN_ODDS) & (joined["pinnacle_odds"] <= MAX_ODDS)
# 3. Allowed surfaces only
surf_ok = joined["surface_name"].isin(ALLOWED_SURFACES)
# 4. Allowed tiers only
tier_ok = joined["tier_name"].isin(ALLOWED_TIERS)

joined["bet"] = edge_ok & odds_ok & surf_ok & tier_ok

# ── Half-Kelly stake sizing ───────────────────────────────────────────────────
# Kelly fraction = edge / (odds - 1)
# Half-Kelly stake = 0.5 * Kelly, capped at MAX_STAKE
net_odds = joined["pinnacle_odds"] - 1
joined["kelly_stake"] = np.where(
    joined["bet"] & (net_odds > 0),
    (KELLY_FRACTION * joined["edge"] / net_odds).clip(upper=MAX_STAKE),
    0.0,
)

joined["pnl"] = np.where(
    joined["bet"],
    np.where(joined["actual_won"] == 1,
             joined["kelly_stake"] * (joined["pinnacle_odds"] - 1),
             -joined["kelly_stake"]),
    0.0,
)

# ── Results ───────────────────────────────────────────────────────────────────
bets = joined[joined["bet"]].copy()

print(f"\n{'='*60}")
print(f"Config: {CFG_PATH}")
print(f"Surfaces: {sorted(ALLOWED_SURFACES)}  Tiers: {sorted(ALLOWED_TIERS)}")
print(f"Odds: [{MIN_ODDS}, {MAX_ODDS}]  Edge: >{EDGE_THRESH*100:.0f}%  "
      f"(>{EDGE_THRESH_FAV*100:.0f}% if odds<{FAVORITE_ODDS_CUTOFF})  staking=half-Kelly x{KELLY_FRACTION}")
print(f"{'='*60}")
print(f"Total opportunities:  {len(joined)}")
print(f"Bets placed:          {len(bets)}")

if len(bets) == 0:
    print("No bets placed.")
else:
    total_staked = bets["kelly_stake"].sum()
    win_rate     = bets["actual_won"].mean()
    total_pnl    = bets["pnl"].sum()
    roi          = total_pnl / total_staked * 100 if total_staked > 0 else 0
    avg_odds     = bets["pinnacle_odds"].mean()
    avg_stake    = bets["kelly_stake"].mean()
    clv          = (bets["model_prob"] - bets["pinnacle_implied"]).mean() * 100

    print(f"Win rate:             {win_rate:.3f}")
    print(f"Total staked:         {total_staked:.2f} units")
    print(f"Total PnL:            {total_pnl:+.2f} units")
    print(f"ROI (on staked):      {roi:+.2f}%")
    print(f"Avg closing odds:     {avg_odds:.3f}")
    print(f"Avg stake (Kelly):    {avg_stake:.3f}")
    print(f"Avg CLV:              {clv:+.3f}%")

    # Monthly
    bets["month"] = pd.to_datetime(bets["match_date"]).dt.to_period("M")
    monthly = (bets.groupby("month")
               .agg(bets=("pnl","count"), staked=("kelly_stake","sum"), pnl=("pnl","sum"))
               .reset_index())
    monthly["roi_pct"] = (monthly["pnl"] / monthly["staked"] * 100).round(1)
    print("\nMonthly PnL:")
    print(monthly.to_string(index=False))

    # By surface
    surf = (bets.groupby("surface_name")
            .agg(bets=("pnl","count"),
                 staked=("kelly_stake","sum"),
                 pnl=("pnl","sum"),
                 win_rate=("actual_won","mean"),
                 avg_edge=("edge","mean"))
            .reset_index())
    surf["roi_pct"] = (surf["pnl"] / surf["staked"] * 100).round(1)
    print("\nBy surface:")
    print(surf.to_string(index=False))

    # By tier
    tier = (bets.groupby("tier_name")
            .agg(bets=("pnl","count"),
                 staked=("kelly_stake","sum"),
                 pnl=("pnl","sum"),
                 win_rate=("actual_won","mean"),
                 avg_edge=("edge","mean"))
            .reset_index())
    tier["roi_pct"] = (tier["pnl"] / tier["staked"] * 100).round(1)
    print("\nBy tier:")
    print(tier.to_string(index=False))

    # By odds band
    bins   = [1.0, 1.5, 2.0, 2.5, 3.0]
    labels = ["1.0-1.5","1.5-2.0","2.0-2.5","2.5-3.0"]
    bets["odds_band"] = pd.cut(bets["pinnacle_odds"], bins=bins, labels=labels)
    band = (bets.groupby("odds_band", observed=True)
            .agg(bets=("pnl","count"),
                 staked=("kelly_stake","sum"),
                 pnl=("pnl","sum"),
                 win_rate=("actual_won","mean"))
            .reset_index())
    band["roi_pct"] = (band["pnl"] / band["staked"] * 100).round(1)
    print("\nBy odds band:")
    print(band.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
out_cols = ["match_id","match_date","player_id","model_prob","pinnacle_odds",
            "pinnacle_implied","edge","kelly_stake","bet","actual_won","pnl",
            "surface_name","tier_name"]
joined[out_cols].to_csv("./backtest_results.csv", index=False)
print(f"\nSaved backtest_results.csv ({len(joined)} rows)")
print("backtest.py complete.")
