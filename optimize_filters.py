"""
optimize_filters.py - grid search over filter combinations.

Parameter space (61,425 total combinations):
  surfaces:        all non-empty subsets of {Hard, Clay, Grass}          7
  tiers:           all non-empty subsets of {Masters 1000, ATP 500,     15
                   ATP 250, Grand Slam}
  odds bands:      lower in [1.50..2.50] x upper in [2.00..4.00]        39
                   step 0.25, lower < upper
  edge thresholds: 3%, 4%, 5%, 6%, 7%                                    5
  staking:         flat, half-Kelly, quarter-Kelly                        3

Metrics per combination: bets, win_rate, roi, clv, sharpe
Min sample size: 50 bets (combinations below this are skipped).
Output: optimization_results.csv (all passing rows, sorted by ROI desc)
"""
import itertools
import time
import os
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
MIN_BETS     = 50
MAX_STAKE    = 5.0
OUTPUT_CSV   = "./optimization_results.csv"
TOP_N        = 20

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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data (done once)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading predictions ...")
preds = pd.read_csv("./test_predictions.csv", low_memory=False)
print(f"  {len(preds)} rows")

print("Loading Pinnacle odds ...")
odds_raw = paginate_all("odds", "match_id,player_id,closing_odds,implied_prob",
                        filters={"bookmaker": "Pinnacle"})
odds_df = pd.DataFrame(odds_raw)
print(f"  {len(odds_df)} odds rows")

print("Loading tournament context ...")
matches_raw = paginate_all("matches",
                           "match_id,match_date,winner_id,loser_id,tournament_id",
                           order="match_date")
matches_df = pd.DataFrame(matches_raw)
matches_df = matches_df[matches_df["match_date"].str.startswith("2024", na=False)]

tours_raw = paginate_all("tournaments", "tournament_id,surface,tier")
tours_df  = pd.DataFrame(tours_raw)
tours_df.rename(columns={"surface": "surface_name", "tier": "tier_name"}, inplace=True)
matches_df = matches_df.merge(tours_df, on="tournament_id", how="left")
ctx_lookup  = matches_df.set_index("match_id")[["surface_name", "tier_name"]].to_dict("index")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build full bet-candidates DataFrame (done once)
# ─────────────────────────────────────────────────────────────────────────────
print("Building bet candidates ...")
odds_idx = {(r["match_id"], r["player_id"]): r for r in odds_raw}

rows = []
for _, pred in preds.iterrows():
    mid   = pred["match_id"]
    w_id  = pred["winner_id"]
    l_id  = pred["loser_id"]
    p_win = float(pred["model_win_prob"])

    ctx          = ctx_lookup.get(mid, {})
    surface_name = ctx.get("surface_name")
    tier_name    = ctx.get("tier_name")

    for player_id, model_prob, actual_won in [
        (w_id, p_win,       1),
        (l_id, 1.0 - p_win, 0),
    ]:
        o = odds_idx.get((mid, player_id))
        if not o:
            continue
        closing  = float(o["closing_odds"])
        imp_prob = float(o["implied_prob"])
        net_odds = closing - 1.0
        edge     = model_prob - imp_prob

        # Precompute Kelly stakes (capped)
        kelly_full = (edge / net_odds) if (net_odds > 0 and edge > 0) else 0.0
        stake_half    = min(0.5  * kelly_full, MAX_STAKE)
        stake_quarter = min(0.25 * kelly_full, MAX_STAKE)

        rows.append({
            "match_id":         mid,
            "match_date":       pred["match_date"],
            "surface_name":     surface_name,
            "tier_name":        tier_name,
            "pinnacle_odds":    closing,
            "pinnacle_implied": imp_prob,
            "edge":             edge,
            "model_prob":       model_prob,
            "actual_won":       actual_won,
            "stake_flat":       1.0,
            "stake_half_kelly": stake_half,
            "stake_qtr_kelly":  stake_quarter,
        })

candidates = pd.DataFrame(rows)
print(f"  {len(candidates)} candidates built")

# Convert to numpy arrays for fast masking
arr_surface  = candidates["surface_name"].to_numpy()
arr_tier     = candidates["tier_name"].to_numpy()
arr_odds     = candidates["pinnacle_odds"].to_numpy()
arr_edge     = candidates["edge"].to_numpy()
arr_won      = candidates["actual_won"].to_numpy()
arr_model    = candidates["model_prob"].to_numpy()
arr_implied  = candidates["pinnacle_implied"].to_numpy()
arr_s_flat   = candidates["stake_flat"].to_numpy()
arr_s_half   = candidates["stake_half_kelly"].to_numpy()
arr_s_qtr    = candidates["stake_qtr_kelly"].to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Define parameter grid
# ─────────────────────────────────────────────────────────────────────────────
ALL_SURFACES = ["Hard", "Clay", "Grass"]
ALL_TIERS    = ["Masters 1000", "ATP 500", "ATP 250", "Grand Slam"]

surface_combos = []
for r in range(1, len(ALL_SURFACES) + 1):
    for combo in itertools.combinations(ALL_SURFACES, r):
        surface_combos.append(frozenset(combo))

tier_combos = []
for r in range(1, len(ALL_TIERS) + 1):
    for combo in itertools.combinations(ALL_TIERS, r):
        tier_combos.append(frozenset(combo))

lower_bounds = np.arange(1.50, 2.51, 0.25).round(2)   # 1.50 1.75 2.00 2.25 2.50
upper_bounds = np.arange(2.00, 4.01, 0.25).round(2)   # 2.00 2.25 ... 4.00
odds_bands   = [(lo, hi) for lo in lower_bounds for hi in upper_bounds if hi > lo + 1e-9]

edge_thresholds = [0.03, 0.04, 0.05, 0.06, 0.07]

stakings = [
    ("flat",         arr_s_flat),
    ("half-Kelly",   arr_s_half),
    ("quarter-Kelly",arr_s_qtr),
]

total = len(surface_combos) * len(tier_combos) * len(odds_bands) * len(edge_thresholds) * len(stakings)
print(f"\nGrid size: {len(surface_combos)} surfaces x {len(tier_combos)} tiers x "
      f"{len(odds_bands)} bands x {len(edge_thresholds)} edges x {len(stakings)} stakes "
      f"= {total:,} combinations")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Precompute surface and tier boolean masks (reused across inner loops)
# ─────────────────────────────────────────────────────────────────────────────
surf_masks: dict = {}
for sc in surface_combos:
    mask = np.zeros(len(candidates), dtype=bool)
    for s in sc:
        mask |= (arr_surface == s)
    surf_masks[sc] = mask

tier_masks: dict = {}
for tc in tier_combos:
    mask = np.zeros(len(candidates), dtype=bool)
    for t in tc:
        mask |= (arr_tier == t)
    tier_masks[tc] = mask


# ─────────────────────────────────────────────────────────────────────────────
# 5. Grid search
# ─────────────────────────────────────────────────────────────────────────────
print("Running grid search ...")
t0 = time.time()

results = []
done = 0

for sc in surface_combos:
    m_surf = surf_masks[sc]
    for tc in tier_combos:
        m_tier = tier_masks[tc]
        m_st   = m_surf & m_tier            # surface + tier mask (reused per odds/edge)

        for (lo, hi) in odds_bands:
            m_odds = m_st & (arr_odds >= lo) & (arr_odds <= hi)
            if m_odds.sum() == 0:
                done += len(edge_thresholds) * len(stakings)
                continue

            for edge_thresh in edge_thresholds:
                m_edge = m_odds & (arr_edge > edge_thresh)
                n_cand = m_edge.sum()
                if n_cand < MIN_BETS:
                    done += len(stakings)
                    continue

                # Precompute shared arrays for this (surface, tier, odds, edge) bucket
                won_e   = arr_won[m_edge]
                model_e = arr_model[m_edge]
                impl_e  = arr_implied[m_edge]

                for staking_name, stake_arr in stakings:
                    done += 1
                    stakes_e = stake_arr[m_edge]

                    # Skip if all stakes zero (can happen for Kelly when all edge<=0)
                    total_staked = stakes_e.sum()
                    if total_staked < 1e-9:
                        continue

                    n_bets = int(n_cand)
                    if n_bets < MIN_BETS:
                        continue

                    # PnL per bet
                    pnl_e = np.where(
                        won_e == 1,
                        stakes_e * (arr_odds[m_edge] - 1),
                        -stakes_e,
                    )

                    total_pnl = pnl_e.sum()
                    roi       = total_pnl / total_staked * 100
                    win_rate  = won_e.mean()
                    clv       = (model_e - impl_e).mean() * 100

                    # Per-unit return for Sharpe
                    unit_ret  = pnl_e / stakes_e   # return relative to stake per bet
                    sharpe    = (unit_ret.mean() / unit_ret.std() * np.sqrt(n_bets)
                                 if unit_ret.std() > 1e-9 else 0.0)

                    results.append({
                        "surfaces":       "+".join(sorted(sc)),
                        "tiers":          "+".join(sorted(tc)),
                        "odds_min":       lo,
                        "odds_max":       hi,
                        "edge_thresh":    edge_thresh,
                        "staking":        staking_name,
                        "bets":           n_bets,
                        "win_rate":       round(win_rate, 4),
                        "roi":            round(roi, 2),
                        "clv":            round(clv, 3),
                        "sharpe":         round(sharpe, 3),
                        "total_pnl":      round(total_pnl, 3),
                        "total_staked":   round(total_staked, 3),
                    })

        if done % 5000 == 0 or done == total:
            elapsed = time.time() - t0
            pct     = done / total * 100
            eta     = (elapsed / done * (total - done)) if done > 0 else 0
            print(f"  {done:>7,}/{total:,}  ({pct:5.1f}%)  "
                  f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s  "
                  f"results so far: {len(results):,}",
                  flush=True)

elapsed = time.time() - t0
print(f"\nGrid search complete in {elapsed:.1f}s")
print(f"Combinations with >= {MIN_BETS} bets: {len(results):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save and print
# ─────────────────────────────────────────────────────────────────────────────
if not results:
    print("No combinations met the minimum sample size.")
else:
    df_res = pd.DataFrame(results).sort_values("roi", ascending=False).reset_index(drop=True)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_res):,} rows to {OUTPUT_CSV}")

    print(f"\n{'='*110}")
    print(f"TOP {TOP_N} COMBINATIONS BY ROI")
    print(f"{'='*110}")
    cols = ["surfaces", "tiers", "odds_min", "odds_max",
            "edge_thresh", "staking", "bets", "win_rate", "roi", "clv", "sharpe"]
    top = df_res.head(TOP_N)[cols].copy()
    top["edge_thresh"] = (top["edge_thresh"] * 100).astype(int).astype(str) + "%"
    top["win_rate"]    = (top["win_rate"] * 100).round(1).astype(str) + "%"
    top["roi"]         = top["roi"].astype(str) + "%"
    print(top.to_string(index=True))

    # Summary stats
    print(f"\n{'='*60}")
    print("Summary across all valid combinations:")
    print(f"  Median ROI:   {df_res['roi'].median():+.2f}%")
    print(f"  % positive:   {(df_res['roi'] > 0).mean()*100:.1f}%")
    print(f"  Best ROI:     {df_res['roi'].max():+.2f}%  "
          f"({df_res.loc[df_res['roi'].idxmax(), 'bets']} bets)")
    print(f"  Best Sharpe:  {df_res['sharpe'].max():.3f}  "
          f"(ROI {df_res.loc[df_res['sharpe'].idxmax(), 'roi']:+.2f}%)")
    print(f"  Best CLV:     {df_res['clv'].max():+.3f}%")

    # Breakdown of best by staking method
    print("\nBest ROI by staking method:")
    for stk in ["flat", "half-Kelly", "quarter-Kelly"]:
        sub = df_res[df_res["staking"] == stk]
        if not sub.empty:
            best = sub.iloc[0]
            print(f"  {stk:<16}  ROI {best['roi']:>+7}%  "
                  f"bets {best['bets']:>4}  sharpe {best['sharpe']:.3f}  "
                  f"surfaces {best['surfaces']}  tiers {best['tiers']}")

    # Update betting_config.json with best combination
    best = df_res.iloc[0]
    best_cfg = {
        "edge_min":              float(best["edge_thresh"].replace("%","")) / 100
                                 if isinstance(best["edge_thresh"], str)
                                 else float(best["edge_thresh"]),
        "edge_min_favorite":     0.07,
        "favorite_odds_cutoff":  1.5,
        "min_odds":              float(best["odds_min"]),
        "max_odds":              float(best["odds_max"]),
        "kelly_fraction":        0.5  if best["staking"] == "half-Kelly"
                                 else 0.25 if best["staking"] == "quarter-Kelly"
                                 else None,
        "max_stake_units":       5.0,
        "allowed_surfaces":      sorted(best["surfaces"].split("+")),
        "allowed_tiers":         sorted(best["tiers"].split("+")),
        "staking":               best["staking"],
        "_optimized_roi":        float(str(best["roi"]).replace("%","")),
        "_optimized_bets":       int(best["bets"]),
        "_optimized_sharpe":     float(best["sharpe"]),
    }
    # Only write flat-key for kelly_fraction when staking is Kelly-based
    if best_cfg["kelly_fraction"] is None:
        best_cfg["kelly_fraction"] = 1.0  # flat = full 1-unit stake

    print(f"\nBest config written → betting_config.json")
    print(json.dumps(best_cfg, indent=2))
    with open("./betting_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
