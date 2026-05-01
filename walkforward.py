"""
walkforward.py - Walk-forward validation of top filter combinations.

Trains model on 2015-2021 only (strictly before filter optimization period),
generates out-of-sample predictions for 2022, 2023, 2024, then evaluates
each of the top-10 filter combos on each year independently.

Ranking: consistency (positive years count), then average ROI.
Robustness threshold: positive ROI in >= 2 out of 3 years.
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TEST_YEARS   = [2022, 2023, 2024]
TOP_N        = 10
MIN_BETS_PER_YEAR = 20   # flag thin samples per year (lower than global 50)
MAX_STAKE    = 5.0

FEATURE_COLS = [
    "d_elo_overall", "d_elo_surface", "d_atp_rank",
    "d_surface_win_rate", "d_recent_form_10",
    "d_serve_rating", "d_return_rating", "serve_vs_return",
    "d_ace_rate", "d_bp_save_rate",
    "d_days_rest", "d_matches_last_14d", "p1_matches_14d", "p2_matches_14d",
    "h2h_overall", "h2h_surface", "h2h_surface_3y",
    "court_speed_index", "round_numeric", "best_of", "tier_numeric",
    "surface_Hard", "surface_Clay", "surface_Grass",
    "tier_Grand_Slam", "tier_Masters_1000", "tier_ATP_500",
    "tier_ATP_250", "tier_Challenger",
]

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
# 1. Train model on 2015-2021 only
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Training model on 2015-2021 (pre-optimization period)")
print("=" * 70)

print("Loading features.csv ...")
df = pd.read_csv("./features.csv", low_memory=False)
df["year"] = pd.to_datetime(df["match_date"]).dt.year
print(f"  {len(df)} rows, years {df['year'].min()}-{df['year'].max()}")

train = df[df["year"] <= 2021].copy()
tests = {yr: df[df["year"] == yr].copy() for yr in TEST_YEARS}
print(f"  Train (≤2021): {len(train)} rows")
for yr in TEST_YEARS:
    print(f"  Test  {yr}:      {len(tests[yr])} rows")

medians = train[FEATURE_COLS].median()
train[FEATURE_COLS] = train[FEATURE_COLS].fillna(medians)
for yr in TEST_YEARS:
    tests[yr][FEATURE_COLS] = tests[yr][FEATURE_COLS].fillna(medians)

X_train = train[FEATURE_COLS].values
y_train = train["winner_won"].values

print("\nFitting ensemble ...")
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
])
xgb = HistGradientBoostingClassifier(
    max_iter=300, max_depth=4, learning_rate=0.05, random_state=42,
)
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=10,
    random_state=42, n_jobs=-1,
)
ensemble   = VotingClassifier(
    estimators=[("lr", lr), ("xgb", xgb), ("rf", rf)],
    voting="soft", weights=[1, 2, 1],
)
calibrated = CalibratedClassifierCV(ensemble, method="sigmoid", cv=5)
calibrated.fit(X_train, y_train)
print("  Training complete.")

print("\nModel accuracy on test years:")
preds_by_year = {}
for yr in TEST_YEARS:
    t = tests[yr]
    X = t[FEATURE_COLS].values
    y = t["winner_won"].values
    proba = calibrated.predict_proba(X)[:, 1]
    acc   = accuracy_score(y, (proba >= 0.5).astype(int))
    print(f"  {yr}: accuracy = {acc:.4f}")

    t_out = t[["match_id", "match_date", "winner_id", "loser_id", "p1_is_winner"]].copy()
    t_out["model_win_prob"] = proba
    winner_rows = (t_out[t_out["p1_is_winner"] == 1]
                   [["match_id", "match_date", "winner_id", "loser_id", "model_win_prob"]]
                   .copy())
    preds_by_year[yr] = winner_rows
    print(f"          {len(winner_rows)} match predictions saved")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load odds + tournament context (all years)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Loading odds and tournament context")
print("=" * 70)

print("Loading Pinnacle odds ...")
odds_raw = paginate_all("odds", "match_id,player_id,closing_odds,implied_prob",
                        filters={"bookmaker": "Pinnacle"})
odds_idx = {(r["match_id"], r["player_id"]): r for r in odds_raw}
print(f"  {len(odds_raw)} rows loaded")

print("Loading matches and tournament metadata ...")
all_matches = paginate_all("matches",
                           "match_id,match_date,winner_id,loser_id,tournament_id",
                           order="match_date")
matches_df = pd.DataFrame(all_matches)

tours_raw = paginate_all("tournaments", "tournament_id,surface,tier")
tours_df  = pd.DataFrame(tours_raw)
tours_df.rename(columns={"surface": "surface_name", "tier": "tier_name"}, inplace=True)
matches_df = matches_df.merge(tours_df, on="tournament_id", how="left")
ctx_lookup = matches_df.set_index("match_id")[["surface_name", "tier_name"]].to_dict("index")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build bet candidates per year
# ─────────────────────────────────────────────────────────────────────────────
def build_candidates(preds_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, pred in preds_df.iterrows():
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

            kelly_full = (edge / net_odds) if (net_odds > 0 and edge > 0) else 0.0
            rows.append({
                "surface_name":     surface_name,
                "tier_name":        tier_name,
                "pinnacle_odds":    closing,
                "edge":             edge,
                "actual_won":       actual_won,
                "stake_flat":       1.0,
                "stake_half_kelly": min(0.5  * kelly_full, MAX_STAKE),
                "stake_qtr_kelly":  min(0.25 * kelly_full, MAX_STAKE),
            })
    return pd.DataFrame(rows)


print("\nBuilding bet candidates per year ...")
candidates_by_year = {}
for yr in TEST_YEARS:
    c = build_candidates(preds_by_year[yr])
    candidates_by_year[yr] = c
    odds_hit = len(c) // 2
    print(f"  {yr}: {len(c)} candidates  ({odds_hit} matches with Pinnacle odds)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Load top 10 filter combinations from optimization_results.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Walk-forward evaluation of top-10 filter combinations")
print("=" * 70)

opt  = pd.read_csv("./optimization_results.csv")
top10 = opt.head(TOP_N).copy().reset_index(drop=True)
print(f"Loaded top {len(top10)} combinations from optimization_results.csv\n")


def evaluate_combo(cands: pd.DataFrame,
                   surfaces_set, tiers_set,
                   odds_lo, odds_hi, edge_thresh, staking):
    mask = (
        cands["surface_name"].isin(surfaces_set) &
        cands["tier_name"].isin(tiers_set) &
        (cands["pinnacle_odds"] >= odds_lo) &
        (cands["pinnacle_odds"] <= odds_hi) &
        (cands["edge"] > edge_thresh)
    )
    bets = cands[mask].copy()
    if staking == "flat":
        bets["stake"] = bets["stake_flat"]
    elif staking == "half-Kelly":
        bets["stake"] = bets["stake_half_kelly"]
    else:                                          # quarter-Kelly
        bets["stake"] = bets["stake_qtr_kelly"]

    n_bets       = len(bets)
    total_staked = bets["stake"].sum() if n_bets > 0 else 0.0

    if n_bets == 0 or total_staked < 1e-9:
        return {"n": 0, "roi": None, "win_rate": None}

    pnl = np.where(
        bets["actual_won"] == 1,
        bets["stake"] * (bets["pinnacle_odds"] - 1),
        -bets["stake"],
    ).sum()
    return {
        "n":        n_bets,
        "roi":      round(pnl / total_staked * 100, 2),
        "win_rate": round(bets["actual_won"].mean() * 100, 1),
    }


summary_rows = []

for idx, row in top10.iterrows():
    surfaces_set = set(row["surfaces"].split("+"))
    tiers_set    = set(row["tiers"].split("+"))
    odds_lo      = float(row["odds_min"])
    odds_hi      = float(row["odds_max"])
    edge_thresh  = float(row["edge_thresh"])
    staking      = row["staking"]

    yr_res = {yr: evaluate_combo(candidates_by_year[yr],
                                 surfaces_set, tiers_set,
                                 odds_lo, odds_hi, edge_thresh, staking)
              for yr in TEST_YEARS}

    positive_years = sum(1 for yr in TEST_YEARS
                         if yr_res[yr]["roi"] is not None and yr_res[yr]["roi"] > 0)
    years_with_bets = sum(1 for yr in TEST_YEARS if yr_res[yr]["n"] > 0)

    valid_rois = [yr_res[yr]["roi"] for yr in TEST_YEARS if yr_res[yr]["roi"] is not None]
    avg_roi = round(np.mean(valid_rois), 2) if valid_rois else None

    summary_rows.append({
        "orig_rank":     idx + 1,
        "surfaces":      row["surfaces"],
        "tiers":         row["tiers"],
        "odds_min":      odds_lo,
        "odds_max":      odds_hi,
        "edge_thresh":   edge_thresh,
        "staking":       staking,
        "opt_roi_2024":  float(row["roi"]),
        **{f"n_{yr}":   yr_res[yr]["n"]       for yr in TEST_YEARS},
        **{f"roi_{yr}": yr_res[yr]["roi"]      for yr in TEST_YEARS},
        **{f"wr_{yr}":  yr_res[yr]["win_rate"] for yr in TEST_YEARS},
        "positive_years": positive_years,
        "avg_roi":        avg_roi,
    })

summary = (pd.DataFrame(summary_rows)
           .sort_values(["positive_years", "avg_roi"], ascending=[False, False])
           .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Print results
# ─────────────────────────────────────────────────────────────────────────────
print(f"{'─'*100}")
print(f"{'#':<3}  {'Surfaces':<14}  {'Tiers':<26}  "
      f"{'Odds':>9}  {'Edge':>5}  {'Staking':>14}  "
      f"{'2022':>10}  {'2023':>10}  {'2024':>10}  "
      f"{'Avg ROI':>8}  {'✓ Yrs':>6}")
print(f"{'─'*100}")

for rank, row in summary.iterrows():
    def fmt_roi(roi, n):
        if roi is None:
            return "   N/A    "
        flag = "*" if n < MIN_BETS_PER_YEAR else " "
        sign = "+" if roi > 0 else ""
        return f"{sign}{roi:+.1f}%({n}){flag}"

    r2022 = fmt_roi(row["roi_2022"], row["n_2022"])
    r2023 = fmt_roi(row["roi_2023"], row["n_2023"])
    r2024 = fmt_roi(row["roi_2024"], row["n_2024"])

    robust = "ROBUST" if row["positive_years"] >= 2 else "      "
    avg_str = f"{row['avg_roi']:+.1f}%" if row["avg_roi"] is not None else "  N/A "

    print(f"{rank+1:<3}  {row['surfaces']:<14}  {row['tiers']:<26}  "
          f"[{row['odds_min']:.2f},{row['odds_max']:.2f}]  "
          f"{row['edge_thresh']*100:.0f}%  {row['staking']:>14}  "
          f"{r2022:>12}  {r2023:>12}  {r2024:>12}  "
          f"{avg_str:>8}  {int(row['positive_years'])}/3  {robust}")

print(f"{'─'*100}")
print(f"* = fewer than {MIN_BETS_PER_YEAR} bets that year (treat with caution)")
print(f"opt_roi was computed on 2024 data during optimization (in-sample for that year).\n")

robust_combos = summary[summary["positive_years"] >= 2]
print(f"ROBUST combinations (positive in ≥2/3 years): {len(robust_combos)}/{len(summary)}")

if len(robust_combos) > 0:
    print("\nTop robust combination:")
    best = robust_combos.iloc[0]
    print(f"  Surfaces: {best['surfaces']}")
    print(f"  Tiers:    {best['tiers']}")
    print(f"  Odds:     [{best['odds_min']:.2f}, {best['odds_max']:.2f}]")
    print(f"  Edge:     >{best['edge_thresh']*100:.0f}%")
    print(f"  Staking:  {best['staking']}")
    print(f"  ROI:  2022={best['roi_2022']:+.1f}% ({int(best['n_2022'])} bets)  "
          f"2023={best['roi_2023']:+.1f}% ({int(best['n_2023'])} bets)  "
          f"2024={best['roi_2024']:+.1f}% ({int(best['n_2024'])} bets)")
    print(f"  Avg ROI across 3 years: {best['avg_roi']:+.1f}%")

summary.to_csv("./walkforward_results.csv", index=False)
print(f"\nFull results saved to walkforward_results.csv")
print("\nwalkforward.py complete.")
