"""
analyze_2023.py - Diagnose the 2023 anomaly in walk-forward validation.

Four questions:
  Q1. Player availability  — top players missing due to injury?
  Q2. Upset distribution   — was 2023 unusually volatile?
  Q3. Market efficiency    — was Pinnacle better calibrated in 2023?
  Q4. Model accuracy       — where was the model systematically wrong in 2023?

Uses saved atp_model.pkl (trained 2015-2022).
  2023 = out-of-sample (validation year)  ← fair
  2024 = out-of-sample (test year)        ← fair
  2022 = in-sample                        ← accuracy overstated; surfaces still informative
"""
import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.metrics import accuracy_score, brier_score_loss

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TEST_YEARS   = [2022, 2023, 2024]
TOP_PLAYERS  = 30     # how many most-active players to track
CALIB_BINS   = 10     # probability buckets for calibration curves

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


def surface_name(row):
    if row["surface_Hard"]:   return "Hard"
    if row["surface_Clay"]:   return "Clay"
    if row["surface_Grass"]:  return "Grass"
    return "Unknown"


def tier_name(row):
    for t in ["tier_Grand_Slam", "tier_Masters_1000", "tier_ATP_500",
              "tier_ATP_250", "tier_Challenger"]:
        if row[t]:
            return t.replace("tier_", "").replace("_", " ")
    return "Other"


SEP = "=" * 72

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("Loading features.csv and model ...")
df = pd.read_csv("./features.csv", low_memory=False)
df["year"]    = pd.to_datetime(df["match_date"]).dt.year
df["surface"] = df.apply(surface_name, axis=1)
df["tier"]    = df.apply(tier_name, axis=1)

# One row per match (winner-as-P1 perspective)
matches = df[df["p1_is_winner"] == 1].copy().reset_index(drop=True)

payload  = joblib.load("./atp_model.pkl")
model    = payload["model"]
medians  = pd.Series(payload["medians"])

df[FEATURE_COLS] = df[FEATURE_COLS].fillna(medians)
matches_filled   = df[df["p1_is_winner"] == 1][FEATURE_COLS].fillna(medians)
matches["model_prob"] = model.predict_proba(matches_filled.values)[:, 1]

print(f"  {len(matches)} matches, {df['year'].min()}-{df['year'].max()}")
print(f"  Model loaded: trained on ≤2022; 2023/2024 are out-of-sample\n")

print("Loading Pinnacle odds ...")
odds_raw = paginate_all("odds", "match_id,player_id,closing_odds,implied_prob",
                        filters={"bookmaker": "Pinnacle"})
odds_df  = pd.DataFrame(odds_raw)
print(f"  {len(odds_df)} rows\n")

print("Loading players ...")
players_raw = paginate_all("players", "player_id,name")
pid_to_name = {r["player_id"]: r["name"] for r in players_raw}


# ─────────────────────────────────────────────────────────────────────────────
# Q1 — Player availability
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("Q1: PLAYER AVAILABILITY (appearances per year)\n")

# Count appearances for each player (winner + loser side)
app_rows = []
for yr in TEST_YEARS:
    yr_m = matches[matches["year"] == yr]
    for pid in pd.concat([yr_m["winner_id"], yr_m["loser_id"]]).unique():
        n = ((yr_m["winner_id"] == pid) | (yr_m["loser_id"] == pid)).sum()
        app_rows.append({"player_id": pid, "year": yr, "matches": n})

app_df = pd.DataFrame(app_rows)
app_wide = (app_df.pivot_table(index="player_id", columns="year",
                                values="matches", fill_value=0)
            .reset_index())
app_wide.columns.name = None
app_wide["name"] = app_wide["player_id"].map(pid_to_name)
# Keep only players who had >= 10 matches in at least one year
mask = app_wide[[2022, 2023, 2024]].max(axis=1) >= 10
app_wide = app_wide[mask].copy()
# Sort by total appearances
app_wide["total"] = app_wide[[2022, 2023, 2024]].sum(axis=1)
app_wide = app_wide.sort_values("total", ascending=False).head(TOP_PLAYERS)

# Flag players with big drops year-over-year
app_wide["drop_22_23"] = app_wide[2022] - app_wide[2023]
app_wide["drop_23_24"] = app_wide[2023] - app_wide[2024]

print(f"{'Player':<28}  {'2022':>5}  {'2023':>5}  {'2024':>5}  "
      f"{'Δ22→23':>8}  {'Δ23→24':>8}")
print("─" * 68)
for _, r in app_wide.iterrows():
    flag_a = " ← BIG DROP" if r["drop_22_23"] >= 15 else ""
    flag_b = " ← BIG DROP" if r["drop_23_24"] >= 15 else ""
    print(f"{str(r['name']):<28}  {int(r[2022]):>5}  {int(r[2023]):>5}  "
          f"{int(r[2024]):>5}  {int(r['drop_22_23']):>+8}  "
          f"{int(r['drop_23_24']):>+8}{flag_a}{flag_b}")

# Total matches per year
print("\nTotal matches in dataset per year:")
for yr in TEST_YEARS:
    n = (matches["year"] == yr).sum()
    print(f"  {yr}: {n} matches")


# ─────────────────────────────────────────────────────────────────────────────
# Q2 — Upset distribution
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("Q2: UPSET DISTRIBUTION BY YEAR AND SURFACE\n")

# An upset = winner had WORSE ATP rank (higher rank number) than loser
# In winner-as-P1 rows: d_atp_rank = p1_rank - p2_rank
# d_atp_rank > 0  →  p1 (winner) ranked worse  →  upset
# Drop rows where d_atp_rank is NaN (no ranking info)
m_ranked = matches.dropna(subset=["d_atp_rank"]).copy()
m_ranked["upset"] = m_ranked["d_atp_rank"] > 0

print("Upset rate (lower-ranked player won) by year:")
upset_yr = (m_ranked.groupby("year")["upset"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "upsets", "count": "total", "mean": "rate"}))
for yr, r in upset_yr.iterrows():
    print(f"  {yr}: {int(r['upsets'])}/{int(r['total'])} = {r['rate']*100:.1f}%")

print("\nUpset rate by year and surface:")
upset_surf = (m_ranked.groupby(["year", "surface"])["upset"]
              .agg(["sum", "count", "mean"])
              .rename(columns={"mean": "rate"}))
for (yr, surf), r in upset_surf.iterrows():
    print(f"  {yr} {surf:<6}: {int(r['sum'])}/{int(r['count'])} = {r['rate']*100:.1f}%")

print("\nUpset rate by year and tier (Masters 1000 / Grand Slam focus):")
upset_tier = (m_ranked[m_ranked["tier"].isin(["Masters 1000", "Grand Slam", "ATP 500"])]
              .groupby(["year", "tier"])["upset"]
              .agg(["sum", "count", "mean"])
              .rename(columns={"mean": "rate"}))
for (yr, tier), r in upset_tier.iterrows():
    print(f"  {yr} {tier:<16}: {int(r['sum'])}/{int(r['count'])} = {r['rate']*100:.1f}%")

# Rank spread of upsets (how big were the upsets?)
print("\nMedian rank gap in upsets (winner_rank - loser_rank, higher = bigger upset):")
for yr in TEST_YEARS:
    upsets_yr = m_ranked[(m_ranked["year"] == yr) & m_ranked["upset"]]
    if len(upsets_yr):
        med = upsets_yr["d_atp_rank"].median()
        p90 = upsets_yr["d_atp_rank"].quantile(0.90)
        print(f"  {yr}: median gap = +{med:.0f}  90th pct = +{p90:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Q3 — Pinnacle market efficiency / calibration
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("Q3: PINNACLE ODDS CALIBRATION BY YEAR\n")

# Join winner-side odds to matches
# odds_df has (match_id, player_id, closing_odds, implied_prob)
# winner side = (match_id, winner_id)
winner_odds = (odds_df.rename(columns={"player_id": "winner_id",
                                        "closing_odds": "w_odds",
                                        "implied_prob": "w_imp"})
               .drop_duplicates(subset=["match_id", "winner_id"]))
m_odds = matches.merge(winner_odds[["match_id", "winner_id", "w_odds", "w_imp"]],
                        on=["match_id", "winner_id"], how="inner")
m_odds["year"] = m_odds["year"].astype(int)

print("Sample sizes with Pinnacle odds:")
for yr in TEST_YEARS:
    n = (m_odds["year"] == yr).sum()
    print(f"  {yr}: {n} matches")

# Calibration: bin by implied prob, compare to actual win rate (always 1 for winner side)
# But we need both sides. Rebuild two-sided for calibration.
loser_odds = (odds_df.rename(columns={"player_id": "loser_id",
                                       "closing_odds": "l_odds",
                                       "implied_prob": "l_imp"})
              .drop_duplicates(subset=["match_id", "loser_id"]))
m_calib = m_odds.merge(loser_odds[["match_id", "loser_id", "l_odds", "l_imp"]],
                        on=["match_id", "loser_id"], how="inner")

# Two-sided: winner row (actual=1, imp=w_imp) + loser row (actual=0, imp=l_imp)
calib_rows = []
for _, r in m_calib.iterrows():
    calib_rows.append({"year": r["year"], "implied": r["w_imp"], "actual": 1})
    calib_rows.append({"year": r["year"], "implied": r["l_imp"], "actual": 0})
calib_df = pd.DataFrame(calib_rows)
calib_df["bucket"] = pd.cut(calib_df["implied"],
                              bins=np.linspace(0, 1, CALIB_BINS + 1),
                              labels=False, include_lowest=True)

print("\nCalibration table (Pinnacle implied prob vs actual win rate):")
print(f"  Bucket centre  | " + "  ".join(f"{yr}" for yr in TEST_YEARS))
print("  " + "─" * 60)

for b in range(CALIB_BINS):
    lo = b / CALIB_BINS
    hi = (b + 1) / CALIB_BINS
    centre = (lo + hi) / 2
    row_parts = []
    for yr in TEST_YEARS:
        sub = calib_df[(calib_df["year"] == yr) & (calib_df["bucket"] == b)]
        if len(sub) >= 5:
            actual_rate = sub["actual"].mean()
            row_parts.append(f"{actual_rate*100:>5.1f}% (n={len(sub):>3})")
        else:
            row_parts.append("   N/A      ")
    print(f"  [{lo:.1f}–{hi:.1f}] ({centre:.2f}) | " + "  ".join(row_parts))

# Brier score per year (lower = better calibrated)
print("\nPinnacle Brier score by year (lower = better calibrated):")
for yr in TEST_YEARS:
    sub = calib_df[calib_df["year"] == yr].copy()
    sub["implied"] = sub["implied"].clip(0, 1)
    bs = brier_score_loss(sub["actual"], sub["implied"])
    print(f"  {yr}: Brier = {bs:.4f}  (n={len(sub)})")

# Mean implied prob of the winner (should be >0.5 if market picks correctly)
print("\nMean Pinnacle implied prob for actual winner by year:")
for yr in TEST_YEARS:
    sub = m_odds[m_odds["year"] == yr]
    print(f"  {yr}: {sub['w_imp'].mean()*100:.1f}%  "
          f"(market favourite won: "
          f"{(sub['w_imp'] > 0.5).mean()*100:.1f}% of matches)")


# ─────────────────────────────────────────────────────────────────────────────
# Q4 — Model accuracy breakdown
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("Q4: MODEL ACCURACY BY YEAR AND SURFACE\n")
print("(2022 is IN-SAMPLE for this model; 2023/2024 are out-of-sample)\n")

matches["pred_correct"] = (matches["model_prob"] >= 0.5).astype(int)

# Accuracy by year
print("Overall accuracy by year:")
for yr in TEST_YEARS:
    sub = matches[matches["year"] == yr]
    acc = sub["pred_correct"].mean()
    bs  = brier_score_loss(sub["winner_won"], sub["model_prob"])
    print(f"  {yr}: accuracy = {acc:.4f}  Brier = {bs:.4f}  (n={len(sub)})")

# Accuracy by year × surface
print("\nAccuracy by year and surface:")
acc_tbl = (matches.groupby(["year", "surface"])["pred_correct"]
           .agg(["mean", "count"])
           .rename(columns={"mean": "accuracy", "count": "n"})
           .reset_index())
for _, r in acc_tbl[acc_tbl["year"].isin(TEST_YEARS)].iterrows():
    print(f"  {int(r['year'])} {r['surface']:<6}: {r['accuracy']:.4f}  (n={int(r['n'])})")

# Accuracy by year × tier
print("\nAccuracy by year and tier:")
acc_tier = (matches[matches["tier"].isin(["Masters 1000", "Grand Slam", "ATP 500", "ATP 250"])]
            .groupby(["year", "tier"])["pred_correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .reset_index())
for _, r in acc_tier[acc_tier["year"].isin(TEST_YEARS)].iterrows():
    print(f"  {int(r['year'])} {r['tier']:<16}: {r['accuracy']:.4f}  (n={int(r['n'])})")

# Where the model was most wrong in 2023: look at high-confidence errors
m23 = matches[matches["year"] == 2023].copy()
m23["confident_wrong"] = ((m23["model_prob"] > 0.65) & (m23["winner_won"] == 0)) | \
                          ((m23["model_prob"] < 0.35) & (m23["winner_won"] == 1))
print(f"\n2023: high-confidence errors (model >65% or <35%, was wrong): "
      f"{m23['confident_wrong'].sum()} / {len(m23)}  "
      f"({m23['confident_wrong'].mean()*100:.1f}%)")

print("\nHigh-confidence errors by surface (2023):")
hce = m23[m23["confident_wrong"]]
for surf, g in hce.groupby("surface"):
    total_surf = (m23["surface"] == surf).sum()
    print(f"  {surf:<6}: {len(g)} errors out of {total_surf}  "
          f"({len(g)/total_surf*100:.1f}% error rate)")

print("\nHigh-confidence errors by tier (2023):")
for tier, g in hce.groupby("tier"):
    total_tier = (m23["tier"] == tier).sum()
    print(f"  {tier:<16}: {len(g)} errors out of {total_tier}  "
          f"({len(g)/total_tier*100:.1f}% error rate)")

# Distribution of model probabilities assigned to actual upsets
print("\nModel probability assigned to the eventual winner — by year:")
print("(lower mean = model was less confident / surprised more often)\n")
for yr in TEST_YEARS:
    sub = matches[matches["year"] == yr]
    print(f"  {yr}:")
    print(f"    Mean model prob for winner:  {sub['model_prob'].mean():.3f}")
    print(f"    % where model was confident (>0.65): "
          f"{(sub['model_prob'] > 0.65).mean()*100:.1f}%")
    print(f"    % where model was wrong + confident: "
          f"{((sub['model_prob'] > 0.65) & (sub['winner_won'] == 0)).mean()*100:.1f}%")

# Focus: betting filter slice (odds 2.25-3.75, our filter zone)
print(f"\n{SEP}")
print("FILTER SLICE: odds [2.25–3.75], edge >5%  (our walk-forward filter)\n")

# m_calib already has model_prob, surface, tier, year from the upstream merge chain.
# Build two-sided candidates directly.
bet_rows = []
for _, r in m_calib.iterrows():
    base_prob = float(r["model_prob"])
    for imp, odds, actual in [
        (r["w_imp"], r["w_odds"], 1),
        (r["l_imp"], r["l_odds"], 0),
    ]:
        model_prob = base_prob if actual == 1 else 1.0 - base_prob
        edge = model_prob - imp
        bet_rows.append({
            "year":       int(r["year"]),
            "surface":    r["surface"],
            "odds":       odds,
            "imp":        imp,
            "model_prob": model_prob,
            "edge":       edge,
            "actual_won": actual,
        })

bet_cands = pd.DataFrame(bet_rows)
filter_bets = bet_cands[
    (bet_cands["odds"] >= 2.25) & (bet_cands["odds"] <= 3.75) &
    (bet_cands["edge"] > 0.05)
].copy()

print("\nTwo-sided bet candidates in filter zone (odds [2.25–3.75], edge >5%):")
print(f"\n{'Year':<6} {'Bets':>5}  {'Win%':>7}  {'Avg Odds':>9}  "
      f"{'Avg Edge':>9}  {'Avg Mkt Impl':>13}  {'Avg Model P':>12}")
print("─" * 72)
for yr in TEST_YEARS:
    sub = filter_bets[filter_bets["year"] == yr]
    if len(sub) == 0:
        print(f"{yr}:  no bets")
        continue
    pnl = np.where(sub["actual_won"] == 1,
                   sub["odds"] - 1, -1.0).sum()
    roi = pnl / len(sub) * 100
    print(f"{yr:<6} {len(sub):>5}  "
          f"{sub['actual_won'].mean()*100:>6.1f}%  "
          f"{sub['odds'].mean():>9.3f}  "
          f"{sub['edge'].mean()*100:>8.2f}%  "
          f"{sub['imp'].mean()*100:>12.1f}%  "
          f"{sub['model_prob'].mean()*100:>11.1f}%  "
          f"ROI={roi:+.1f}%")

print("\nSame breakdown by surface for 2023 specifically:")
for surf in ["Hard", "Clay", "Grass"]:
    sub = filter_bets[(filter_bets["year"] == 2023) & (filter_bets["surface"] == surf)]
    if len(sub) == 0:
        continue
    pnl = np.where(sub["actual_won"] == 1, sub["odds"] - 1, -1.0).sum()
    roi = pnl / len(sub) * 100
    print(f"  2023 {surf:<6}: {len(sub):>3} bets  "
          f"win%={sub['actual_won'].mean()*100:.1f}%  "
          f"avg_odds={sub['odds'].mean():.3f}  ROI={roi:+.1f}%")

print(f"\n{SEP}")
print("analyze_2023.py complete.")
