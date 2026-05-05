"""
train_match_winner.py - Model 1: Match Winner.
Train: 2015-2022  |  Val: 2023  |  Test: 2024
Saves: atp_model_v2.pkl, test_predictions_v2.csv
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

FEATURE_COLS = [
    # Core Elo / ranking
    "d_elo_overall", "d_elo_surface", "d_atp_rank",
    # Win-rate & form
    "d_surface_win_rate", "d_recent_form_10",
    # Serve / return
    "d_serve_rating", "d_return_rating", "serve_vs_return",
    "d_ace_rate", "d_bp_save_rate",
    # Fatigue
    "d_days_rest", "d_matches_last_14d", "p1_matches_14d", "p2_matches_14d",
    "p1_fatigue_games_7d", "p2_fatigue_games_7d", "d_fatigue_games_7d",
    # H2H
    "h2h_overall", "h2h_surface", "h2h_surface_3y", "h2h_last5_weighted",
    # New v2 (elo_mkt_gap excluded: uses closing odds → leakage in training)
    "d_serve_style_csi",
    "p1_surface_transition", "p2_surface_transition", "d_surface_transition",
    "serve_hold_matchup",
    "p1_form_regression", "p2_form_regression",
    "tournament_proximity",
    # Context
    "court_speed_index", "round_numeric", "best_of", "tier_numeric",
    "surface_Hard", "surface_Clay", "surface_Grass",
    "tier_Grand_Slam", "tier_Masters_1000", "tier_ATP_500",
    "tier_ATP_250", "tier_Challenger",
]
TARGET = "winner_won"

print("Loading features_v2.csv ...")
df = pd.read_csv("./features_v2.csv", low_memory=False)
df["year"] = pd.to_datetime(df["match_date"]).dt.year
print(f"  {len(df)} rows,  years {df['year'].min()}–{df['year'].max()}")

# Verify all feature columns exist
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"WARNING: missing columns: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

train = df[df["year"] <= 2022].copy()
val   = df[df["year"] == 2023].copy()
test  = df[df["year"] == 2024].copy()
print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

medians = train[FEATURE_COLS].median().fillna(0.0)
for split in [train, val, test]:
    split[FEATURE_COLS] = split[FEATURE_COLS].fillna(medians).fillna(0.0)

X_train, y_train = train[FEATURE_COLS].values, train[TARGET].values
X_val,   y_val   = val[FEATURE_COLS].values,   val[TARGET].values
X_test,  y_test  = test[FEATURE_COLS].values,  test[TARGET].values

print("\nFitting ensemble ...")
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
])
xgb = HistGradientBoostingClassifier(
    max_iter=300, max_depth=4, learning_rate=0.05, random_state=42,
)
rf  = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=10,
    random_state=42, n_jobs=-1,
)
ensemble   = VotingClassifier(
    estimators=[("lr", lr), ("xgb", xgb), ("rf", rf)],
    voting="soft", weights=[1, 2, 1],
)
calibrated = CalibratedClassifierCV(ensemble, method="sigmoid", cv=5)
calibrated.fit(X_train, y_train)
print("  Done.")

def evaluate(name, X, y):
    proba = calibrated.predict_proba(X)[:, 1]
    acc   = accuracy_score(y, (proba >= 0.5).astype(int))
    brier = brier_score_loss(y, proba)
    ll    = log_loss(y, proba)
    print(f"  {name}: accuracy={acc:.4f}  brier={brier:.4f}  logloss={ll:.4f}")
    return proba

print("\nEvaluation:")
evaluate("Val  2023", X_val,  y_val)
test_proba = evaluate("Test 2024", X_test, y_test)

# Save test predictions
test_out = test[["match_id", "match_date", "winner_id", "loser_id", "p1_is_winner"]].copy()
test_out["model_win_prob"] = test_proba
test_out = test_out[test_out["p1_is_winner"] == 1].drop(columns=["p1_is_winner"])
test_out.to_csv("./test_predictions_v2.csv", index=False)
print(f"\n  Saved test_predictions_v2.csv ({len(test_out)} rows)")

# Save model
payload = {
    "model":        calibrated,
    "feature_cols": FEATURE_COLS,
    "medians":      medians.to_dict(),
    "version":      "v2",
}
joblib.dump(payload, "./atp_model_v2.pkl")
print("  Saved atp_model_v2.pkl")
print("\ntrain_match_winner.py complete.")
