"""
train_model.py - trains soft-voting ensemble with Platt calibration.
Chronological split: train 2015-2022, validate 2023, test 2024.
Saves model to atp_model.pkl.
"""
import os
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
TARGET = "winner_won"

print("Loading features.csv ...")
df = pd.read_csv("./features.csv", low_memory=False)
print(f"  {len(df)} rows loaded")

df["year"] = pd.to_datetime(df["match_date"]).dt.year

train = df[df["year"] <= 2022].copy()
val   = df[df["year"] == 2023].copy()
test  = df[df["year"] == 2024].copy()
print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Fill missing with median of training set
medians = train[FEATURE_COLS].median()
for split in [train, val, test]:
    split[FEATURE_COLS] = split[FEATURE_COLS].fillna(medians)

X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values
X_val   = val[FEATURE_COLS].values
y_val   = val[TARGET].values
X_test  = test[FEATURE_COLS].values
y_test  = test[TARGET].values

# ── Base estimators ───────────────────────────────────────────────────────────
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
])

xgb = HistGradientBoostingClassifier(
    max_iter=300, max_depth=4, learning_rate=0.05,
    random_state=42,
)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=10,
    random_state=42, n_jobs=-1,
)

# ── Soft-voting ensemble ──────────────────────────────────────────────────────
print("Training ensemble ...")
ensemble = VotingClassifier(
    estimators=[("lr", lr), ("xgb", xgb), ("rf", rf)],
    voting="soft",
    weights=[1, 2, 1],
)

# Platt scaling via CalibratedClassifierCV
calibrated = CalibratedClassifierCV(ensemble, method="sigmoid", cv=5)
calibrated.fit(X_train, y_train)
print("  Training complete")


def evaluate(name, X, y, model):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc   = accuracy_score(y, preds)
    brier = brier_score_loss(y, proba)
    ll    = log_loss(y, proba)
    print(f"  {name}: accuracy={acc:.4f}  brier={brier:.4f}  logloss={ll:.4f}")
    return proba


print("\nEvaluation:")
val_proba  = evaluate("Val  2023", X_val,  y_val,  calibrated)
test_proba = evaluate("Test 2024", X_test, y_test, calibrated)

# ── Save predictions for backtest ─────────────────────────────────────────────
# Keep only the "P1=winner" rows (p1_is_winner==1) so backtest sees one row per match
# with model_win_prob = prob that the actual winner wins.
test_out = test[["match_id", "match_date", "winner_id", "loser_id",
                 "p1_is_winner",
                 "surface_Hard", "surface_Clay", "surface_Grass",
                 "tier_Grand_Slam", "tier_Masters_1000", "tier_ATP_500",
                 "tier_ATP_250", "tier_Challenger"]].copy()
test_out["model_win_prob"] = test_proba
# Filter to winner-as-P1 rows only
test_out = test_out[test_out["p1_is_winner"] == 1].drop(columns=["p1_is_winner"])
test_out.to_csv("./test_predictions.csv", index=False)
print(f"\n  Saved test_predictions.csv ({len(test_out)} rows)")

# ── Save model ────────────────────────────────────────────────────────────────
model_payload = {
    "model":        calibrated,
    "feature_cols": FEATURE_COLS,
    "medians":      medians.to_dict(),
}
joblib.dump(model_payload, "./atp_model.pkl")
print("  Saved atp_model.pkl")
print("\ntrain_model.py complete.")
