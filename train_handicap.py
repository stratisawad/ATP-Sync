"""
train_handicap.py - Model 2: Games Handicap.

Predicts game_diff = winner_games - loser_games.
Outputs: atp_handicap_model.pkl

The model predicts the EXPECTED game differential (regression).
For probability over a handicap line L, we model the residuals as Gaussian
and compute P(actual_diff > L) = 1 - CDF((L - predicted_mean) / residual_std).

Train: 2015-2022  |  Val: 2023  |  Test: 2024
"""
import joblib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLS = [
    "d_elo_surface", "elo_win_prob",
    "court_speed_index", "tier_numeric", "best_of",
    "surface_Hard", "surface_Clay", "surface_Grass",
    "p1_hold_surface", "p2_hold_surface", "d_hold_surface",
    "p1_ace_rate", "p2_ace_rate", "d_ace_rate",
    "p1_first_serve_pct", "p2_first_serve_pct", "d_first_serve_pct",
    "p1_avg_games_surface", "p2_avg_games_surface",
    "p1_fatigue_games_7d", "p2_fatigue_games_7d", "d_fatigue_games_7d",
]
TARGET = "game_diff"

print("Loading handicap_features.csv ...")
df = pd.read_csv("./handicap_features.csv", low_memory=False)
df["year"] = pd.to_datetime(df["match_date"]).dt.year
print(f"  {len(df)} rows,  years {df['year'].min()}–{df['year'].max()}")
print(f"  Target: game_diff  mean={df[TARGET].mean():.2f}  std={df[TARGET].std():.2f}")

avail = [c for c in FEATURE_COLS if c in df.columns]
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"  WARNING: missing features: {missing}")
FEATURE_COLS = avail

train = df[df["year"] <= 2022].copy()
val   = df[df["year"] == 2023].copy()
test  = df[df["year"] == 2024].copy()
print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

medians = train[FEATURE_COLS].median()
for split in [train, val, test]:
    split[FEATURE_COLS] = split[FEATURE_COLS].fillna(medians)

X_train, y_train = train[FEATURE_COLS].values, train[TARGET].values
X_val,   y_val   = val[FEATURE_COLS].values,   val[TARGET].values
X_test,  y_test  = test[FEATURE_COLS].values,  test[TARGET].values

print("\nFitting HistGradientBoostingRegressor ...")
model = HistGradientBoostingRegressor(
    max_iter=400, max_depth=4, learning_rate=0.05,
    min_samples_leaf=15, random_state=42,
)
model.fit(X_train, y_train)
print("  Done.")


def evaluate(name, X, y):
    preds = model.predict(X)
    mae   = mean_absolute_error(y, preds)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    # Baseline: always predict mean
    base_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
    corr, _  = scipy_stats.pearsonr(y, preds)
    print(f"  {name}:  MAE={mae:.3f}  RMSE={rmse:.3f}  "
          f"vs_baseline={base_mae:.3f}  corr={corr:.3f}")
    return preds


print("\nEvaluation:")
val_preds  = evaluate("Val  2023", X_val,  y_val)
test_preds = evaluate("Test 2024", X_test, y_test)

# Estimate residual std from validation set (for probability computation)
val_residuals = y_val - val_preds
residual_std  = float(np.std(val_residuals))
print(f"\n  Residual std (val 2023): {residual_std:.3f}")
print(f"  This is used for P(actual_diff > handicap_line) = "
      f"1 - Φ((line - predicted) / {residual_std:.2f})")

# Show how well model predicts handicap coverage
for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
    actual_pct = (y_test > line).mean() * 100
    pred_pct   = (test_preds > line).mean() * 100
    print(f"  Handicap {line:+.1f}: actual {actual_pct:.1f}%  model {pred_pct:.1f}%")

# Save model
payload = {
    "model":        model,
    "feature_cols": FEATURE_COLS,
    "medians":      medians.to_dict(),
    "residual_std": residual_std,
    "target_mean":  float(y_train.mean()),
    "target_std":   float(y_train.std()),
    "version":      "v1",
}
joblib.dump(payload, "./atp_handicap_model.pkl")
print("\n  Saved atp_handicap_model.pkl")
print("train_handicap.py complete.")
