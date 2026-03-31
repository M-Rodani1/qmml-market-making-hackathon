import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GAME MECHANICS
# =============================================================================
#
# 1. Submit bid=80, ask=400 every round → NEVER be the market maker
# 2. MM quotes revealed → you SEE their bid and ask
# 3. You CHOOSE to buy at MM's ask or sell at MM's bid
# 4. You CHOOSE your sizing (minimum 10 shares)
# 5. Cash goes negative = eliminated
#
# STRATEGY:
#   Tier 1 (stocks 1-2): high confidence → BIG size
#   Tier 2 (stock 3): moderate confidence → MODERATE size
#   Tier 3 (stocks 4-9): low/no confidence → MINIMUM size (10 shares)
#

STARTING_CASH = 100_000
MIN_SHARES = 10

# =============================================================================
# STOCK-SPECIFIC MODEL CONFIGS (final, validated)
# =============================================================================

STOCK_CONFIGS = {
    1: {'model': 'linear',   'features': ['col_0', 'col_1', 'col_2']},
    2: {'model': 'linear',   'features': 'all'},
    3: {'model': 'linear',   'features': ['col_0', 'col_1', 'col_2']},
    4: {'model': 'gbr',      'features': 'significant'},
    5: {'model': 'ridge100', 'features': 'all'},
    6: {'model': 'mean',     'features': None},
    7: {'model': 'ridge100', 'features': 'all'},
    8: {'model': 'ridge100', 'features': 'all'},
    9: {'model': 'mean',     'features': None},
}

# =============================================================================
# STEP 1: BUILD PREDICTOR
# =============================================================================

def build_predictor(stock_id, train, test):
    X = train.drop('target', axis=1)
    y = train['target']
    n = len(train)
    config = STOCK_CONFIGS[stock_id]

    # Mean baseline — no model needed
    if config['model'] == 'mean':
        return y.mean(), y.std(), 0.0

    # Feature selection
    if config['features'] == 'all':
        feat_cols = list(X.columns)
    elif config['features'] == 'significant':
        corrs = train.corr()['target'].drop('target')
        feat_cols = list(corrs[corrs.abs() > 0.05].index)
        if not feat_cols:
            feat_cols = list(X.columns)
    else:
        feat_cols = config['features']

    X_clean = X[feat_cols]
    test_clean = test[feat_cols]

    # Model selection
    if config['model'] == 'linear':
        model = LinearRegression()
    elif config['model'] == 'ridge100':
        model = Ridge(alpha=100)
    elif config['model'] == 'gbr':
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )

    # CV RMSE
    cv_folds = min(5, n)
    scores = cross_val_score(model, X_clean, y, cv=cv_folds, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())

    # Fit and predict
    model.fit(X_clean, y)
    r2 = model.score(X_clean, y)
    prediction = model.predict(test_clean)[0]
    prediction = np.clip(prediction, y.min(), y.max())

    return prediction, rmse, r2

# =============================================================================
# STEP 2: SIZING
# =============================================================================
#
# Tier 1 (R² > 0.90): high confidence
#   Max loss if wrong by 2×RMSE: shares × 2×RMSE
#   Target: risk up to 15% of cash → shares = 0.15 × cash / (2 × RMSE)
#
# Tier 2 (R² > 0.50): moderate confidence
#   Target: risk up to 5% of cash → shares = 0.05 × cash / (2 × RMSE)
#
# Tier 3 (R² <= 0.50): low/no confidence
#   Minimum 10 shares, no more
#

def get_shares(prediction, rmse, r2, cash, n_rows):
    """Calculate number of shares to trade."""

    if r2 > 0.90 and n_rows >= 500:
        # Tier 1: high confidence AND enough data to trust it
        risk_budget = 0.15 * cash
        max_loss_per_share = 2 * rmse  # worst case ~2 sigma move
        shares = int(risk_budget / max_loss_per_share)
        tier = "TIER 1 — BIG"
    elif r2 > 0.50 or (r2 > 0.90 and n_rows < 500):
        # Tier 2: moderate confidence — risk up to 5% of cash
        risk_budget = 0.05 * cash
        max_loss_per_share = 2 * rmse
        shares = int(risk_budget / max_loss_per_share)
        tier = "TIER 2 — MODERATE"
    else:
        # Tier 3: low/no confidence — minimum only
        shares = MIN_SHARES
        tier = "TIER 3 — MINIMUM"

    # Floor at minimum, cap to avoid catastrophic loss
    shares = max(MIN_SHARES, shares)

    # Safety cap: never risk more than 25% of cash on one trade
    max_shares = int(0.25 * cash / max(prediction, 1))
    shares = min(shares, max_shares)

    return shares, tier

# =============================================================================
# STEP 3: LIVE DECISION LOGIC
# =============================================================================

def decide(prediction, mm_bid, mm_ask):
    """
    Called AFTER seeing MM quotes.
    Returns direction and expected profit per share.
    """
    mm_mid = (mm_bid + mm_ask) / 2

    if prediction > mm_ask:
        direction = "BUY"
        trade_price = mm_ask
        expected_profit = prediction - mm_ask
    elif prediction < mm_bid:
        direction = "SELL"
        trade_price = mm_bid
        expected_profit = mm_bid - prediction
    else:
        # Prediction inside spread — weak signal
        if prediction > mm_mid:
            direction = "BUY (weak)"
            trade_price = mm_ask
            expected_profit = prediction - mm_ask  # will be negative
        else:
            direction = "SELL (weak)"
            trade_price = mm_bid
            expected_profit = mm_bid - prediction  # will be negative

    return direction, trade_price, expected_profit

# =============================================================================
# STEP 4: RUN — FULL STRATEGY OUTPUT
# =============================================================================

print("=" * 85)
print("QMML MARKET MAKING HACKATHON — ALPHABETAPHI STRATEGY")
print("=" * 85)
print(f"\nStarting cash: £{STARTING_CASH:,}")
print(f"Submission every round: BID = 80, ASK = 400 (never be the MM)")

print(f"\n{'Stock':>5} {'Model':>10} {'Pred':>8} {'RMSE':>8} {'R²':>6} {'Shares':>7} {'Tier':>22}")
print("-" * 75)

cash = STARTING_CASH
all_predictions = {}

for i in range(1, 10):
    train = pd.read_csv(f'data/stock_{i}_train.csv')
    test = pd.read_csv(f'data/stock_{i}_test.csv')

    prediction, rmse, r2 = build_predictor(i, train, test)
    shares, tier = get_shares(prediction, rmse, r2, cash, len(train))

    model_name = STOCK_CONFIGS[i]['model']
    print(f"{i:>5} {model_name:>10} {prediction:>8.2f} {rmse:>8.2f} {r2:>6.2f} {shares:>7} {tier:>22}")

    all_predictions[i] = {
        'prediction': prediction,
        'rmse': rmse,
        'r2': r2,
        'shares': shares,
        'tier': tier
    }

print(f"\n{'='*45}")
print("LIVE DECISION CHEAT SHEET")
print("="*45)
print(f"\nWhen MM quotes are revealed each round:\n")

for i in range(1, 10):
    p = all_predictions[i]
    pred = p['prediction']
    shares = p['shares']
    rmse = p['rmse']

    print(f"STOCK {i}: prediction = {pred:.2f} (±{rmse:.2f})")
    print(f"  If MM ask < {pred:.2f} → BUY {shares} shares at MM ask")
    print(f"  If MM bid > {pred:.2f} → SELL {shares} shares at MM bid")
    print(f"  If prediction inside spread → trade {MIN_SHARES} shares toward prediction")
    print()

print("="*85)
print("RISK LIMITS")
print("="*85)
print(f"\n  • Never let any single trade risk more than 25% of current cash")
print(f"  • Stocks 4-9: ALWAYS trade minimum {MIN_SHARES} shares regardless of signal")
print(f"  • If cash drops below £50,000: reduce ALL sizes by half")
print(f"  • If cash drops below £25,000: trade minimum {MIN_SHARES} on everything")
print(f"  • Goal: SURVIVE all 9 rounds, profit on stocks 1-2")