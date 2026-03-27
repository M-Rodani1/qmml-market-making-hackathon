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
# 1. Submit bid/ask → tightest spread = market maker (BAD)
# 2. MM quotes revealed to everyone
# 3. You CHOOSE to buy at MM's ask or sell at MM's bid
# 4. You CHOOSE your sizing (minimum required)
# 5. Cash goes negative = eliminated
#
# STRATEGY:
#   - Always submit bid=80, ask=400 → never be MM
#   - Use prediction to decide BUY vs SELL once MM quotes are revealed
#   - Size based on confidence: big on stocks 1-2, minimum on stocks 6-9
#   - Never risk enough to get eliminated
#

STARTING_CASH = 100_000

# =============================================================================
# STOCK-SPECIFIC MODEL CONFIGS
# =============================================================================

STOCK_CONFIGS = {
    1: {'model': 'linear',   'feature_thresh': 0.05},
    2: {'model': 'linear',   'feature_thresh': 0.05},
    3: {'model': 'linear',   'feature_thresh': 0.05},
    4: {'model': 'gbr',      'feature_thresh': 0.05},
    5: {'model': 'ridge100', 'feature_thresh': 0.05},
    6: {'model': 'mean',     'feature_thresh': None},
    7: {'model': 'ridge100', 'feature_thresh': 0.05},
    8: {'model': 'ridge100', 'feature_thresh': 0.05},
    9: {'model': 'mean',     'feature_thresh': None},
}

# =============================================================================
# STEP 1: BUILD PREDICTOR
# =============================================================================

def build_predictor(stock_id, train, test):
    X = train.drop('target', axis=1)
    y = train['target']
    n = len(train)
    config = STOCK_CONFIGS[stock_id]

    if config['model'] == 'mean':
        return y.mean(), y.std(), 0.0

    corrs = train.corr()['target'].drop('target')
    sig_cols = list(corrs[corrs.abs() > config['feature_thresh']].index)
    if not sig_cols:
        sig_cols = list(X.columns)

    X_clean = X[sig_cols]
    test_clean = test[sig_cols]

    if config['model'] == 'linear':
        model = LinearRegression()
    elif config['model'] == 'ridge100':
        model = Ridge(alpha=100)
    elif config['model'] == 'gbr':
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )

    cv_folds = min(5, n)
    scores = cross_val_score(model, X_clean, y, cv=cv_folds, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())

    model.fit(X_clean, y)
    r2 = model.score(X_clean, y)
    prediction = model.predict(test_clean)[0]
    prediction = np.clip(prediction, y.min(), y.max())

    return prediction, rmse, r2

# =============================================================================
# STEP 2: SIZING — how much to bet based on confidence
# =============================================================================
#
# Kelly-inspired sizing:
#   - Your edge = how accurate your prediction is (R², RMSE)
#   - Never risk more than X% of remaining cash on one trade
#   - Scale size with confidence
#

def get_size_pct(r2, rmse, prediction):
    """Return what % of current cash to risk on this stock."""

    if r2 > 0.95:
        return 0.25      # very high confidence — 25% of cash
    elif r2 > 0.90:
        return 0.15       # high confidence — 15%
    elif r2 > 0.50:
        return 0.10       # moderate — 10%
    elif r2 > 0.10:
        return 0.05       # low — 5%
    else:
        return 0.02       # no signal — trade minimum (2%)

# =============================================================================
# STEP 3: DECISION LOGIC (called AFTER seeing MM quotes)
# =============================================================================

def decide(prediction, mm_bid, mm_ask):
    """
    After seeing MM quotes, decide direction.
    Buy at MM ask if prediction > ask (underpriced).
    Sell at MM bid if prediction < bid (overpriced).
    If prediction between bid and ask, pick closer edge, trade minimum.
    """
    mm_mid = (mm_bid + mm_ask) / 2

    if prediction > mm_ask:
        return "BUY", mm_ask, prediction - mm_ask   # expected profit per unit
    elif prediction < mm_bid:
        return "SELL", mm_bid, mm_bid - prediction   # expected profit per unit
    else:
        # Prediction inside the spread — weak signal
        if prediction > mm_mid:
            return "BUY (weak)", mm_ask, prediction - mm_ask
        else:
            return "SELL (weak)", mm_bid, mm_bid - prediction

# =============================================================================
# STEP 4: RUN — predictions + sizing plan
# =============================================================================

print("=" * 80)
print("SUBMISSION: bid=80, ask=400 for ALL stocks (never be the MM)")
print(f"Starting cash: £{STARTING_CASH:,}")
print("=" * 80)

print(f"\n{'Stock':>5} {'Model':>10} {'Pred':>8} {'RMSE':>8} {'R²':>8} {'Size%':>7} {'MaxBet':>10}")
print("-" * 65)

cash = STARTING_CASH
predictions = {}

for i in range(1, 10):
    train = pd.read_csv(f'stock_{i}_train.csv')
    test = pd.read_csv(f'stock_{i}_test.csv')

    prediction, rmse, r2 = build_predictor(i, train, test)
    size_pct = get_size_pct(r2, rmse, prediction)
    max_bet = cash * size_pct

    predictions[i] = {
        'prediction': prediction,
        'rmse': rmse,
        'r2': r2,
        'size_pct': size_pct,
        'max_bet': max_bet
    }

    model_name = STOCK_CONFIGS[i]['model']
    print(f"{i:>5} {model_name:>10} {prediction:>8.2f} {rmse:>8.2f} {r2:>8.4f} {size_pct*100:>6.0f}% £{max_bet:>9,.0f}")

print(f"\n{'='*80}")
print("LIVE DECISION GUIDE")
print("When MM quotes are revealed, use this logic:")
print(f"{'='*80}")
print(f"\n{'Stock':>5} {'Prediction':>10} {'Action':>45}")
print("-" * 65)

for i in range(1, 10):
    p = predictions[i]
    pred = p['prediction']
    pct = p['size_pct']

    if p['r2'] > 0.90:
        conf = "BIG"
    elif p['r2'] > 0.10:
        conf = "SMALL"
    else:
        conf = "MINIMUM"

    print(f"{i:>5} {pred:>10.2f}   If pred > MM ask → BUY  ({conf} size)")
    print(f"{'':>5} {'':>10}   If pred < MM bid → SELL ({conf} size)")