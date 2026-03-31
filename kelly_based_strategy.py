import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import norm, t
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# settings
# =============================================================================

STARTING_CASH = 100_000
MIN_SHARES = 10
MAX_SHARES = 500  # safety cap

# =============================================================================
# strategy.py methodology
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

def build_predictor(stock_id, train, test):
    X = train.drop('target', axis=1)
    y = train['target']
    config = STOCK_CONFIGS[stock_id]

    if config['model'] == 'mean':
        return y.mean(), y.std()

    # Feature selection
    if config['features'] == 'all':
        feat_cols = list(X.columns)
    elif config['features'] == 'significant':
        corrs = train.corr()['target'].drop('target')
        feat_cols = list(corrs[corrs.abs() > 0.05].index)
        if not feat_cols: feat_cols = list(X.columns)
    else:
        feat_cols = config['features']

    X_clean, test_clean = X[feat_cols], test[feat_cols]

    # Model Selection
    if config['model'] == 'linear':
        model = LinearRegression()
    elif config['model'] == 'ridge100':
        model = Ridge(alpha=100)
    elif config['model'] == 'gbr':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)

    # RMSE
    cv_folds = min(5, len(train))
    scores = cross_val_score(model, X_clean, y, cv=cv_folds, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())

    model.fit(X_clean, y)
    prediction = np.clip(model.predict(test_clean)[0], y.min(), y.max())

    return prediction, rmse

# =============================================================================
# confidence probabilites
# =============================================================================
def get_market_probabilities(prediction, rmse, mm_bid, mm_ask, n_rows):
    """
    Calculates probabilities using T-Distribution for small samples 
    and Normal Distribution for large samples.
    """
    dof = n_rows - 1

    if n_rows < 100:
        dist = t(df=dof, loc=prediction, scale=rmse)
        dist_type = f"T-Dist (df={dof})"
    else:
        dist = norm(loc=prediction, scale=rmse)
        dist_type = "Normal Dist"

    prob_lower = dist.cdf(mm_bid)
    prob_higher = 1 - dist.cdf(mm_ask)
    prob_inside = dist.cdf(mm_ask) - dist.cdf(mm_bid)
    
    return prob_lower, prob_higher, prob_inside, dist_type

# =============================================================================
# market bet size
# =============================================================================
def get_kelly_size(p_win, prediction, mm_price, rmse, current_cash, profile='conservative'):
    """Calculates optimal position sizing based on risk profile."""
    if p_win <= 0.50:
        return MIN_SHARES
    
    q = 1 - p_win
    gain = abs(prediction - mm_price)
    loss = rmse

    if profile == 'conservative':
        loss, multiplier = rmse, 0.5    # half-kelly, professional fund manager approach
    elif profile == 'aggressive':
        loss, multiplier = rmse, 1.0    # full-kelly, Full RMSE risk
    elif profile == 'super aggressive':
        loss, multiplier = rmse * 0.4, 1.0  # full-kelly and ignores 60% of noise

    b = gain / max(loss, 0.01)
    
    # kelly criteria formula
    kelly_f = p_win - (q / b)
    
    safe_f = kelly_f * multiplier
    
    if safe_f <= 0:
        return MIN_SHARES
        
    suggested_shares = (safe_f * current_cash) / mm_price
    return np.clip(suggested_shares, MIN_SHARES, MAX_SHARES)

# also can consider maximising expected utility
# =============================================================================
# script
# =============================================================================

print("=" * 45)
print("RISK ANALYSIS")
print("=" * 45)

while True:
    stock_input = input("\nWhich Stock ID to analyze? (1-9 or 'exit'): ").lower()
    
    if stock_input == 'exit':
        break

    try:
        i = int(stock_input)
        if i not in STOCK_CONFIGS:
            print("Invalid Stock ID. Choose 1-9.")
            continue

        train = pd.read_csv(f'data/stock_{i}_train.csv')
        test = pd.read_csv(f'data/stock_{i}_test.csv')
        n_rows = len(train)
        pred, rmse = build_predictor(i, train, test)

        print(f"\nSTOCK {i} ANALYSIS:")
        print(f"Model Prediction: {pred:.2f} | Model RMSE: {rmse:.2f}")

        cash =  float(input("  Current Bankroll (£): "))
        m_bid = float(input(f"  Enter MM BID for Stock {i}: "))
        m_ask = float(input(f"  Enter MM ASK for Stock {i}: "))

        # get confidence probabilities
        lower, higher, inside, dist_type = get_market_probabilities(pred, rmse, m_bid, m_ask, n_rows)

        if higher > lower:
            p_win = higher
            mm_price = m_ask
            direction = "BUY"
        else:
            p_win = lower
            mm_price = m_bid
            direction = "SELL"
        
        size_con = get_kelly_size(p_win, pred, mm_price, rmse, cash, 'conservative')
        size_agg = get_kelly_size(p_win, pred, mm_price, rmse, cash, 'aggressive')
        size_super_agg = get_kelly_size(p_win, pred, mm_price, rmse, cash, 'super aggressive')

        print("-" * 45)
        print(f"  Calculation Method: {dist_type} | Edge: {p_win:.1%}")
        print("-" * 45)
        print(f"  Probability TRUE value is LOWER than Bid:  {lower:.3%}")
        print(f"  Probability TRUE value is HIGHER than Ask: {higher:.3%}")
        print(f"  Probability MM is CORRECT (Inside Spread): {inside:.3%}")
        print()

        print("  kelly recommendation:")

        print(f"  CONSERVATIVE:      {direction} {size_con:.2f} shares")
        print(f"  AGGRESSIVE:        {direction} {size_agg:.2f} shares")
        print(f"  SUPER AGGRESSIVE:  {direction} {size_super_agg:.2f} shares")



    except FileNotFoundError:
        print(f"Stock {i}: Data files not found. Skipping...")
    except Exception as e:
        print(f"Error on Stock {i}: {e}")