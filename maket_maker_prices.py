import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import norm, t, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# settings
# =============================================================================

STARTING_CASH = 100_000
MIN_SHARES = 10
MAX_SHARES = 500  

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

    if config['features'] == 'all':
        feat_cols = list(X.columns)
    elif config['features'] == 'significant':
        corrs = train.corr()['target'].drop('target')
        feat_cols = list(corrs[corrs.abs() > 0.05].index)
        if not feat_cols: feat_cols = list(X.columns)
    else:
        feat_cols = config['features']

    X_clean, test_clean = X[feat_cols], test[feat_cols]

    if config['model'] == 'linear':
        model = LinearRegression()
    elif config['model'] == 'ridge100':
        model = Ridge(alpha=100)
    elif config['model'] == 'gbr':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)

    cv_folds = min(5, len(train))
    scores = cross_val_score(model, X_clean, y, cv=cv_folds, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())

    model.fit(X_clean, y)
    prediction = np.clip(model.predict(test_clean)[0], y.min(), y.max())

    return prediction, rmse

# =============================================================================
# market maker price generation
# =============================================================================
def get_mm_quotes(prediction, rmse, n_rows):
    """
    Generates Bid/Ask quotes for different risk profiles.
    Returns: Dict of profiles with Bid, Ask, Spread, and Probability of success.
    """
    profiles = {
        'CONSERVATIVE (2 stds)': 1.96,  # 95% CI
        'MODERATE (1 std)':     1.00,  # 1 std
        'MID-TIER (0.8 std)': 0.8,  # 0.8 std
        'MID-TIER (0.75 std)': 0.75,  # 0.8 std
        'MID-TIER (0.7 std)': 0.7,  # 0.8 std
        'MID-TIER (0.65 std)': 0.65,  # 0.8 std
        'MID-TIER (0.60 std)': 0.6,  # 0.8 std
        'MID-TIER (0.55 std)': 0.55,  # 0.8 std
        'AGGRESSIVE (0.5 std)':     0.50   # 0.5 std
    }
    
    quotes = {}
    dof = n_rows - 1
    dist = t(df=dof, loc=prediction, scale=rmse) if n_rows < 100 else norm(loc=prediction, scale=rmse)

    for name, mult in profiles.items():
        bid = prediction - (mult * rmse)
        ask = prediction + (mult * rmse)
        prob_inside = dist.cdf(ask) - dist.cdf(bid)
        
        ev_inside = mult * rmse
        
        quotes[name] = {
            'bid': round(bid, 2),
            'ask': round(ask, 2),
            'spread': round(ask - bid, 2),
            'prob': prob_inside
        }
    return quotes

def get_mm_quotes_advanced(prediction, rmse, train_targets, n_rows):
    """
    Generates Bid/Ask quotes adjusted for Skewness (bias) and Kurtosis (fat tails).
    """
    s = skew(train_targets)
    k = kurtosis(train_targets)
    
    # If skew is positive, outliers are to the upside -> shift quotes slightly higher.
    # If skew is negative, outliers are to the downside -> shift quotes slightly lower.
    skew_shift = (s * 0.1) * rmse
    adjusted_center = prediction + skew_shift
     
    # If k > 1, the tails are 'fat'. We widen the spread to protect against outliers.
    k_buffer = 1.0
    if k > 1:
        k_buffer = 1 + (np.log1p(k) * 0.2) # Dampened scaling for high kurtosis

    profiles = {
        '2 stds': 2.00,
        '1 stds':     1.00,
        '0.8 stds':     0.8,
        '0.75 stds':     0.75,
        '0.70 stds':     0.7,
        '0.65 stds':     0.65,
        '0.6 stds':     0.6,
        '0.55 stds':     0.55,
        '0.5 stds':     0.50 
    }
    
    quotes = {}
    dof = n_rows - 1
    dist = t(df=dof, loc=prediction, scale=rmse) if n_rows < 100 else norm(loc=prediction, scale=rmse)

    for name, mult in profiles.items():
        half_width = mult * rmse * k_buffer
        
        bid = adjusted_center - half_width
        ask = adjusted_center + half_width
        
        prob_inside = dist.cdf(ask) - dist.cdf(bid)
        
        quotes[name] = {
            'bid': round(bid, 2),
            'ask': round(ask, 2),
            'spread': round(ask - bid, 2),
            'prob': prob_inside,
            'skew': s,
            'kurt': k
        }
    return quotes

# =============================================================================
# script
# =============================================================================

print("=" * 60)
print("MARKET MAKER QUOTES")
print("=" * 60)

while True:
    stock_input = input("\nWhich Stock ID to analyze? (1-9 or 'exit'): ").lower()
    if stock_input == 'exit': break

    try:
        i = int(stock_input)
        train = pd.read_csv(f'data/stock_{i}_train.csv')
        test = pd.read_csv(f'data/stock_{i}_test.csv')
        pred, rmse = build_predictor(i, train, test)
        
        mm_options = get_mm_quotes(pred, rmse, len(train))
        mm_options2 = get_mm_quotes_advanced(pred, rmse, train['target'], len(train))

        first_key = list(mm_options2.keys())[0]
        print(f"Stats -> Skew: {mm_options2[first_key]['skew']:.2f} | Kurtosis: {mm_options2[first_key]['kurt']:.2f}")  

        print(f"\n--- SUGGESTED QUOTES (no adjustment)---")
        for profile, data in mm_options.items():
            print(f"{profile:25} | BID: {data['bid']:<8} ASK: {data['ask']:<8} | Spread: {data['spread']:<6} | Confidence: {data['prob']:.1%}")

        
        print(f"\n--- SUGGESTED QUOTES (skew and kurtosis adjusted)---")
        for profile, data in mm_options2.items():
            print(f"{profile:25} | BID: {data['bid']:<8} ASK: {data['ask']:<8} | Spread: {data['spread']:<6} | Confidence: {data['prob']:.1%}")

    except Exception as e:
        print(f"Error: {e}")
