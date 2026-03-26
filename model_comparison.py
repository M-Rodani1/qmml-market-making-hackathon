import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

for i in range(3, 10):
    train = pd.read_csv(f'stock_{i}_train.csv')
    test = pd.read_csv(f'stock_{i}_test.csv')
    X = train.drop('target', axis=1)
    y = train['target']
    n = len(train)
    cv = min(5, n)
    
    print(f"\n{'='*50}")
    print(f"STOCK {i} | {n} rows | {len(X.columns)} features")
    print(f"{'='*50}")
    
    models = {}
    
    # Baseline: predict the mean
    models['Mean'] = y.std()
    
    # Linear
    s = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='neg_mean_squared_error')
    models['Linear'] = np.sqrt(-s.mean())
    
    # Ridge
    for alpha in [1, 10, 100]:
        s = cross_val_score(Ridge(alpha=alpha), X, y, cv=cv, scoring='neg_mean_squared_error')
        models[f'Ridge(a={alpha})'] = np.sqrt(-s.mean())
    
    # GBR and RF — skip if fewer than 100 rows
    if n >= 100:
        # For large datasets, subsample to 2000 rows to keep CV fast
        if n > 2000:
            idx = np.random.default_rng(42).choice(n, 2000, replace=False)
            X_cv, y_cv = X.iloc[idx], y.iloc[idx]
        else:
            X_cv, y_cv = X, y

        gb = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        s = cross_val_score(gb, X_cv, y_cv, cv=cv, scoring='neg_mean_squared_error')
        models['GBR'] = np.sqrt(-s.mean())

        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        s = cross_val_score(rf, X_cv, y_cv, cv=cv, scoring='neg_mean_squared_error')
        models['RF'] = np.sqrt(-s.mean())
    
    # Sort by RMSE
    sorted_m = sorted(models.items(), key=lambda x: x[1])
    for name, rmse in sorted_m:
        tag = " <-- BEST" if name == sorted_m[0][0] else ""
        if name == "Mean": tag += " (baseline)"
        print(f"  {name:15s} RMSE: {rmse:.2f}{tag}")
    
    best_rmse = sorted_m[0][1]
    mean_rmse = models['Mean']
    print(f"\n  Best beats mean baseline by {(mean_rmse - best_rmse) / mean_rmse * 100:.1f}%")