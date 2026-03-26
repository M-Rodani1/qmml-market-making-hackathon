import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Load data
train = pd.read_csv('data/stock_1_train.csv')
test = pd.read_csv('data/stock_1_test.csv')

X_train = train.drop('target', axis=1)
y_train = train['target']

# Step 1: Check correlations with target
print("=== FEATURE CORRELATIONS ===")
print(train.corr()['target'].drop('target'))

# Step 2: Fit baseline model (all features)
lr_all = LinearRegression()
lr_all.fit(X_train, y_train)

scores = cross_val_score(lr_all, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_all = np.sqrt(-scores.mean())

print(f"\n=== BASELINE MODEL (all 5 features) ===")
print(f"CV RMSE: {rmse_all:.4f}")
print(f"R²: {lr_all.score(X_train, y_train):.6f}")
print(f"Intercept: {lr_all.intercept_:.4f}")
for col, coef in zip(X_train.columns, lr_all.coef_):
    print(f"  {col}: {coef:.4f}")

# Step 3: Drop noise features, fit clean model
X_clean = X_train[['col_0', 'col_1', 'col_2']]
lr_clean = LinearRegression()
lr_clean.fit(X_clean, y_train)

scores = cross_val_score(lr_clean, X_clean, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_clean = np.sqrt(-scores.mean())

print(f"\n=== CLEAN MODEL (col_0, col_1, col_2 only) ===")
print(f"CV RMSE: {rmse_clean:.4f}")
print(f"R²: {lr_clean.score(X_clean, y_train):.6f}")
print(f"Intercept: {lr_clean.intercept_:.4f}")
for col, coef in zip(X_clean.columns, lr_clean.coef_):
    print(f"  {col}: {coef:.4f}")

# Step 4: Residual analysis — check for nonlinearity
residuals = y_train - lr_clean.predict(X_clean)

print(f"\n=== RESIDUAL ANALYSIS ===")
print(f"Residual mean: {residuals.mean():.4f}")
print(f"Residual std: {residuals.std():.4f}")

# Check residuals vs squared features
print(f"\nCorrelation of residuals with squared features:")
for col in ['col_0', 'col_1', 'col_2']:
    corr = np.corrcoef(residuals, X_train[col]**2)[0, 1]
    print(f"  resid vs {col}²: {corr:.4f}")

# Check residuals vs interactions
print(f"\nCorrelation of residuals with interactions:")
pairs = [('col_0','col_1'), ('col_0','col_2'), ('col_1','col_2')]
for c1, c2 in pairs:
    interaction = X_train[c1] * X_train[c2]
    corr = np.corrcoef(residuals, interaction)[0, 1]
    print(f"  resid vs {c1}*{c2}: {corr:.4f}")

# Step 5: Predict on test
test_clean = test[['col_0', 'col_1', 'col_2']]
pred_all = lr_all.predict(test)[0]
pred_clean = lr_clean.predict(test_clean)[0]

print(f"\n=== TEST PREDICTIONS ===")
print(f"All features model:   {pred_all:.4f}")
print(f"Clean 3-feature model: {pred_clean:.4f}")