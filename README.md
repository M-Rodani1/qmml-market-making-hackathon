# QMML Market Making Hackathon — AlphaBetaPhi

## Results

**2nd Place Overall** out of 93 teams — £205,726 profit (3x starting cash)

**1st Place Sortino Ratio** — best risk-adjusted returns in the competition

Final standings after 9 rounds:

| Rank | Team | Cash | Profit |
|------|------|------|--------|
| 1 | ConcurrentModificationExceptions | £306,879 | +£206,879 |
| **2** | **AlphaBetaPhi** | **£305,727** | **+£205,727** |
| 3 | Uncs Republic | £292,246 | +£192,246 |

Gap to 1st: £1,153

## Competition Format

The QMML Market Making Hackathon (25 Mar – 1 Apr 2026) was a 9-round live trading competition hosted by the Machine Learning Society at Queen Mary University of London. Each round:

1. Teams receive a dataset with features and a target (hidden stock price)
2. Teams train a model and submit a **bid** and **ask** price
3. The tightest spread becomes the **market maker** — everyone else trades against them
4. After seeing the MM's quotes, teams choose to **buy** or **sell** and set their **sizing**
5. The true price is revealed and P&L is calculated

Starting cash was £100,000. Going negative meant elimination.

## Our Strategy

Three principles drove our approach:

**1. Never be the market maker.** We submitted bid=80, ask=400 every round. The MM gets picked off by 90+ teams — it's the worst position. We avoided it completely.

**2. Bet big where we had an edge, minimum where we didn't.** We classified each stock into confidence tiers based on model R² and sample size, then sized accordingly:

| Stock | Model | R² | RMSE | Prediction | Sizing |
|-------|-------|-----|------|-----------|--------|
| 1 | Linear Regression | 0.98 | 4.87 | 273.88 | BIG |
| 2 | Linear Regression | 0.96 | 9.70 | 220.88 | BIG |
| 3 | Linear Regression | 0.93 | 23.83 | 268.80 | MODERATE |
| 4 | Gradient Boosting | 0.27 | 24.68 | 238.90 | MINIMUM |
| 5 | Ridge (α=100) | 0.20 | 28.09 | 249.68 | MINIMUM |
| 6 | Mean | 0.00 | 54.92 | 172.24 | MINIMUM |
| 7 | Ridge (α=100) | 0.04 | 15.55 | 214.70 | MINIMUM |
| 8 | Ridge (α=100) | 0.04 | 26.04 | 202.01 | MINIMUM |
| 9 | Mean | 0.00 | 43.74 | 218.60 | MINIMUM |

**3. Survive.** Risk management rules prevented any single bad trade from eliminating us. This discipline is what won us the Sortino ratio prize.

## Key Findings

- **Stocks 1–2** were essentially solved by linear regression. R² > 0.96 with only 3 informative features each. Extensive residual analysis confirmed the remaining error was irreducible Gaussian noise.
- **Stock 4** had mild nonlinearity that gradient boosting captured (R² improved from 0.14 to 0.27).
- **Stocks 6 and 9** had zero predictive signal — no model we tested beat simply predicting the mean. We threw everything at these: linear models, ridge, lasso, GBR, random forests, KNN, SVR, neural nets, PCA, polynomial features, mutual information analysis. Nothing worked.
- **Stock 7** had 20,000 rows and 25 features but R² of only 0.04. We exhaustively verified this wasn't a modelling failure — the features genuinely don't predict the target.
- **Stock 3** was the most interesting analytically. Only 29 rows but R² of 0.93. Cook's distance analysis revealed one influential outlier (row 26) whose removal would shift the prediction by 10 points. We chose to keep it — conservative was the right call with so little data.

## Repo Structure

```
├── data/                    # Train/test CSVs for all 9 stocks
├── round_1/                 # Round 1 EDA notebook
├── round_2/                 # Round 2 EDA notebook
├── round_3/                 # Round 3 EDA notebook
├── round_4/                 # Round 4 EDA notebook
├── round_5/                 # Round 5 EDA notebook
├── round_6/                 # Round 6 EDA notebook
├── round_7/                 # Round 7 EDA notebook
├── round_8/                 # Round 8 EDA notebook
├── round_9/                 # Round 9 EDA notebook
├── predictor.py             # Baseline linear regression (stock 1)
├── model_comparison.py      # Model comparison across stocks 3-9
└── strategy.py              # Full strategy: predictions, sizing, live cheat sheet
```

## How to Run

```bash
pip install pandas numpy scikit-learn

# Full strategy output with live decision cheat sheet
python strategy.py

# Model comparison across harder stocks
python model_comparison.py
```

## Team

- **Mohamed Rodani** — prediction models, strategy design, sizing logic, rounds 1/3/5/7/9
- **Kieran Cook** — EDA, feature engineering, rounds 2/4/6/8

## Tech Stack

Python, scikit-learn (LinearRegression, Ridge, GradientBoostingRegressor), pandas, numpy, matplotlib, seaborn
