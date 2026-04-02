# QMML Market Making Hackathon — AlphaBetaPhi

## Results

**2nd Place Overall** out of 93 teams — £205,727 profit (3x starting cash)

**1st Place Sortino Ratio** — best risk-adjusted returns in the competition

Final standings after 9 rounds:

| Rank | Team | Cash | Profit |
|------|------|------|--------|
| 1 | ConcurrentModificationExceptions | £306,879 | +£206,879 |
| **2** | **AlphaBetaPhi** | **£305,727** | **+£205,727** |
| 3 | Uncs Republic | £292,246 | +£192,246 |

Gap to 1st: £1,153 — a **0.38%** difference. One slightly different sizing decision on one round.

## Competition Format

The QMML Market Making Hackathon was a 9-round live trading competition hosted by the Machine Learning Society at Queen Mary University of London. Each round:

1. Teams receive a dataset with features and a target (hidden stock price)
2. Teams train a model and submit a **bid** and **ask** price
3. The tightest spread becomes the **market maker** — everyone else trades against them
4. After seeing the MM's quotes, teams choose to **buy** or **sell** and set their **sizing**
5. The true price is revealed and P&L is calculated

Starting cash was £100,000. Going negative meant elimination.

## Our Strategy

Three principles drove our approach:

**1. Never be the market maker.** We submitted bid=80, ask=400 every round. The MM gets picked off by 90+ teams — it's the worst position. We avoided it completely.

**2. Bet big where we had an edge, minimum where we didn't.** We classified each stock into confidence tiers based on model R² and sample size, then sized using Kelly criterion-based logic:

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
- **Stock 3** was the most interesting analytically. Only 29 rows but R² of 0.93. Cook's distance analysis revealed one influential outlier whose removal would shift the prediction by 10 points and drop LOO RMSE from 23.83 to 18.59. We chose to keep it — conservative was the right call with so little data.

## Repo Structure

```
├── data/                        # Train/test CSVs for all 9 stocks
├── round_1/notebook.ipynb       # Round 1 EDA — stock 1 deep dive
├── round_2/notebook.ipynb       # Round 2 EDA
├── round_3/round_3.ipynb        # Round 3 EDA — 29 rows, outlier analysis
├── round_4/round_4.ipynb        # Round 4 EDA
├── round_5/round_5.ipynb        # Round 5 EDA — 20 features, weak signal
├── round_6/round_6.ipynb        # Round 6 EDA
├── round_7/round7.ipynb         # Round 7 EDA — 20k rows, no signal
├── round_8/round_8.ipynb        # Round 8 EDA
├── round_9/round_9.ipynb        # Round 9 EDA — no model beats mean
├── strategy.py                  # Pre-competition strategy: predictions, tier-based sizing, live cheat sheet
├── kelly_based_strategy.py      # Live-round tool: Kelly criterion sizing with t-distribution confidence intervals
└── market_maker_prices.py       # MM quote generator: bid/ask at various spread widths, adjusted for skew and kurtosis
```

### Scripts

**`strategy.py`** — The pre-competition strategy engine. Runs all 9 models, outputs predictions, RMSE, sizing tiers, and a live decision cheat sheet for each round.

**`kelly_based_strategy.py`** — Interactive tool used during the live finals. Takes the MM's bid/ask as input, calculates the probability the true value is above/below the quotes using t-distributions (small samples) or normal distributions (large samples), then outputs Kelly criterion-based position sizes at conservative, aggressive, and super-aggressive levels. Adjusts for residual skewness and kurtosis.

**`market_maker_prices.py`** — Generates potential bid/ask quotes at different confidence levels (0.5σ to 2σ) for each stock. Includes both standard and skew/kurtosis-adjusted versions. Used for scenario planning before the live rounds.

## How to Run

```bash
pip install pandas numpy scikit-learn scipy

# Pre-competition strategy overview
python strategy.py

# Live-round Kelly sizing (interactive)
python kelly_based_strategy.py

# MM quote generation (interactive)
python market_maker_prices.py
```

## Team

- **Mohamed Rodani** — prediction models, strategy design, sizing logic, rounds 1/3/5/7/9
- **Kieran Cook** — EDA, feature engineering, Kelly criterion implementation, rounds 2/4/6/8

## Tech Stack

Python, scikit-learn, scipy, pandas, numpy, matplotlib, seaborn
