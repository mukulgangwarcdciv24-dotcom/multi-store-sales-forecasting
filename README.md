# Multi-Store Sales Forecasting — 1C Software (Kaggle)

Predicting next-month item sales across 60 shops using time-series regression on the [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales) Kaggle competition dataset provided by 1C Company, one of Russia's largest software retailers.

---

## Overview

This project walks through the full data science pipeline — from raw, messy retail data to a competition-ready submission — with a focus on:

- **Heavy feature engineering** on multilingual (Russian) categorical text data
- **Lag feature construction** to convert an unsupervised time-series problem into a supervised regression task
- **Model comparison** between a fast linear baseline and a high-performance XGBoost regressor

The final XGBoost model achieved a **Kaggle holdout RMSE of 0.93763**, outperforming the linear regression baseline (RMSE: 1.06389) at the cost of significantly longer training time.

| Model | RMSE (Kaggle Holdout) | Fit Time |
|---|---|---|
| Linear Regression | 1.06389 | ~13 sec |
| XGBoost | **0.93763** | ~1508 sec |

---

## Dataset

Data is sourced from the Kaggle competition and consists of:

| File | Description |
|---|---|
| `sales_train.csv` | Daily sales records (Jan 2013 – Oct 2015) |
| `items.csv` | Item metadata and names |
| `item_categories.csv` | Item category names |
| `shops.csv` | Shop names |
| `test.csv` | Shop/item pairs to predict for the next month |

> Download the data from the [Kaggle competition page](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data) and place it in a `data/` directory.

---

## Project Structure

```
├── data/
│   ├── sales_train.csv
│   ├── items.csv
│   ├── item_categories.csv
│   ├── shops.csv
│   └── test.csv
├── supp/
│   └── tscv.png                  # Time series CV diagram
├── Time_Series_Regression_-_Multi-Store_Sales.ipynb
└── README.md
```

---

## Key Steps

### 1. Data Cleaning & Outlier Removal
- Removed extreme outliers in `item_cnt_day` (> 900) and `item_price` (> 300,000)
- Retained negative values (returns/refunds) as meaningful signal

### 2. Categorical Feature Engineering
All categorical text features are in Russian. The approach for each:

- **Shops**: Split shop names into two sub-features, corrected near-duplicates using fuzzy matching (`fuzzywuzzy`), grouped low-frequency values, and label-encoded
- **Item Categories**: Split on `-` delimiter, standardized inconsistent PC-related entries, and label-encoded
- **Items**: Extracted bracketed/parenthesized sub-strings from item names, identified product types (Xbox 360, PC, etc.), grouped rare types as `other`, and label-encoded

### 3. Matrix Construction
Built a complete Cartesian product matrix of every `(month, shop_id, item_id)` combination across all 34 training months, then merged monthly aggregated sales counts (`item_cnt_month`) back in. Month 34 (the test set) was appended to enable a single train/predict pipeline.

### 4. Lag Feature Engineering
Engineered sliding-window lag features to expose historical trends to the model:

- `item_cnt_month` lagged by 1, 2, 3 months
- Rolling mean item count by: date, date×item, date×shop, date×shop×item
- Rolling mean by: item category (split 1 & 2), full category, item type, item name token
- Average item price: overall and by date×item (lagged 1–3 months)

### 5. Time Series Cross-Validation
Used a custom **forward-chaining** CV strategy (rather than random k-fold) to respect temporal ordering. Starting from month 28, the model is validated on each successive month up to 33, giving 5 folds.

### 6. Models
- **Linear Regression** — fast baseline, trained and validated with time-series CV
- **Lasso Regression** (α=2.0) — tested for regularization; performed ~12% worse than plain LR, indicating underfitting rather than overfitting
- **XGBoost Regressor** — tuned with early stopping; substantially better RMSE at the cost of compute time

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost fuzzywuzzy matplotlib seaborn
```

> Python 3.8+ recommended. A GPU is optional but will significantly reduce XGBoost training time (`tree_method='gpu_hist'`).

---

## Usage

1. Clone the repo and install dependencies
2. Download and place the Kaggle dataset in `data/`
3. Open and run `Time_Series_Regression_-_Multi-Store_Sales.ipynb` end-to-end

To submit to Kaggle directly from the notebook:
```bash
kaggle competitions submit -c competitive-data-science-predict-future-sales -f <submission_file>.csv -m "<message>"
```

---

## Results & Learnings

- Time-series data requires **temporal cross-validation** — random k-fold leaks future information into training
- **Lag features** are the most impactful transformation for this type of problem; they give the model direct access to historical patterns without requiring architectural changes
- XGBoost significantly outperforms linear models here, suggesting non-linear interactions between shop, item, and temporal features that linear models cannot capture
- Further gains are possible through more granular price features and deeper domain-informed name parsing

---

## Potential Improvements

- Grid search over XGBoost hyperparameters (computationally expensive; sketch code included in notebook)
- Additional price-based features (price deltas, price relative to category mean)
- More sophisticated NLP on shop/item names for richer categorization
- Ensembling with LightGBM or CatBoost
