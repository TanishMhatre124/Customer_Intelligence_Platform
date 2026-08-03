# Customer Intelligence & Churn Prediction Platform

An end-to-end customer analytics pipeline built on the
[Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):
data cleaning → feature engineering → RFM customer segmentation → churn prediction →
interactive dashboard.

See [`docs/business_problem.md`](docs/business_problem.md) for the full business context.

## Project Structure

```
Customer_Intelligence_Platform/
├── data/
│   ├── raw/                     # Original Olist CSVs (not modified)
│   └── processed/                # Cleaned tables, engineered features, RFM output
├── models/                       # Saved churn model (joblib)
├── reports/                      # model_comparison.csv, high_risk_customers.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_Relationship_Analysis.ipynb
│   ├── 03_Data_Cleaning.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Exploratory_Data_Analysis.ipynb
│   ├── 06_Customer_Segmentation.ipynb
│   ├── 07_Churn_Prediction.ipynb
│   └── 08_Dashboard.ipynb
├── src/
│   ├── config.py                 # Paths and constants
│   ├── data_loader.py            # Load raw CSVs
│   ├── data_cleaning.py          # Type fixes, missing value/duplicate checks
│   ├── feature_engineering.py    # Customer features + RFM segmentation
│   ├── model.py                  # Churn labeling, train/test split, model training
│   ├── evaluation.py             # Metrics, model comparison, feature importance
│   ├── eda.py                    # Reusable summary helpers for EDA/dashboard
│   └── utils.py                  # Small shared helpers
├── main.py                       # Runs the full pipeline end-to-end
└── requirements.txt
```

## Pipeline Overview

| Stage | What happens | Key output |
|---|---|---|
| **Clean** | Fix data types, check missing values/duplicates across all 8 raw tables | `data/processed/*_clean.csv` |
| **Feature Engineering** | Build one row per customer: total orders, spend, average order value, recency | `data/processed/customer_features.csv` |
| **Segmentation** | RFM scoring → 6 business segments (Champions, Loyal Customers, Potential Loyalists, Need Attention, At Risk, Lost Customers) | `data/processed/rfm_customer_segments.csv` |
| **Churn Prediction** | Train Logistic Regression, Decision Tree, and Random Forest on RFM features; pick the best by ROC-AUC | `models/churn_model_*.joblib`, `reports/model_comparison.csv` |
| **Retention Model (v2)** | Leakage-free model: predicts real 90-day repeat purchase from first-order behavior only | `data/processed/retention_model_scored.csv` |
| **Dashboard** | Interactive Plotly dashboard combining every phase | `notebooks/08_Dashboard.ipynb`, `streamlit_app.py` |

## Streamlit Dashboard

For a live, browsable web app (rather than a notebook), run:

```bash
streamlit run streamlit_app.py
```

This opens a multi-page dashboard in your browser (`localhost:8501`) with sidebar navigation
across: **Overview, Customer Segmentation, Churn Prediction, Geography, High-Risk Watchlist,
Retention Model (Advanced), Model Performance, and a Raw Data Explorer** — every metric and
chart from the notebooks in one interactive, dark-themed app, with segment/state filters,
downloadable CSVs, and caveat panels explaining each model's limitations honestly. It reads
directly from `data/processed/`, `reports/`, and `models/`, so run `python main.py` at least
once first (it now also trains the leakage-free retention model as its final step).

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the data

Download the [Olist dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
and place the CSVs in `data/raw/` (already included in this repo).

### 3. Run the full pipeline

```bash
python main.py
```

This runs cleaning → feature engineering → segmentation → churn modeling in one go, and
prints a summary of each stage (dataset shapes, segment counts, churn rate, model comparison,
feature importance, and how many customers were flagged as high-risk).

### 4. Explore interactively

Run the notebooks in order (`01` → `08`) if you want to see the analysis, decisions, and
reasoning behind each step rather than just the pipeline's final output. Notebook `08` is the
dashboard and is the best starting point if you just want the business-facing results.

## Customer Segments

Segments are assigned from RFM (Recency, Frequency, Monetary) quintile scores:

- **Champions** — most recent, most frequent buyers
- **Loyal Customers** — frequent, fairly recent buyers
- **Potential Loyalists** — recent buyers, not yet frequent
- **Need Attention** — mid-recency, at risk of drifting away
- **At Risk** — infrequent buyers who haven't purchased recently
- **Lost Customers** — long gone, low engagement historically

## Churn Model — Important Caveat

Churn is defined as `Recency > 180 days AND Frequency <= 2 orders`, then predicted using
`Recency`, `Frequency`, and `Monetary`. Because the label is built from two of the three
input features, the model scores near-perfectly — this reflects the model faithfully
learning the rule, **not** genuine predictive power on unseen future behavior. This is
explained in detail in `notebooks/07_Churn_Prediction.ipynb`, along with what a
leakage-free version of this model would need. The model is still useful here as an
automatic, explainable implementation of the churn rule, and its probability scores add
nuance beyond a hard cutoff — treat it as a segmentation/prioritization tool, not a
forecast.

## Improved Retention Model (leakage-free)

`notebooks/09_Improved_Retention_Model.ipynb` (and `src/retention_model.py`) build a
genuinely predictive alternative:

- **Label:** did the customer place a second order within **90 days** of their first one?
  A real, verifiable future outcome — not a rule built from the model's own inputs. Only
  customers whose first order had at least 90 days of runway before the dataset's cutoff
  are included, so every customer had a fair chance to reorder.
- **Features:** only what was knowable from the **first order** — delivery time, review
  score, payment method, price, product category, and seasonality. Nothing from later
  behavior.
- **Result:** a modest, honest ROC-AUC (~0.60) instead of a suspicious 1.00.

**Key findings from this model:**
- Only ~2.3% of eligible customers reorder within 90 days — repeat purchase is rare on
  this marketplace overall.
- **Product category matters far more than service quality.** Home/decor and sporting
  goods buyers reorder at 3-4x the rate of gadget/gift buyers.
- Delivery speed and first-order review score have **little effect** on repeat purchase —
  a useful negative result suggesting logistics/CS investment isn't the highest-leverage
  retention lever here; category-based cross-sell is.
- Use `Repeat_Probability` to rank and target first-time buyers, rather than treating it
  as a hard yes/no prediction (the outcome is too rare for that to be meaningful).
- Explore it interactively on the **Retention Model (Advanced)** page of the Streamlit app.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Plotly, Jupyter.

## Author

Tanish Mhatre
