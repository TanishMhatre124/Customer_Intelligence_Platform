# Business Problem

## Context

Olist is a Brazilian e-commerce marketplace that connects small and medium-sized merchants
across Brazil to major online sales channels. This project uses the public
[Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(orders, customers, products, payments, reviews, and sellers from 2016-2018) to answer a
question every marketplace and retail business eventually has to answer:

> **Which customers matter most, and which of them are at risk of leaving?**

## Business Objectives

1. **Understand customer value** — segment the customer base so that marketing, retention,
   and customer service resources are directed at the customers who generate the most value,
   not spread evenly across everyone.
2. **Identify churn risk** — flag customers who show signs of disengagement (long time since
   last purchase, few orders) so the business can intervene before that revenue is lost for good.
3. **Surface actionable insight, not just charts** — every analysis phase ends with a
   business interpretation, not just a plot.

## Approach

The project follows a standard analytics pipeline, implemented as a sequence of notebooks
(and mirrored in reusable `src/` modules):

| Phase | Notebook | Output |
|---|---|---|
| 1 | Data Understanding | Initial inventory of all 8 raw tables |
| 2 | Relationship Analysis | How the tables join together (order → customer → items → payments → reviews) |
| 3 | Data Cleaning | Typed, validated tables in `data/processed/` |
| 4 | Feature Engineering | One row per customer: orders, spend, recency, delivery experience |
| 5 | Exploratory Data Analysis | Distributions, trends, and relationships in the cleaned data |
| 6 | Customer Segmentation | RFM (Recency, Frequency, Monetary) scoring into 6 business segments |
| 7 | Churn Prediction | Classification models trained on RFM features to flag at-risk customers |
| 8 | Dashboard | Interactive summary combining every phase into decision-ready views |

## Defining Churn

Because this is a historical, order-level dataset (not a subscription service with a clear
cancellation event), churn is **operationally defined** rather than directly observed:

```
Churn = 1 if (Recency > 180 days) AND (Frequency <= 2 lifetime orders)
```

This is a reasonable working definition for a marketplace where most customers only order
occasionally, but it is a **rule**, not ground truth. Notebook 7 discusses the modeling
implication of this choice directly (data leakage risk) and what a production-grade
version of this model would need instead: a forward-looking outcome (e.g., "did this
customer purchase again in the next N days?") rather than a rule built from the same
features used to predict it.

## Success Criteria

- A clean, reproducible pipeline from raw CSVs to customer segments and churn scores.
- Customer segments that are business-interpretable (a marketer could act on "At Risk" or
  "Champions" without needing to understand RFM math).
- A ranked, exportable list of high-value customers at risk of churning, suitable for a
  retention campaign.
- Clear documentation of modeling limitations so results aren't over-trusted.
