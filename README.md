# E-Commerce Customer Segmentation (RFM + K-Means)

Segmented 4,300+ customers from 400K+ retail transactions into actionable marketing segments using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering.

## Problem Statement

Treating all customers identically wastes marketing budget. This project groups customers by purchasing behaviour so the business can target each segment differently — retain high-value customers, win back lapsing ones, and avoid overspending on dormant ones.

## Dataset

UCI Online Retail — ~541K transactions from a UK-based online retailer (Dec 2010 to Dec 2011). After cleaning (removing missing CustomerIDs, cancellations, returns, and duplicates), 400K+ transactions across ~4,300 customers remain.

Download: https://archive.ics.uci.edu/dataset/352/online+retail

## Tech Stack

Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn).

## Approach

1. **Cleaning** — dropped ~135K rows with missing CustomerID, removed cancelled invoices (prefix "C"), negative quantities, zero prices, and duplicates.
2. **RFM engineering** — per customer: Recency (days since last purchase from a snapshot date), Frequency (unique invoices), Monetary (total spend).
3. **Skew handling** — log1p transform on all three RFM features, then StandardScaler, since K-Means is distance-based and raw Monetary is dominated by a few very large customers.
4. **Choosing k** — elbow method (WCSS) plus silhouette scores across k = 2–10; both supported k = 4.
5. **Clustering & naming** — fit K-Means (k=4), profiled clusters on raw RFM averages, and mapped them to business segments: Champions, Loyal Customers, At Risk, Hibernating.
6. **Output** — segment scatter plots, revenue-share charts, and `customer_segments.csv` for downstream marketing use.

## Key Findings

- Champions (a small fraction of customers) contribute a disproportionate majority of revenue — the classic Pareto pattern.
- A clear At Risk segment exists: historically valuable customers with rising recency, ideal for win-back campaigns.
- Hibernating customers are numerous but low-value; reactivation spend should be minimal.

## Recommended Actions per Segment

- **Champions** — loyalty programme, early product access, referral incentives.
- **Loyal Customers** — cross-sell bundles and frequency-boosting offers.
- **At Risk** — time-limited win-back discounts before full churn.
- **Hibernating** — low-cost email reactivation only.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
# place "Online Retail.xlsx" in this folder
python customer_segmentation_rfm_kmeans.py
```

Outputs: skew-transformation charts, elbow/silhouette plots, segment visualizations, and `customer_segments.csv`.
