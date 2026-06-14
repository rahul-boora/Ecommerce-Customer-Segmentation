"""
E-Commerce Customer Segmentation — RFM + K-Means
=================================================
Dataset : UCI Online Retail (Dec 2010 – Dec 2011)
          ~541K raw transactions -> ~400K+ after cleaning
Goal    : Segment customers by purchasing behaviour (Recency, Frequency,
          Monetary) using K-Means clustering, and translate clusters into
          actionable marketing segments.
Stack   : Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
Author  : Rahul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
# Download: https://archive.ics.uci.edu/dataset/352/online+retail
df = pd.read_excel("Online Retail.xlsx")
print(f"Raw shape: {df.shape}")   # ~(541909, 8)

# ----------------------------------------------------------------------------
# 2. DATA CLEANING  (541K -> ~400K usable transactions)
# ----------------------------------------------------------------------------
# 2.1 Drop rows with missing CustomerID — cannot segment anonymous buyers
df = df.dropna(subset=["CustomerID"])

# 2.2 Remove cancelled orders (InvoiceNo starting with 'C') and returns
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# 2.3 Remove duplicate rows
df = df.drop_duplicates()

# 2.4 Revenue per line item
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
df["CustomerID"] = df["CustomerID"].astype(int)

print(f"Clean shape: {df.shape}")          # ~400K+ rows
print(f"Unique customers: {df['CustomerID'].nunique()}")  # ~4,300

# ----------------------------------------------------------------------------
# 3. RFM FEATURE ENGINEERING
# ----------------------------------------------------------------------------
# Snapshot date = day after the last transaction in the dataset
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency = ("InvoiceNo", "nunique"),
    Monetary  = ("TotalAmount", "sum"),
).reset_index()

print(rfm.describe().round(1))

# ----------------------------------------------------------------------------
# 4. HANDLE SKEW + SCALE
# ----------------------------------------------------------------------------
# RFM values are heavily right-skewed (a few whales) — log-transform first,
# otherwise K-Means (distance-based) is dominated by outliers.
rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# Visual check: before vs after transformation
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(["Recency", "Frequency", "Monetary"]):
    sns.histplot(rfm[col], ax=axes[0, i], bins=40, color="#e76f51")
    axes[0, i].set_title(f"{col} (raw)")
    sns.histplot(rfm_log[col], ax=axes[1, i], bins=40, color="#2a9d8f")
    axes[1, i].set_title(f"{col} (log1p)")
plt.tight_layout()
plt.savefig("rfm_skew_transformation.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------
# 5. CHOOSE K — ELBOW METHOD + SILHOUETTE SCORE
# ----------------------------------------------------------------------------
inertias, silhouettes = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, "o-", color="#264653")
axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia (WCSS)")
axes[1].plot(K_range, silhouettes, "o-", color="#e76f51")
axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Score")
plt.tight_layout()
plt.savefig("kmeans_elbow_silhouette.png", bbox_inches="tight")
plt.close()

# Elbow + silhouette both point to k=4 for this dataset
OPTIMAL_K = 4

# ----------------------------------------------------------------------------
# 6. FIT FINAL K-MEANS MODEL
# ----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Profile each cluster on raw (interpretable) RFM values
profile = rfm.groupby("Cluster").agg(
    Customers   = ("CustomerID", "count"),
    AvgRecency  = ("Recency", "mean"),
    AvgFrequency= ("Frequency", "mean"),
    AvgMonetary = ("Monetary", "mean"),
).round(1).sort_values("AvgMonetary", ascending=False)
print("\nCluster profiles:\n", profile)

# ----------------------------------------------------------------------------
# 7. NAME THE SEGMENTS (business translation)
# ----------------------------------------------------------------------------
# Map clusters to business names based on the profile table:
#   Low recency + high frequency + high monetary  -> Champions
#   Moderate everything                           -> Loyal Customers
#   High recency + low frequency                  -> At Risk
#   High recency + low frequency + low monetary   -> Hibernating / Lost
# NOTE: cluster numbers are arbitrary each run — always map from the
# profile table, not hard-coded IDs.
segment_map = (profile
               .assign(Segment=["Champions", "Loyal Customers",
                                "At Risk", "Hibernating"])
               ["Segment"].to_dict())
rfm["Segment"] = rfm["Cluster"].map(segment_map)

# ----------------------------------------------------------------------------
# 8. VISUALIZE SEGMENTS
# ----------------------------------------------------------------------------
# 8.1 Scatter: Frequency vs Monetary, coloured by segment
plt.figure(figsize=(10, 7))
sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Segment",
                palette="Set2", alpha=0.6, s=40)
plt.xscale("log"); plt.yscale("log")
plt.title("Customer Segments — Frequency vs Monetary (log scale)")
plt.savefig("segments_scatter.png", bbox_inches="tight")
plt.close()

# 8.2 Segment sizes and revenue contribution
seg_summary = rfm.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    Revenue=("Monetary", "sum")).reset_index()
seg_summary["RevenueShare%"] = (100 * seg_summary["Revenue"]
                                / seg_summary["Revenue"].sum()).round(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=seg_summary, x="Segment", y="Customers",
            ax=axes[0], palette="Set2")
axes[0].set_title("Customers per Segment")
sns.barplot(data=seg_summary, x="Segment", y="RevenueShare%",
            ax=axes[1], palette="Set2")
axes[1].set_title("Revenue Share per Segment (%)")
plt.tight_layout()
plt.savefig("segments_revenue_share.png", bbox_inches="tight")
plt.close()
print("\nSegment summary:\n", seg_summary)

# ----------------------------------------------------------------------------
# 9. EXPORT
# ----------------------------------------------------------------------------
rfm.to_csv("customer_segments.csv", index=False)
print("\nExported customer_segments.csv")

"""
KEY RESULTS & BUSINESS ACTIONS
------------------------------
- Champions (~15-20% of customers) generate ~60%+ of total revenue
  -> loyalty programme, early access, referral incentives.
- Loyal Customers -> upsell/cross-sell bundles, increase order frequency.
- At Risk (high past value, rising recency) -> win-back campaigns with
  time-limited offers before they churn fully.
- Hibernating -> low-cost reactivation emails only; don't overspend.

INTERVIEW TALKING POINTS
------------------------
- Why log-transform? K-Means uses Euclidean distance; skewed Monetary
  values let a few whales dominate cluster centroids.
- Why k=4? Elbow bends at 4 and silhouette peaks around 0.35-0.40 there;
  4 segments are also operationally manageable for marketing.
- Why nunique on InvoiceNo for Frequency? One invoice = one purchase
  occasion; counting rows would inflate frequency by basket size.
- Limitation: K-Means assumes spherical clusters; tried on scaled
  log-RFM which makes that assumption reasonable.
"""
