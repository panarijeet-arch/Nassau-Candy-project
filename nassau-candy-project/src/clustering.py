"""
clustering.py
==============
Route / customer / product clustering using K-Means, with automatic
selection of k via the Elbow Method + Silhouette Score.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROUTE_CLUSTER_FEATURES = [
    "Simulated_Lead_Time_Days", "Shipping_Distance_KM",
    "Shipping_Efficiency_Score", "Operational_Risk_Score", "Profit_Margin",
]


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> dict:
    """Run elbow method + silhouette score across a range of k, return both."""
    inertias = {}
    silhouettes = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = float(km.inertia_)
        silhouettes[k] = float(silhouette_score(X_scaled, labels))

    best_k = max(silhouettes, key=silhouettes.get)
    return {"inertias": inertias, "silhouettes": silhouettes, "best_k": best_k}


def cluster_routes(df: pd.DataFrame, k: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Cluster orders into route segments (e.g. fast/efficient vs slow/congested)
    based on lead time, distance, efficiency, risk, and margin.
    """
    df = df.copy()
    X = df[ROUTE_CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_search = find_optimal_k(X_scaled)
    chosen_k = k or k_search["best_k"]

    km = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    df["Route_Cluster"] = km.fit_predict(X_scaled)

    # Label clusters by their mean efficiency score (business-readable names)
    cluster_means = df.groupby("Route_Cluster")["Shipping_Efficiency_Score"].mean().sort_values()
    rank_to_label = {}
    n = len(cluster_means)
    labels_pool = ["Slow / Congested", "Below Average", "Average", "Above Average", "Fast / Efficient"]
    for i, cluster_id in enumerate(cluster_means.index):
        label_idx = int(i / max(n - 1, 1) * (len(labels_pool) - 1))
        rank_to_label[cluster_id] = labels_pool[label_idx]

    df["Route_Cluster_Label"] = df["Route_Cluster"].map(rank_to_label)

    meta = {
        "k_search": k_search,
        "chosen_k": chosen_k,
        "cluster_labels": rank_to_label,
    }
    logger.info("Route clustering complete with k=%s (silhouette-selected)", chosen_k)
    return df, meta
