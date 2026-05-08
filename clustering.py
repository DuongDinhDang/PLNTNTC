"""
clustering.py - K-Means clustering và gán nhãn năng lực
"""

import numpy as np
from sklearn.cluster import KMeans
from config import ABILITY_LABELS


def weighted_centroid_score(centroid, feature_cols):
    """
    Tính điểm trọng số ASK để xếp thứ tự cụm:
    A*0.4 + KN*0.4 + TR*0.2
    """
    weight_map = {
        "A_norm": 0.4,
        "KN_norm": 0.4,
        "TR_norm": 0.2,
    }

    score = 0.0
    total_weight = 0.0
    for i, col in enumerate(feature_cols):
        if col in weight_map:
            score += centroid[i] * weight_map[col]
            total_weight += weight_map[col]

    if total_weight > 0:
        return score / total_weight

    return float(np.mean(centroid))


def run_kmeans_and_label(score_df, selected_feature_cols):
    """Chạy K-Means, sau đó gán nhãn cụm đúng thứ tự năng lực."""
    data_x = score_df[selected_feature_cols].to_numpy()
    n_students = len(score_df)
    n_clusters = min(5, n_students)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_ids = kmeans.fit_predict(data_x)

    centroid_scores = []
    for cid, center in enumerate(kmeans.cluster_centers_):
        score = weighted_centroid_score(center, selected_feature_cols)
        centroid_scores.append((cid, score))
    centroid_scores.sort(key=lambda x: x[1])

    active_labels = ABILITY_LABELS[:n_clusters]
    cluster_to_label = {
        cid: active_labels[i] for i, (cid, _) in enumerate(centroid_scores)
    }

    labeled_df = score_df.copy()
    labeled_df["ClusterID"] = cluster_ids
    labeled_df["Nhãn năng lực"] = labeled_df["ClusterID"].map(cluster_to_label)

    return labeled_df, kmeans
