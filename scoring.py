"""
scoring.py - Tính điểm tiêu chí TR/KN/A/SL
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_scores(df):
    """
    Tính điểm tiêu chí:
    - TR (Trình độ) = trung bình Q1..Q20
    - KN (Kỹ năng) = trung bình Q21..Q40
    - A (Thái độ) = trung bình Q41..Q60
    - SL (Định lượng) = trung bình toàn bộ Q1..Q60
    Chuẩn hóa về thang 0-1.
    """
    score_df = df.copy()

    tr_cols = [f"Q{i}" for i in range(1, 21)]
    kn_cols = [f"Q{i}" for i in range(21, 41)]
    a_cols = [f"Q{i}" for i in range(41, 61)]
    all_cols = [f"Q{i}" for i in range(1, 61)]

    score_df["TR"] = score_df[tr_cols].mean(axis=1)
    score_df["KN"] = score_df[kn_cols].mean(axis=1)
    score_df["A"] = score_df[a_cols].mean(axis=1)
    score_df["SL"] = score_df[all_cols].mean(axis=1)

    score_df["KN_ProblemSolving"] = score_df[[f"Q{i}" for i in range(21, 28)]].mean(axis=1)
    score_df["KN_Teamwork"] = score_df[[f"Q{i}" for i in range(28, 35)]].mean(axis=1)
    score_df["KN_Communication"] = score_df[[f"Q{i}" for i in range(35, 41)]].mean(axis=1)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    norm_cols = [
        "TR",
        "KN",
        "A",
        "SL",
        "KN_ProblemSolving",
        "KN_Teamwork",
        "KN_Communication",
    ]
    score_df[[f"{c}_norm" for c in norm_cols]] = scaler.fit_transform(score_df[norm_cols])

    return score_df
