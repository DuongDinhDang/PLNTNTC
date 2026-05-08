"""
alerts.py - Hệ thống cảnh báo AIS (Artificial Immune System)
"""

import numpy as np


def run_ais_alert(df):
    """Negative Selection: cảnh báo khi A hoặc KN dưới ngưỡng an toàn 0.25."""
    out_df = df.copy()
    out_df["AIS_Cảnh báo"] = np.where(
        (out_df["A_norm"] < 0.25) | (out_df["KN_norm"] < 0.25),
        "ALERT",
        "OK",
    )
    return out_df
