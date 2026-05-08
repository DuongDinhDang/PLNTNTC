"""
grouping.py - Chia nhóm học tập theo các chiến lược
"""

import math
import pandas as pd


def build_learning_groups(
    df,
    group_size=5,
    random_state=42,
    max_groups=10,
    strategy="balanced",
    target_skill_col="KN_ProblemSolving_norm",
    skill_cols=("KN_ProblemSolving_norm", "KN_Teamwork_norm", "KN_Communication_norm"),
):
    """
    Chia nhóm học tập theo các chế độ:
    - balanced: Trộn đều mức năng lực để các nhóm cân bằng.
    - homogeneous: Gom cùng trình độ vào nhóm gần nhau.
    - intervention: Tách nhóm ALERT để giáo viên can thiệp sớm.
    """
    data = df.copy().sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    total_students = len(data)
    n_groups = max(1, math.ceil(total_students / group_size))
    if max_groups is not None:
        n_groups = min(n_groups, max(1, int(max_groups)))

    effective_group_size = max(group_size, math.ceil(total_students / n_groups))
    groups = [{"name": f"Nhóm {i+1}", "members": []} for i in range(n_groups)]

    def available_group_indices():
        return [i for i in range(n_groups) if len(groups[i]["members"]) < effective_group_size]

    if strategy == "homogeneous":
        level_order = ["Xuất sắc", "Giỏi", "Khá", "Trung bình", "Yếu (At Risk)"]
        for label in level_order:
            pool = data[data["Nhãn năng lực"] == label].copy()
            pool = pool.sample(frac=1.0, random_state=random_state).to_dict("records")

            while pool:
                candidate_idx = available_group_indices()
                if not candidate_idx:
                    break

                best_idx = min(
                    candidate_idx,
                    key=lambda i: (
                        -sum(1 for m in groups[i]["members"] if m["Nhãn năng lực"] == label),
                        len(groups[i]["members"]),
                    ),
                )
                groups[best_idx]["members"].append(pool.pop())

    elif strategy == "intervention":
        alert_df = data[data["AIS_Cảnh báo"] == "ALERT"].copy().reset_index(drop=True)
        ok_df = data[data["AIS_Cảnh báo"] != "ALERT"].copy().reset_index(drop=True)

        rows = []

        if len(alert_df) > 0:
            n_alert_groups = max(1, math.ceil(len(alert_df) / effective_group_size))
            for g in range(n_alert_groups):
                start = g * effective_group_size
                end = start + effective_group_size
                chunk = alert_df.iloc[start:end]
                for _, row in chunk.iterrows():
                    rec = row.to_dict()
                    rec["Nhóm học tập"] = (
                        "Nhóm Can thiệp AIS" if n_alert_groups == 1 else f"Nhóm Can thiệp AIS {g+1}"
                    )
                    rows.append(rec)

        if len(ok_df) > 0:
            ok_total = len(ok_df)
            ok_n_groups = max(1, math.ceil(ok_total / effective_group_size))
            ok_groups = [{"name": f"Nhóm {i+1}", "members": []} for i in range(ok_n_groups)]

            level_order = ["Xuất sắc", "Giỏi", "Khá", "Trung bình", "Yếu (At Risk)"]
            buckets = {}
            for label in level_order:
                temp = ok_df[ok_df["Nhãn năng lực"] == label].copy()
                temp = temp.sample(frac=1.0, random_state=random_state).to_dict("records")
                buckets[label] = temp

            for label in level_order:
                while buckets[label]:
                    candidate_idx = [
                        i
                        for i in range(ok_n_groups)
                        if len(ok_groups[i]["members"]) < effective_group_size
                    ]
                    if not candidate_idx:
                        break

                    best_idx = min(
                        candidate_idx,
                        key=lambda i: (
                            sum(1 for m in ok_groups[i]["members"] if m["Nhãn năng lực"] == label),
                            len(ok_groups[i]["members"]),
                        ),
                    )
                    ok_groups[best_idx]["members"].append(buckets[label].pop())

            for g in ok_groups:
                for m in g["members"]:
                    rec = dict(m)
                    rec["Nhóm học tập"] = g["name"]
                    rows.append(rec)

        if len(rows) > 0:
            return pd.DataFrame(rows)

    elif strategy == "skill_specialized":
        if target_skill_col not in data.columns:
            target_skill_col = "KN_ProblemSolving_norm"

        data_sorted = data.sort_values(target_skill_col, ascending=False).reset_index(drop=True)
        for idx, (_, row) in enumerate(data_sorted.iterrows()):
            group_idx = idx % n_groups
            groups[group_idx]["members"].append(row.to_dict())
    else:
        level_order = ["Xuất sắc", "Giỏi", "Khá", "Trung bình", "Yếu (At Risk)"]
        buckets = {}
        for label in level_order:
            temp = data[data["Nhãn năng lực"] == label].copy()
            temp = temp.sample(frac=1.0, random_state=random_state).to_dict("records")
            buckets[label] = temp

        for label in level_order:
            while buckets[label]:
                candidate_idx = [
                    i for i in range(n_groups) if len(groups[i]["members"]) < group_size
                ]
                if not candidate_idx:
                    break

                best_idx = min(
                    candidate_idx,
                    key=lambda i: (
                        sum(1 for m in groups[i]["members"] if m["Nhãn năng lực"] == label),
                        len(groups[i]["members"]),
                    ),
                )
                groups[best_idx]["members"].append(buckets[label].pop())

    rows = []
    for g in groups:
        for m in g["members"]:
            rec = dict(m)
            rec["Nhóm học tập"] = g["name"]
            rows.append(rec)

    group_df = pd.DataFrame(rows)

    if len(group_df) > 0 and max_groups is not None:
        hard_limit = max(1, int(max_groups))
        group_order = list(dict.fromkeys(group_df["Nhóm học tập"].astype(str).tolist()))
        if len(group_order) > hard_limit:
            keep_groups = group_order[:hard_limit]
            overflow_groups = set(group_order[hard_limit:])
            group_counts = group_df["Nhóm học tập"].value_counts().to_dict()

            overflow_idx = group_df[group_df["Nhóm học tập"].isin(overflow_groups)].index.tolist()
            for idx in overflow_idx:
                target_group = min(keep_groups, key=lambda g: group_counts.get(g, 0))
                group_df.at[idx, "Nhóm học tập"] = target_group
                group_counts[target_group] = group_counts.get(target_group, 0) + 1

    assigned_ids = set(group_df.get("MSSV", pd.Series(dtype=str)).astype(str).tolist())
    remaining = data[~data["MSSV"].astype(str).isin(assigned_ids)]
    if len(remaining) > 0 and len(group_df) > 0:
        for _, row in remaining.iterrows():
            sizes = group_df.groupby("Nhóm học tập").size().to_dict()
            target_group = min(sizes, key=sizes.get)
            rec = row.to_dict()
            rec["Nhóm học tập"] = target_group
            group_df = pd.concat([group_df, pd.DataFrame([rec])], ignore_index=True)

    return group_df
