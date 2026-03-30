import math

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# Module 0: Cấu hình chung
# ==========================================================
st.set_page_config(page_title="ASK - Phân loại sinh viên thông minh", layout="wide")

ABILITY_LABELS = ["Yếu (At Risk)", "Trung bình", "Khá", "Giỏi", "Xuất sắc"]
LABEL_ORDER = {label: idx for idx, label in enumerate(ABILITY_LABELS)}


# ==========================================================
# Module 1 + Module 2: Nhập liệu và Tiền xử lý
# ==========================================================
def create_dummy_data(n_students=50, random_state=42):
    """Tạo dữ liệu giả lập 50 sinh viên, 60 câu hỏi (thang 1-5)."""
    rng = np.random.default_rng(random_state)

    student_ids = [f"SV{str(i).zfill(3)}" for i in range(1, n_students + 1)]
    full_names = [f"Sinh viên {i}" for i in range(1, n_students + 1)]

    # Tạo 5 mức năng lực cơ bản để dữ liệu có cấu trúc cụm rõ hơn
    base_levels = np.array([1.8, 2.4, 3.0, 3.7, 4.4])
    student_level = rng.integers(0, 5, size=n_students)

    data = {
        "StudentID": student_ids,
        "Họ tên": full_names,
    }

    for q in range(1, 61):
        # Điều chỉnh nhẹ theo nhóm câu hỏi để tạo khác biệt TR/KN/A
        if q <= 20:
            shift = 0.00  # TR
        elif q <= 40:
            shift = 0.15  # KN
        else:
            shift = -0.05  # A

        values = base_levels[student_level] + shift + rng.normal(0, 0.45, size=n_students)
        values = np.clip(np.rint(values), 1, 5).astype(int)
        data[f"Q{q}"] = values

    return pd.DataFrame(data)


def read_uploaded_file(uploaded_file):
    """Đọc 1 file CSV/Excel từ uploader."""
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError(f"Định dạng file không hỗ trợ: {uploaded_file.name}")


def read_and_merge_uploaded_files(uploaded_files):
    """Đọc nhiều file CSV/Excel và gộp thành một DataFrame."""
    if not uploaded_files:
        return None

    frames = []
    for file_obj in uploaded_files:
        temp_df = read_uploaded_file(file_obj)
        temp_df = temp_df.copy()
        temp_df["__SourceFile"] = file_obj.name
        frames.append(temp_df)

    return pd.concat(frames, ignore_index=True, sort=False)


def resolve_duplicate_student_ids(df, policy="keep_last"):
    """
    Xử lý trùng StudentID sau khi gộp nhiều file.
    policy:
    - keep_first: giữ bản ghi xuất hiện trước
    - keep_last: giữ bản ghi xuất hiện sau
    - make_unique: giữ tất cả, thêm hậu tố _2, _3...
    """
    out_df = df.copy()
    if "StudentID" not in out_df.columns:
        return out_df, 0

    sid_series = out_df["StudentID"].astype(str).str.strip()
    sid_series = sid_series.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    out_df["StudentID"] = sid_series

    dup_count = int(out_df["StudentID"].duplicated(keep=False).sum())
    if dup_count == 0:
        return out_df, 0

    if policy == "keep_first":
        out_df = out_df.drop_duplicates(subset=["StudentID"], keep="first")
        return out_df.reset_index(drop=True), dup_count

    if policy == "keep_last":
        out_df = out_df.drop_duplicates(subset=["StudentID"], keep="last")
        return out_df.reset_index(drop=True), dup_count

    # make_unique
    seen = {}
    new_ids = []
    for sid in out_df["StudentID"].tolist():
        if pd.isna(sid):
            new_ids.append(sid)
            continue

        if sid not in seen:
            seen[sid] = 1
            new_ids.append(sid)
        else:
            seen[sid] += 1
            new_ids.append(f"{sid}_{seen[sid]}")

    out_df["StudentID"] = new_ids
    return out_df, dup_count


def ensure_question_columns(df):
    """Đảm bảo dữ liệu có đủ cột Q1..Q60; nếu thiếu thì báo lỗi rõ ràng."""
    expected = [f"Q{i}" for i in range(1, 61)]
    missing = [c for c in expected if c not in df.columns]

    if missing:
        raise ValueError(
            "Thiếu các cột câu hỏi: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    return expected


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

    # Tách chi tiết kỹ năng để hỗ trợ chế độ chia nhóm chuyên biệt theo kỹ năng
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


# ==========================================================
# Module 3: Thuật toán (K-Means, KNN, AIS)
# ==========================================================
def weighted_centroid_score(centroid, feature_cols):
    """
    Tính điểm trọng số ASK để xếp thứ tự cụm:
    A*0.4 + S*0.4 + K*0.2
    Mapping: A -> A_norm, S -> KN_norm, K -> TR_norm
    Nếu thiếu một thành phần do giáo viên không chọn, tự chuẩn hóa lại trọng số phần còn lại.
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

    # Trường hợp chỉ chọn SL hoặc tiêu chí khác ASK: dùng trung bình centroid
    return float(np.mean(centroid))


def run_kmeans_and_label(score_df, selected_feature_cols):
    """Chạy K-Means, sau đó gán nhãn cụm đúng thứ tự năng lực."""
    data_x = score_df[selected_feature_cols].to_numpy()
    n_students = len(score_df)
    n_clusters = min(5, n_students)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_ids = kmeans.fit_predict(data_x)

    # Sắp xếp centroid từ thấp -> cao theo điểm trọng số ASK
    centroid_scores = []
    for cid, center in enumerate(kmeans.cluster_centers_):
        score = weighted_centroid_score(center, selected_feature_cols)
        centroid_scores.append((cid, score))
    centroid_scores.sort(key=lambda x: x[1])

    # Với n_clusters < 5 thì lấy n nhãn đầu tiên
    active_labels = ABILITY_LABELS[:n_clusters]
    cluster_to_label = {
        cid: active_labels[i] for i, (cid, _) in enumerate(centroid_scores)
    }

    labeled_df = score_df.copy()
    labeled_df["ClusterID"] = cluster_ids
    labeled_df["Nhãn năng lực"] = labeled_df["ClusterID"].map(cluster_to_label)

    return labeled_df, kmeans


def train_knn_classifier(df, selected_feature_cols):
    """Huấn luyện KNN để sẵn sàng phân loại sinh viên mới."""
    x_train = df[selected_feature_cols].to_numpy()
    y_train = df["Nhãn năng lực"].to_numpy()

    n_neighbors = min(5, len(df))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    return knn


def run_ais_alert(df):
    """Negative Selection: cảnh báo khi A hoặc KN dưới ngưỡng an toàn 0.25."""
    out_df = df.copy()
    out_df["AIS_Cảnh báo"] = np.where(
        (out_df["A_norm"] < 0.25) | (out_df["KN_norm"] < 0.25),
        "ALERT",
        "OK",
    )
    return out_df


# ==========================================================
# Module 4: Đầu ra - thống kê SL, chia nhóm DS, biểu đồ
# ==========================================================
def build_learning_groups(
    df,
    group_size=5,
    random_state=42,
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
    groups = [{"name": f"Nhóm {i+1}", "members": []} for i in range(n_groups)]

    def available_group_indices():
        return [i for i in range(n_groups) if len(groups[i]["members"]) < group_size]

    if strategy == "homogeneous":
        # Gom nhóm theo cùng trình độ: cố gắng để độ đa dạng nhãn trong nhóm thấp.
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
        # Tách riêng sinh viên ALERT thành nhóm can thiệp, phần còn lại chia cân bằng.
        alert_df = data[data["AIS_Cảnh báo"] == "ALERT"].copy().reset_index(drop=True)
        ok_df = data[data["AIS_Cảnh báo"] != "ALERT"].copy().reset_index(drop=True)

        rows = []

        # 1) Tạo các nhóm can thiệp cho ALERT
        if len(alert_df) > 0:
            n_alert_groups = max(1, math.ceil(len(alert_df) / group_size))
            for g in range(n_alert_groups):
                start = g * group_size
                end = start + group_size
                chunk = alert_df.iloc[start:end]
                for _, row in chunk.iterrows():
                    rec = row.to_dict()
                    rec["Nhóm học tập"] = (
                        "Nhóm Can thiệp AIS" if n_alert_groups == 1 else f"Nhóm Can thiệp AIS {g+1}"
                    )
                    rows.append(rec)

        # 2) Chia phần OK theo chiến lược cân bằng
        if len(ok_df) > 0:
            ok_total = len(ok_df)
            ok_n_groups = max(1, math.ceil(ok_total / group_size))
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
                        i for i in range(ok_n_groups) if len(ok_groups[i]["members"]) < group_size
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

        # Ưu tiên sinh viên có điểm cao ở kỹ năng mục tiêu vào cùng chiến lược nhóm
        data_sorted = data.sort_values(target_skill_col, ascending=False).reset_index(drop=True)
        for idx, (_, row) in enumerate(data_sorted.iterrows()):
            group_idx = idx % n_groups
            groups[group_idx]["members"].append(row.to_dict())
    else:
        # Chế độ cân bằng: phân phối đều từng mức năng lực vào các nhóm.
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

                # Ưu tiên nhóm đang thiếu nhãn này và thiếu người hơn
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

    # Trường hợp rất hiếm: còn sót do dữ liệu lỗi, thêm vào nhóm đang ít người nhất
    assigned_ids = set(group_df.get("StudentID", pd.Series(dtype=str)).astype(str).tolist())
    remaining = data[~data["StudentID"].astype(str).isin(assigned_ids)]
    if len(remaining) > 0 and len(group_df) > 0:
        for _, row in remaining.iterrows():
            sizes = group_df.groupby("Nhóm học tập").size().to_dict()
            target_group = min(sizes, key=sizes.get)
            rec = row.to_dict()
            rec["Nhóm học tập"] = target_group
            group_df = pd.concat([group_df, pd.DataFrame([rec])], ignore_index=True)

    return group_df


# ==========================================================
# Giao diện chính
# ==========================================================
st.title("WEB APP PHÂN LOẠI SINH VIÊN THÔNG MINH (ASK)")
st.caption("Ứng dụng hỗ trợ giáo viên đánh giá và phân nhóm sinh viên bằng K-Means, KNN và AIS.")

st.subheader("Thiết lập phân tích")

top_left, top_right = st.columns([2, 1])
with top_left:
    uploaded_files = st.file_uploader(
        "Tải lên một hoặc nhiều file khảo sát 60 câu (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
with top_right:
    run_btn = st.button("Chạy phân tích", type="primary", use_container_width=True)

dedupe_mode_label = st.selectbox(
    "Nếu trùng mã sinh viên khi gộp nhiều file, bạn muốn xử lý thế nào?",
    options=[
        "Giữ dữ liệu ở file tải lên sau cùng",
        "Giữ dữ liệu ở file tải lên đầu tiên",
        "Giữ tất cả dữ liệu và tự đánh số mã sinh viên",
    ],
    index=0,
)

dedupe_policy_map = {
    "Giữ dữ liệu ở file tải lên sau cùng": "keep_last",
    "Giữ dữ liệu ở file tải lên đầu tiên": "keep_first",
    "Giữ tất cả dữ liệu và tự đánh số mã sinh viên": "make_unique",
}
dedupe_policy = dedupe_policy_map[dedupe_mode_label]

st.markdown("### BẢNG CHỌN TIÊU CHÍ PHÂN LOẠI (CRITERIA SELECTION)")
criteria_table = pd.DataFrame(
    [
        {
            "Nhóm tiêu chí": "TR - Trình độ (Knowledge)",
            "Các tiêu chí cụ thể": "Khả năng hiểu bản chất bài học; Tư duy logic & Thuật toán; Khả năng hệ thống hóa kiến thức",
            "Ý nghĩa đối với thuật toán": "Dùng K-Means để chia nhóm theo năng lực nhận thức.",
        },
        {
            "Nhóm tiêu chí": "KN - Kỹ năng (Skills)",
            "Các tiêu chí cụ thể": "Giải quyết vấn đề (Problem Solving); Làm việc nhóm (Teamwork); Thuyết trình & Giao tiếp",
            "Ý nghĩa đối với thuật toán": "Dùng để xác định vai trò của SV trong nhóm (Leader, Member, Tech).",
        },
        {
            "Nhóm tiêu chí": "A - Thái độ (Attitude)",
            "Các tiêu chí cụ thể": "Tính chủ động (Proactivity); Sự kiên trì (Persistence); Kỷ luật & Chuyên cần",
            "Ý nghĩa đối với thuật toán": "Đầu vào quan trọng cho AIS để phát hiện sinh viên có nguy cơ bỏ học.",
        },
        {
            "Nhóm tiêu chí": "SL - Định lượng (Metrics)",
            "Các tiêu chí cụ thể": "Tốc độ tiếp thu (Learning Speed); Số lượng bài tập đã hoàn thành; Tỉ lệ lỗi (Error Rate)",
            "Ý nghĩa đối với thuật toán": "Cung cấp dữ liệu thực tế (Hard data) thay vì chỉ là khảo sát cảm tính.",
        },
    ]
)
st.dataframe(criteria_table, use_container_width=True, hide_index=True, height=260)

st.markdown("#### Chọn tiêu chí cụ thể để phân tích")
left_col, right_col = st.columns(2)

with left_col:
    with st.expander("TR - Trình độ (Knowledge)", expanded=True):
        tr_1 = st.checkbox("Khả năng hiểu bản chất bài học", value=True, key="tr_1")
        tr_2 = st.checkbox("Tư duy logic & Thuật toán", value=True, key="tr_2")
        tr_3 = st.checkbox("Khả năng hệ thống hóa kiến thức", value=True, key="tr_3")

    with st.expander("A - Thái độ (Attitude)", expanded=True):
        a_1 = st.checkbox("Tính chủ động (Proactivity)", value=True, key="a_1")
        a_2 = st.checkbox("Sự kiên trì (Persistence)", value=True, key="a_2")
        a_3 = st.checkbox("Kỷ luật & Chuyên cần", value=True, key="a_3")

with right_col:
    with st.expander("KN - Kỹ năng (Skills)", expanded=True):
        kn_1 = st.checkbox("Giải quyết vấn đề (Problem Solving)", value=True, key="kn_1")
        kn_2 = st.checkbox("Làm việc nhóm (Teamwork)", value=True, key="kn_2")
        kn_3 = st.checkbox("Thuyết trình & Giao tiếp", value=True, key="kn_3")

    with st.expander("SL - Định lượng (Metrics)", expanded=False):
        sl_1 = st.checkbox("Tốc độ tiếp thu (Learning Speed)", value=False, key="sl_1")
        sl_2 = st.checkbox("Số lượng bài tập đã hoàn thành", value=False, key="sl_2")
        sl_3 = st.checkbox("Tỉ lệ lỗi (Error Rate)", value=False, key="sl_3")

use_tr = any([tr_1, tr_2, tr_3])
use_kn = any([kn_1, kn_2, kn_3])
use_a = any([a_1, a_2, a_3])
use_sl = any([sl_1, sl_2, sl_3])

criterion_map = {
    "TR": (use_tr, "TR_norm"),
    "KN": (use_kn, "KN_norm"),
    "A": (use_a, "A_norm"),
    "SL": (use_sl, "SL_norm"),
}

selected_feature_cols = [col for _, (flag, col) in criterion_map.items() if flag]

selected_criteria_labels = []
if tr_1:
    selected_criteria_labels.append("TR: Khả năng hiểu bản chất bài học")
if tr_2:
    selected_criteria_labels.append("TR: Tư duy logic & Thuật toán")
if tr_3:
    selected_criteria_labels.append("TR: Hệ thống hóa kiến thức")
if kn_1:
    selected_criteria_labels.append("KN: Giải quyết vấn đề")
if kn_2:
    selected_criteria_labels.append("KN: Làm việc nhóm")
if kn_3:
    selected_criteria_labels.append("KN: Thuyết trình & Giao tiếp")
if a_1:
    selected_criteria_labels.append("A: Tính chủ động")
if a_2:
    selected_criteria_labels.append("A: Sự kiên trì")
if a_3:
    selected_criteria_labels.append("A: Kỷ luật & Chuyên cần")
if sl_1:
    selected_criteria_labels.append("SL: Tốc độ tiếp thu")
if sl_2:
    selected_criteria_labels.append("SL: Số lượng bài tập đã hoàn thành")
if sl_3:
    selected_criteria_labels.append("SL: Tỉ lệ lỗi")

st.markdown("#### Chiến lược chia nhóm học tập")
group_mode = st.radio(
    "Mục tiêu chia nhóm",
    options=[
        "Cân bằng năng lực (Heterogeneous Grouping)",
        "Đồng nhất năng lực (Homogeneous Grouping)",
        "Cảnh báo & Hỗ trợ (Intervention - AIS)",
    ],
    horizontal=True,
)

skill_option_map = {
    "Giải quyết vấn đề (Problem Solving)": "KN_ProblemSolving_norm",
    "Làm việc nhóm (Teamwork)": "KN_Teamwork_norm",
    "Thuyết trình & Giao tiếp": "KN_Communication_norm",
}

# Tự chọn kỹ năng theo dõi dựa trên checkbox KN đã chọn, không cần thêm input UI.
if kn_1:
    selected_skill_col = skill_option_map["Giải quyết vấn đề (Problem Solving)"]
    selected_skill_label = "Giải quyết vấn đề (Problem Solving)"
elif kn_2:
    selected_skill_col = skill_option_map["Làm việc nhóm (Teamwork)"]
    selected_skill_label = "Làm việc nhóm (Teamwork)"
elif kn_3:
    selected_skill_col = skill_option_map["Thuyết trình & Giao tiếp"]
    selected_skill_label = "Thuyết trình & Giao tiếp"
else:
    selected_skill_col = skill_option_map["Giải quyết vấn đề (Problem Solving)"]
    selected_skill_label = "Giải quyết vấn đề (Problem Solving)"

group_size = st.slider("Sĩ số mỗi nhóm", min_value=3, max_value=8, value=5, step=1)

if run_btn:
    try:
        # 1) Thu thập dữ liệu
        if uploaded_files:
            raw_df = read_and_merge_uploaded_files(uploaded_files)
            source_file_count = raw_df["__SourceFile"].nunique() if "__SourceFile" in raw_df.columns else 0
            st.success(
                f"Đã đọc và gộp dữ liệu từ {source_file_count} file. Tổng số dòng trước làm sạch: {len(raw_df)}."
            )
        else:
            raw_df = create_dummy_data(n_students=50, random_state=42)
            st.info("Không có file tải lên. Hệ thống đang dùng dữ liệu giả lập 50 sinh viên.")

        # 2) Kiểm tra dữ liệu đầu vào
        ensure_question_columns(raw_df)

        # Đảm bảo có cột định danh cơ bản
        if "StudentID" not in raw_df.columns:
            raw_df.insert(0, "StudentID", [f"SV{str(i).zfill(3)}" for i in range(1, len(raw_df) + 1)])
        if "Họ tên" not in raw_df.columns:
            raw_df.insert(1, "Họ tên", [f"Sinh viên {i}" for i in range(1, len(raw_df) + 1)])

        # Làm sạch StudentID trùng do gộp file
        raw_df, duplicate_count = resolve_duplicate_student_ids(raw_df, policy=dedupe_policy)
        if duplicate_count > 0:
            if dedupe_policy == "make_unique":
                st.warning(
                    f"Phát hiện {duplicate_count} dòng có StudentID bị trùng. Đã tự đổi ID để giữ toàn bộ bản ghi."
                )
            else:
                st.warning(
                    f"Phát hiện {duplicate_count} dòng có StudentID bị trùng. Đã xử lý theo lựa chọn: {dedupe_mode_label}."
                )

        # Cột kỹ thuật chỉ dùng để truy vết file nguồn, không cần trong xử lý tiếp theo
        if "__SourceFile" in raw_df.columns:
            raw_df = raw_df.drop(columns=["__SourceFile"])

        # 3) Tiền xử lý
        scored_df = preprocess_scores(raw_df)

        if selected_criteria_labels:
            st.info("Tiêu chí cụ thể đang chọn: " + " | ".join(selected_criteria_labels))

        if len(selected_feature_cols) == 0:
            st.error("Vui lòng chọn ít nhất 1 tiêu chí để phân tích.")
            st.stop()

        # 4) K-Means + gán nhãn cụm năng lực
        clustered_df, kmeans_model = run_kmeans_and_label(scored_df, selected_feature_cols)

        # 4.1) Bổ sung mức tương đối theo vị trí trong lớp (percentile)
        clustered_df["ASK_Điểm_tổng_hợp"] = (
            0.4 * clustered_df["A_norm"]
            + 0.4 * clustered_df["KN_norm"]
            + 0.2 * clustered_df["TR_norm"]
        )
        clustered_df["ASK_Bách_phân_vị"] = (
            clustered_df["ASK_Điểm_tổng_hợp"].rank(pct=True, method="average") * 100
        ).round(1)
        clustered_df["Nhãn năng lực (tương đối)"] = (
            clustered_df["Nhãn năng lực"]
            + " | P"
            + clustered_df["ASK_Bách_phân_vị"].astype(str)
        )

        # 5) KNN classifier (sẵn sàng cho sinh viên mới)
        knn_model = train_knn_classifier(clustered_df, selected_feature_cols)

        # 6) AIS cảnh báo
        final_df = run_ais_alert(clustered_df)

        final_df["Điểm kỹ năng mục tiêu"] = final_df[selected_skill_col].round(3)

        # 7) DS - chia nhóm học tập
        if group_mode.startswith("Cân bằng"):
            grouping_strategy = "balanced"
        elif group_mode.startswith("Đồng nhất"):
            grouping_strategy = "homogeneous"
        else:
            grouping_strategy = "intervention"

        grouped_df = build_learning_groups(
            final_df,
            group_size=group_size,
            random_state=42,
            strategy=grouping_strategy,
            target_skill_col=selected_skill_col,
            skill_cols=tuple(skill_option_map.values()),
        )

        # Đưa cột nhóm về final
        final_df = final_df.merge(
            grouped_df[["StudentID", "Nhóm học tập"]].drop_duplicates(),
            on="StudentID",
            how="left",
        )

        # --------------------------------------------------
        # Hiển thị kết quả Module 1 + Module 4
        # --------------------------------------------------
        st.subheader("1) Bảng dữ liệu thô (Raw Data)")
        st.dataframe(raw_df, use_container_width=True, height=280)

        st.subheader("2) Thống kê SL - Số lượng sinh viên mỗi cụm")
        count_df = (
            final_df["Nhãn năng lực"]
            .value_counts()
            .rename_axis("Nhãn năng lực")
            .reset_index(name="Số lượng")
        )
        count_df["order"] = count_df["Nhãn năng lực"].map(lambda x: LABEL_ORDER.get(x, 999))
        count_df = count_df.sort_values("order").drop(columns=["order"])
        st.dataframe(count_df, use_container_width=True)

        st.subheader("3) Danh sách gợi ý chia nhóm học tập (DS)")
        if grouping_strategy == "balanced":
            st.caption("Chế độ hiện tại: Cân bằng năng lực giữa các nhóm.")
        elif grouping_strategy == "homogeneous":
            st.caption("Chế độ hiện tại: Đồng nhất năng lực, nhóm theo trình độ tương đồng.")
        else:
            st.caption("Chế độ hiện tại: Cảnh báo & Hỗ trợ (AIS), sinh viên ALERT được tách nhóm can thiệp.")

        group_view = final_df[
            [
                "Nhóm học tập",
                "StudentID",
                "Họ tên",
                "Nhãn năng lực",
                "Nhãn năng lực (tương đối)",
                "Điểm kỹ năng mục tiêu",
                "AIS_Cảnh báo",
            ]
        ].sort_values(
            ["Nhóm học tập", "Nhãn năng lực", "StudentID"],
            key=lambda col: col.map(LABEL_ORDER) if col.name == "Nhãn năng lực" else col,
        )
        st.dataframe(group_view, use_container_width=True, height=350)

        # KNN demo nhanh cho sinh viên mới (theo tiêu chí đã chọn)
        with st.expander("Dự đoán nhanh cho sinh viên mới bằng KNN (tùy chọn)"):
            st.write("Nhập các điểm đã chuẩn hóa (0.0 - 1.0) theo tiêu chí đang chọn.")
            new_point = []
            for feature in selected_feature_cols:
                val = st.slider(f"{feature}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                new_point.append(val)

            if st.button("Dự đoán nhãn cho sinh viên mới"):
                pred = knn_model.predict([new_point])[0]
                st.success(f"KNN dự đoán: {pred}")

        # Nút download kết quả
        output_cols = [
            "StudentID",
            "Họ tên",
            "TR",
            "KN",
            "A",
            "SL",
            "TR_norm",
            "KN_norm",
            "A_norm",
            "SL_norm",
            "ClusterID",
            "Nhãn năng lực",
            "ASK_Điểm_tổng_hợp",
            "ASK_Bách_phân_vị",
            "Nhãn năng lực (tương đối)",
            "Điểm kỹ năng mục tiêu",
            "AIS_Cảnh báo",
            "Nhóm học tập",
        ]
        export_df = final_df[output_cols].copy()
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download CSV kết quả cuối cùng",
            data=csv_data,
            file_name="ket_qua_phan_loai_ASK.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
else:
    st.info("Chọn tiêu chí và nhấn 'Chạy phân tích' để bắt đầu.")
