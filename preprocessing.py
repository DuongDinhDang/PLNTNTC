"""
preprocessing.py - Đọc, chuẩn hóa và xử lý dữ liệu file
"""

import unicodedata
import numpy as np
import pandas as pd
from config import LIKERT_MAP


def create_dummy_data(n_students=50, random_state=42):
    """Tạo dữ liệu giả lập 50 sinh viên, 60 câu hỏi (thang 1-5)."""
    rng = np.random.default_rng(random_state)

    student_ids = [f"SV{str(i).zfill(3)}" for i in range(1, n_students + 1)]
    full_names = [f"Sinh viên {i}" for i in range(1, n_students + 1)]

    base_levels = np.array([1.8, 2.4, 3.0, 3.7, 4.4])
    student_level = rng.integers(0, 5, size=n_students)

    data = {
        "MSSV": student_ids,
        "Họ tên": full_names,
    }

    for q in range(1, 61):
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


def normalize_likert_value(value):
    """Chuyển câu trả lời Likert dạng chữ sang thang điểm 1-5."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer, float, np.floating)):
        if pd.isna(value):
            return np.nan
        if 1 <= value <= 5:
            return value
        return np.nan

    text = str(value).strip().lower()

    try:
        numeric_value = float(text)
        if 1 <= numeric_value <= 5:
            if numeric_value.is_integer():
                return int(numeric_value)
            return numeric_value
        return np.nan
    except ValueError:
        pass

    normalized_text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    normalized_text = " ".join(normalized_text.split())

    result = LIKERT_MAP.get(normalized_text, np.nan)
    return result


def normalize_column_headers(df):
    """Chuẩn hóa tên cột để phù hợp với các biến thể từ Google Form."""
    col_mapping = {}
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        if col_lower in ["họ tên", "họ và tên"]:
            col_mapping[col] = "Họ tên"
        elif col_lower == "mssv:":
            col_mapping[col] = "MSSV"
    
    if col_mapping:
        df = df.rename(columns=col_mapping)
    
    return df


def read_uploaded_file(uploaded_file):
    """Đọc 1 file CSV/Excel từ uploader."""
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {uploaded_file.name}")

    df = normalize_column_headers(df)

    question_cols = [c for c in df.columns if str(c).startswith("Q")]
    for col in question_cols:
        df[col] = df[col].apply(normalize_likert_value)

    return df


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
    Xử lý trùng MSSV sau khi gộp nhiều file.
    """
    out_df = df.copy()
    if "MSSV" not in out_df.columns:
        return out_df, 0

    sid_series = out_df["MSSV"].astype(str).str.strip()
    sid_series = sid_series.replace({"":np.nan, "nan": np.nan, "None": np.nan})
    out_df["MSSV"] = sid_series

    dup_count = int(out_df["MSSV"].duplicated(keep=False).sum())
    if dup_count == 0:
        return out_df, 0

    if policy == "keep_first":
        out_df = out_df.drop_duplicates(subset=["MSSV"], keep="first")
        return out_df.reset_index(drop=True), dup_count

    if policy == "keep_last":
        out_df = out_df.drop_duplicates(subset=["MSSV"], keep="last")
        return out_df.reset_index(drop=True), dup_count

    # make_unique
    seen = {}
    new_ids = []
    for sid in out_df["MSSV"].tolist():
        if pd.isna(sid):
            new_ids.append(sid)
            continue

        if sid not in seen:
            seen[sid] = 1
            new_ids.append(sid)
        else:
            seen[sid] += 1
            new_ids.append(f"{sid}_{seen[sid]}")

    out_df["MSSV"] = new_ids
    return out_df, dup_count


def ensure_question_columns(df):
    """Đảm bảo dữ liệu có đủ cột Q1..Q60."""
    expected = [f"Q{i}" for i in range(1, 61)]
    missing = [c for c in expected if c not in df.columns]

    if missing:
        raise ValueError(
            "Thiếu các cột câu hỏi: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    return expected
