"""
Module quản lý nhóm học tập: đổi tên, xóa nhóm, xóa thành viên
"""

import pandas as pd


def rename_group(df, old_name, new_name):
    """Đổi tên nhóm học tập."""
    result_df = df.copy()
    result_df["Nhóm học tập"] = result_df["Nhóm học tập"].replace(old_name, new_name)
    return result_df


def remove_student_from_group(df, mssv):
    """Xóa sinh viên khỏi nhóm (xóa dòng)."""
    result_df = df.copy()
    result_df = result_df[result_df["MSSV"] != str(mssv)].reset_index(drop=True)
    return result_df


def delete_group(df, group_name):
    """Xóa toàn bộ nhóm (xóa tất cả sinh viên trong nhóm)."""
    result_df = df.copy()
    result_df = result_df[result_df["Nhóm học tập"] != group_name].reset_index(drop=True)
    return result_df


def get_group_list(df):
    """Lấy danh sách tất cả nhóm."""
    if "Nhóm học tập" not in df.columns:
        return []
    return sorted(df["Nhóm học tập"].unique().tolist())


def get_group_members(df, group_name):
    """Lấy danh sách thành viên của 1 nhóm."""
    if "Nhóm học tập" not in df.columns:
        return pd.DataFrame()
    return df[df["Nhóm học tập"] == group_name].copy()


def move_student_to_group(df, mssv, target_group):
    """Di chuyển sinh viên sang nhóm khác."""
    result_df = df.copy()
    mssv = str(mssv)
    
    # Tìm sinh viên
    if mssv not in result_df["MSSV"].astype(str).values:
        raise ValueError(f"Không tìm thấy sinh viên {mssv}")
    
    # Cập nhật nhóm
    result_df.loc[result_df["MSSV"].astype(str) == mssv, "Nhóm học tập"] = target_group
    return result_df


def merge_groups(df, source_group, target_group):
    """Gộp 2 nhóm lại (chuyển tất cả members từ source sang target rồi xóa source)."""
    result_df = df.copy()
    result_df.loc[result_df["Nhóm học tập"] == source_group, "Nhóm học tập"] = target_group
    return result_df


def split_group(df, group_name, num_groups=2):
    """Tách 1 nhóm thành nhiều nhóm nhỏ hơn (chia đều thành viên)."""
    result_df = df.copy()
    group_members = result_df[result_df["Nhóm học tập"] == group_name].copy()
    
    if len(group_members) == 0:
        return result_df
    
    # Tìm tên nhóm mới (tăng số lần trong danh sách nhóm)
    existing_groups = result_df["Nhóm học tập"].unique().tolist()
    max_num = 0
    for g in existing_groups:
        if g.startswith(group_name.split()[0]):  # Lấy tiền tố
            try:
                num = int(g.split()[-1])
                max_num = max(max_num, num)
            except:
                pass
    
    # Chia thành viên
    chunk_size = (len(group_members) + num_groups - 1) // num_groups
    for i in range(num_groups):
        start = i * chunk_size
        end = start + chunk_size
        chunk = group_members.iloc[start:end]
        
        new_group_name = f"{group_name} {i+1}"
        result_df.loc[chunk.index, "Nhóm học tập"] = new_group_name
    
    return result_df
