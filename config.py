"""
config.py - Các hằng số cấu hình chung cho ASK
"""

ABILITY_LABELS = ["Yếu (At Risk)", "Trung bình", "Khá", "Giỏi", "Xuất sắc"]
LABEL_ORDER = {label: idx for idx, label in enumerate(ABILITY_LABELS)}

LIKERT_MAP = {
    "hoan toan khong dong y": 1,
    "hoan toan khong đong y": 1,
    "khong dong y": 2,
    "khong đong y": 2,
    "phan van": 3,
    "dong y": 4,
    "đong y": 4,
    "hoan toan dong y": 5,
    "hoan toan đong y": 5,
}
