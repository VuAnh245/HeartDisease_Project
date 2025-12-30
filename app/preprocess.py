"""
Module chuyên trách xây dựng preprocessor (ColumnTransformer + Pipeline)
để xử lý dữ liệu trước khi đưa vào mô hình.

Được sử dụng bởi:
- train.py: Khi huấn luyện mô hình
- main.py (API): Khi load model và predict (pipeline đã bao gồm preprocessor)

Tất cả logic phải khớp chính xác với utils.py và notebook gốc.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Import các thứ tự ordinal từ utils.py để đảm bảo đồng bộ tuyệt đối
from .utils import (
    AGE_ORDER,
    GEN_HEALTH_ORDER,
    BMI_CAT_ORDER,
    EXPECTED_FEATURES,  # Dùng để kiểm tra (tùy chọn)
)


def get_preprocessor():
    """
    Tạo ColumnTransformer giống hệt như trong notebook gốc.

    Phân nhóm feature:
    - Ordinal: AgeCategory, GeneralHealth, BMI_cat → có thứ tự rõ ràng
    - Nominal: Sex, SmokerStatus, HadDiabetes → OneHot
    - Binary: Các feature đã được encode thành 0/1 → passthrough

    Thêm SimpleImputer để tăng tính robust khi có dữ liệu thiếu (dù file no_nans).
    """
    
    # Định nghĩa nhóm feature (phải khớp chính xác với utils.EXPECTED_FEATURES)
    ordinal_features = ["AgeCategory", "GeneralHealth", "BMI_cat"]
    nominal_features = ["Sex", "SmokerStatus", "HadDiabetes"]
    binary_features = [
        "PhysicalActivities",
        "AlcoholDrinkers",
        "ChestScan",
        "HadArthritis",
        "Sleep_risk"
    ]

    # --- Ordinal Pipeline ---
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[AGE_ORDER, GEN_HEALTH_ORDER, BMI_CAT_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    # --- Nominal Pipeline (OneHot) ---
    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False  # Trả về numpy array để dễ xử lý
        ))
    ])

    # --- Binary Pipeline (chỉ impute, giữ nguyên 0/1) ---
    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),  # Mặc định No → 0
        ("passthrough", "passthrough")
    ])

    # --- Tổng hợp ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_pipeline, ordinal_features),
            ("nominal", nominal_pipeline, nominal_features),
            ("binary",  binary_pipeline,  binary_features)
        ],
        remainder="drop",                    # Bỏ tất cả cột không dùng
        verbose_feature_names_out=False      # Không thêm prefix để giữ tên sạch
    )

    return preprocessor


# Optional: Hàm kiểm tra nhanh khi chạy riêng file này
if __name__ == "__main__":
    from .utils import load_dataset, preprocess_single_input
    import numpy as np

    preprocessor = get_preprocessor()
    print("Preprocessor được tạo thành công!")
    print("Các feature mong đợi:", EXPECTED_FEATURES)
    print("\nVí dụ xử lý 1 bản ghi:")

    # Test với dữ liệu mẫu
    sample_raw = {
        "AgeCategory": "Age 65 to 69",
        "BMI": 32.5,
        "GeneralHealth": "Fair",
        "Sex": "Male",
        "SmokerStatus": "Former smoker",
        "PhysicalActivities": "No",
        "SleepHours": 5,
        "HadDiabetes": "Yes",
        "ChestScan": "Yes",
        "AlcoholDrinkers": "No",
        "HadArthritis": "Yes"
    }

    df_input = preprocess_single_input(sample_raw)
    transformed = preprocessor.fit_transform(df_input)
    
    print("Input sau engineering:\n", df_input)
    print(f"Shape sau transform: {transformed.shape}")
    print("Sample output (một phần):", transformed[0][:10])