import os
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_FILE = os.path.join(DATA_DIR, 'heart_2022_no_nans.csv')

# --- 2. THỨ TỰ ORDINAL - KHỚP 100% VỚI NOTEBOOK GỐC ---
AGE_ORDER = [
    "18-24", "25-29", "30-34", "35-39", "40-44",
    "45-49", "50-54", "55-59", "60-64", "65-69",
    "70-74", "75-79", "80 or older"
]

GEN_HEALTH_ORDER = ["Excellent", "Very good", "Good", "Fair", "Poor"]

BMI_CAT_ORDER = [0, 1, 2, 3]  # Underweight → Obese

# --- 3. DANH SÁCH FEATURES CUỐI CÙNG SAU ENGINEERING (Đúng thứ tự train) ---
EXPECTED_FEATURES = [
    "AgeCategory", "GeneralHealth", "Sex", "SmokerStatus",
    "PhysicalActivities", "HadDiabetes", "ChestScan",
    "AlcoholDrinkers", "HadArthritis",
    "BMI_cat", "Sleep_risk"
]

# --- 4. FEATURE ENGINEERING (Dùng chung cho train và predict) ---
def process_age_category(age_val):
    """Chuẩn hóa AgeCategory: 'Age 18 to 24' → '18-24'"""
    if pd.isna(age_val):
        return "18-24"
    return str(age_val).replace("Age ", "").replace(" to ", "-").strip()

def get_bmi_cat_code(bmi: float) -> int:
    """Phân loại BMI theo chuẩn WHO"""
    if pd.isna(bmi) or bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3

def get_sleep_risk_code(hours: float) -> int:
    """Rủi ro nếu ngủ <6 hoặc >9 tiếng → 1"""
    if pd.isna(hours):
        return 0
    return 1 if hours < 6 or hours > 9 else 0

# --- 5. PREPROCESSOR - KHỚP CHÍNH XÁC VỚI PIPELINE TRONG NOTEBOOK ---
def get_preprocessor():
    """
    Tái tạo ColumnTransformer giống hệt notebook:
    - Ordinal: AgeCategory, GeneralHealth, BMI_cat
    - OneHot: Sex, SmokerStatus, HadDiabetes
    - Passthrough: Các binary (đã encode 0/1)
    """
    ordinal_features = ["AgeCategory", "GeneralHealth", "BMI_cat"]
    nominal_features = ["Sex", "SmokerStatus", "HadDiabetes"]
    binary_features = [
        "PhysicalActivities", "AlcoholDrinkers",
        "ChestScan", "HadArthritis", "Sleep_risk"
    ]

    ordinal_pipeline = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(
            categories=[AGE_ORDER, GEN_HEALTH_ORDER, BMI_CAT_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    nominal_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_pipeline, ordinal_features),
            ("nominal", nominal_pipeline, nominal_features),
            ("binary", "passthrough", binary_features)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor

# --- 6. HÀM XỬ LÝ ĐẦU VÀO ĐƠN (DÀNH RIÊNG CHO API PREDICT) ---
def preprocess_single_input(raw_data: dict) -> pd.DataFrame:
    """
    Chuyển dict input từ frontend thành DataFrame đã qua feature engineering,
    sẵn sàng đưa vào model.predict_proba()
    """
    processed = {}

    # 1. Các feature cần engineering
    processed["AgeCategory"] = process_age_category(raw_data.get("AgeCategory"))
    processed["BMI_cat"] = get_bmi_cat_code(raw_data.get("BMI"))
    processed["Sleep_risk"] = get_sleep_risk_code(raw_data.get("SleepHours"))

    # 2. Binary features: Yes/No → 1/0
    binary_map = {"Yes": 1, "No": 0}
    for col in ["PhysicalActivities", "ChestScan", "AlcoholDrinkers", "HadArthritis"]:
        val = raw_data.get(col, "No")
        processed[col] = binary_map.get(val, 0)

    # 3. Nominal & ordinal giữ nguyên string (sẽ được xử lý bởi preprocessor)
    for col in ["GeneralHealth", "Sex", "SmokerStatus", "HadDiabetes"]:
        processed[col] = raw_data.get(col)

    # Tạo DataFrame và sắp xếp đúng thứ tự
    df = pd.DataFrame([processed])
    df = df[EXPECTED_FEATURES]  # Đảm bảo thứ tự và đầy đủ cột

    return df

# --- 7. HÀM HỖ TRỢ LOAD/SAVE DATA & MODEL ---
def load_dataset() -> pd.DataFrame:
    """Load file CSV dữ liệu chính"""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {DATA_FILE}")
    print(f"✅ Đã tải dữ liệu từ: {DATA_FILE}")
    return pd.read_csv(DATA_FILE)

def save_model_bundle(bundle: dict, filename: str):
    """Lưu model bundle (pipeline + threshold + metadata)"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(bundle, path)
    print(f"✅ Đã lưu model: {path}")

def load_model_bundle(filename: str):
    """Load model bundle từ file .pkl"""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Không tìm thấy model: {path}")
    bundle = joblib.load(path)
    print(f"✅ Đã tải model: {path}")
    return bundle