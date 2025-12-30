# app/predict.py
"""
Module chuyên trách dự đoán (prediction) – dùng cho:
- CLI demo/test nhanh mô hình
- Kiểm tra end-to-end: input → feature engineering → predict → kết quả

Logic 100% nhất quán với:
- utils.py (đường dẫn, feature engineering, EXPECTED_FEATURES)
- preprocess.py (preprocessor trong pipeline)
- main.py (API prediction logic)
"""

import os
import pandas as pd
import joblib

# Import chuẩn từ utils.py để đảm bảo đồng bộ tuyệt đối
from .utils import (
    MODEL_DIR,
    EXPECTED_FEATURES,          # Đã đổi từ FINAL_FEATURES → EXPECTED_FEATURES
    preprocess_single_input,    # Hàm mạnh nhất: xử lý toàn bộ feature engineering
    load_model_bundle           # Hàm load model an toàn, có thông báo
)


def predict_cli(model_name: str = "xgboost_high_recall.pkl"):
    """
    CLI đơn giản để test nhanh mô hình đã train.
    Mục đích: Xác nhận toàn bộ pipeline hoạt động đúng từ input thô đến kết quả.
    """
    print("\n" + "="*70)
    print("🏥 HEART DISEASE RISK SCREENING - CLI PREDICTION TOOL")
    print("="*70)

    # 1. Load model bundle
    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model: {model_path}")
        print("👉 Hãy chạy lệnh huấn luyện trước:")
        print("   python -m app.train")
        return

    try:
        bundle = load_model_bundle(model_name)  # Dùng hàm chuẩn từ utils
        pipeline = bundle['pipeline']
        threshold = bundle.get('threshold', 0.2)  # Fallback nếu bundle không có threshold
        print(f"✅ Đã tải thành công model: {model_name}")
        print(f"   • Threshold high-recall được lưu: {threshold:.3f}")
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        return

    # 2. Dữ liệu đầu vào giả lập (trường hợp nguy cơ rất cao)
    raw_input = {
        "AgeCategory": "Age 65 to 69",                   # → 65-69
        "BMI": 33.8,                                     # → BMI_cat = 3 (Obese)
        "SleepHours": 4,                                 # → Sleep_risk = 1 (<6h)
        "GeneralHealth": "Poor",                         # Rất kém
        "Sex": "Male",
        "SmokerStatus": "Current smoker - now smokes every day",
        "PhysicalActivities": "No",
        "HadDiabetes": "Yes",
        "ChestScan": "Yes",                              # Từng nghi ngờ tim/phổi
        "AlcoholDrinkers": "Yes",
        "HadArthritis": "Yes"
    }

    print("\n📋 DỮ LIỆU ĐẦU VÀO TỪ NGƯỜI DÙNG (RAW):")
    for key, value in raw_input.items():
        print(f"   • {key.ljust(20)} : {value}")

    # 3. Feature Engineering – dùng hàm chuẩn từ utils (tránh viết lại logic)
    try:
        df_pred = preprocess_single_input(raw_input)
        print(f"\n✅ Feature engineering hoàn tất (sử dụng preprocess_single_input)")
        print(f"   • Shape: {df_pred.shape}")
        print(f"   • Các cột: {list(df_pred.columns)}")
    except Exception as e:
        print(f"❌ Lỗi trong quá trình feature engineering: {e}")
        return

    # 4. Dự đoán
    try:
        proba = pipeline.predict_proba(df_pred)[0][1]           # Xác suất lớp 1
        probability_percent = proba * 100

        print("\n" + "─"*70)
        print("📊 KẾT QUẢ DỰ ĐOÁN TỪ MÔ HÌNH AI")
        print("─"*70)
        print(f"   • Xác suất nguy cơ cao (raw probability) : {probability_percent:.2f}%")
        print(f"   • Ngưỡng high-recall đã lưu trong model   : {threshold * 100:.2f}%")

        if proba >= threshold:
            print("\n   ⚠️  KẾT LUẬN: NGUY CƠ CAO (HIGH RISK)")
            print("   → Khuyến nghị: Đi khám chuyên khoa Tim mạch NGAY LẬP TỨC!")
        else:
            print("\n   ✅  KẾT LUẬN: NGUY CƠ THẤP (LOW RISK)")
            print("   → Khuyến nghị: Tiếp tục duy trì lối sống lành mạnh.")

        print("="*70 + "\n")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình dự đoán: {e}")


# Hàm tiện ích: Dự đoán từ dict bất kỳ (có thể import dùng ở nơi khác)
def predict_from_dict(raw_data: dict, model_name: str = "xgboost_high_recall.pkl") -> dict:
    """
    Hàm tái sử dụng để predict từ dữ liệu dict (giống logic API).
    Trả về dict kết quả dễ đọc.
    """
    bundle = load_model_bundle(model_name)
    pipeline = bundle['pipeline']
    threshold = bundle.get('threshold', 0.2)

    df = preprocess_single_input(raw_data)
    proba = pipeline.predict_proba(df)[0][1]

    return {
        "model_used": model_name,
        "probability_percent": round(proba * 100, 2),
        "threshold": round(threshold, 3),
        "final_risk_level": "Cao" if proba >= threshold else "Thấp",
        "prediction_class": 1 if proba >= threshold else 0
    }


# Chạy CLI khi gọi trực tiếp file
if __name__ == "__main__":
    # Bạn có thể thay đổi model để test nhanh
    available_models = [
        "logistic_regression_high_recall.pkl",
        "random_forest_high_recall.pkl",
        "xgboost_high_recall.pkl"
    ]
    print("Các model có sẵn trong thư mục models/:")
    for m in available_models:
        path = os.path.join(MODEL_DIR, m)
        status = "✅ Có" if os.path.exists(path) else "❌ Không có"
        print(f"   • {m} → {status}")

    # Chạy với model ưu tiên (XGBoost nếu có)
    chosen = "xgboost_high_recall.pkl"
    if not os.path.exists(os.path.join(MODEL_DIR, chosen)):
        chosen = "random_forest_high_recall.pkl"  # Fallback

    predict_cli(model_name=chosen)