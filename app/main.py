import uvicorn
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'models'))

models = {}
thresholds = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*70)
    print(f"🚀 KHỞI ĐỘNG SERVER - ĐANG TẢI MODEL TỪ: {MODEL_DIR}")

    model_files = {
        'rf': 'random_forest_high_recall.pkl',
        'logreg': 'logistic_regression_high_recall.pkl',
        'xgb': 'xgboost_high_recall.pkl'
    }

    loaded_count = 0
    for key, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                bundle = joblib.load(path)
                pipeline = bundle['pipeline'] if isinstance(bundle, dict) and 'pipeline' in bundle else bundle
                threshold = bundle.get('threshold', 0.2)  # fallback nếu không có threshold trong bundle
                models[key] = pipeline
                thresholds[key] = threshold
                print(f"   ✅ [{key.upper()}] Loaded – Threshold: {threshold:.3f}")
                loaded_count += 1
            except Exception as e:
                print(f"   ❌ [{key.upper()}] Lỗi load: {e}")
        else:
            print(f"   ⚠️ [{key.upper()}] Không tìm thấy file: {filename}")

    if loaded_count == 0:
        raise RuntimeError("❌ Không load được model nào! Hãy chạy python -m app.train trước.")

    print("="*70 + "\n")
    yield
    models.clear()
    thresholds.clear()

app = FastAPI(
    lifespan=lifespan,
    title="Heart Disease Risk Screening API",
    description="API sàng lọc nguy cơ tim mạch – High Recall Mode",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMA ĐẦU VÀO ---
class HealthData(BaseModel):
    model_selection: str = "xgb"  # mặc định XGBoost
    AgeCategory: str
    BMI: float
    GeneralHealth: str
    Sex: str
    SmokerStatus: str
    PhysicalActivities: str
    SleepHours: int
    HadDiabetes: str
    ChestScan: str
    AlcoholDrinkers: str
    HadArthritis: str

# --- UTILS ---
def process_age_category(age_str: str) -> str:
    mapping = {
        "18-24": "18-24", "25-29": "25-29", "30-34": "30-34", "35-39": "35-39",
        "40-44": "40-44", "45-49": "45-49", "50-54": "50-54", "55-59": "55-59",
        "60-64": "60-64", "65-69": "65-69", "70-74": "70-74", "75-79": "75-79",
        "80 or older": "80 or older"
    }
    cleaned = str(age_str).strip().replace("Age ", "").replace(" to ", "-")
    return mapping.get(cleaned, "18-24")

def get_bmi_cat(bmi: float) -> int:
    if bmi < 18.5: return 0
    elif bmi < 25: return 1
    elif bmi < 30: return 2
    else: return 3

def get_sleep_risk(hours: int) -> int:
    return 1 if hours < 6 or hours > 9 else 0

EXPECTED_FEATURES = [
    "AgeCategory", "GeneralHealth", "Sex", "SmokerStatus",
    "PhysicalActivities", "HadDiabetes", "ChestScan",
    "AlcoholDrinkers", "HadArthritis",
    "BMI_cat", "Sleep_risk"
]

def get_risk_assessment(prob: float):
    if prob < 20:
        return "Thấp", "#1cc88a"
    elif prob < 50:
        return "Trung bình", "#f6c23e"
    elif prob < 80:
        return "Cao", "#fd7e14"
    else:
        return "Rất cao", "#dc3545"

def generate_advice(raw_data: dict, prob: float) -> list:
    advice = []

    # Đánh giá tổng quát
    if prob >= 80:
        advice.append("⚠️ <b>CẢNH BÁO KHẨN CẤP:</b> Nguy cơ tim mạch rất cao. Hãy đến khám chuyên khoa tim mạch NGAY LẬP TỨC!")
    elif prob >= 50:
        advice.append("⚠️ <b>NGUY CƠ CAO:</b> Cần theo dõi y tế định kỳ và thay đổi lối sống khẩn trương.")
    elif prob >= 20:
        advice.append("🛡️ <b>LƯU Ý:</b> Đã có một số yếu tố nguy cơ. Nên cải thiện lối sống để phòng ngừa.")
    else:
        advice.append("✅ <b>TỐT:</b> Nguy cơ hiện tại ở mức thấp. Hãy duy trì thói quen lành mạnh!")

    # Lời khuyên cụ thể
    if raw_data.get('HadDiabetes') == 'Yes':
        advice.append("🩸 <b>Tiểu đường:</b> Đây là yếu tố nguy cơ MẠNH NHẤT với bệnh tim. Kiểm soát đường huyết là ưu tiên hàng đầu.")

    bmi = raw_data.get('BMI', 0)
    if bmi >= 30:
        advice.append("⚖️ <b>Béo phì:</b> Giảm cân (dù chỉ 5-10%) sẽ giảm đáng kể áp lực lên tim.")
    elif bmi >= 25:
        advice.append("⚖️ <b>Thừa cân:</b> Cân nặng đang ở mức cần chú ý. Kết hợp ăn uống và vận động để kiểm soát.")

    sleep = raw_data.get('SleepHours', 7)
    if sleep < 6:
        advice.append("💤 <b>Thiếu ngủ nghiêm trọng:</b> Ngủ dưới 6 tiếng thường xuyên làm tăng viêm và nguy cơ cao huyết áp.")
    elif sleep > 9:
        advice.append("💤 <b>Ngủ quá nhiều:</b> Có thể liên quan đến các vấn đề sức khỏe khác, nên tham khảo bác sĩ.")

    smoker = raw_data.get('SmokerStatus', '').lower()
    if 'current' in smoker:
        advice.append("🚬 <b>Hút thuốc lá:</b> Cai thuốc là biện pháp HIỆU QUẢ NHẤT để giảm nguy cơ tim mạch ngay lập tức.")

    if raw_data.get('PhysicalActivities') == 'No':
        advice.append("🏃 <b>Ít vận động:</b> Hãy dành ít nhất 30 phút/ngày cho hoạt động thể chất (đi bộ nhanh, đạp xe...).")

    if raw_data.get('ChestScan') == 'Yes':
        advice.append("🩻 <b>Từng chụp CT ngực:</b> Có thể đã có dấu hiệu bất thường trước đây. Nên theo dõi định kỳ.")

    if raw_data.get('AlcoholDrinkers') == 'Yes':
        advice.append("🍷 <b>Uống rượu nặng:</b> Hạn chế rượu bia giúp kiểm soát huyết áp và bảo vệ tim mạch.")

    if raw_data.get('HadArthritis') == 'Yes':
        advice.append("🦴 <b>Viêm khớp:</b> Viêm mãn tính có thể liên quan đến xơ vữa động mạch. Kiểm soát viêm tốt sẽ có lợi cho tim.")

    return advice

# --- ENDPOINT CHÍNH ---
@app.post("/predict")
def predict_heart_disease(data: HealthData):
    if not models:
        raise HTTPException(status_code=500, detail="Server chưa tải model. Vui lòng khởi động lại.")

    try:
        # Chuyển Pydantic model thành dict
        raw = data.dict()
        model_key = raw.get('model_selection', 'xgb').lower()

        # Fallback nếu model không tồn tại
        if model_key not in models:
            model_key = 'xgb'

        pipeline = models[model_key]
        threshold = thresholds.get(model_key, 0.2)

        # Feature Engineering
        processed = {
            'AgeCategory': process_age_category(raw['AgeCategory']),
            'GeneralHealth': raw['GeneralHealth'],
            'Sex': raw['Sex'],
            'SmokerStatus': raw['SmokerStatus'],
            'PhysicalActivities': 1 if raw['PhysicalActivities'] == 'Yes' else 0,
            'HadDiabetes': raw['HadDiabetes'],
            'ChestScan': 1 if raw['ChestScan'] == 'Yes' else 0,
            'AlcoholDrinkers': 1 if raw['AlcoholDrinkers'] == 'Yes' else 0,
            'HadArthritis': 1 if raw['HadArthritis'] == 'Yes' else 0,
            'BMI_cat': get_bmi_cat(raw['BMI']),
            'Sleep_risk': get_sleep_risk(raw['SleepHours'])
        }

        df = pd.DataFrame([processed])[EXPECTED_FEATURES]

        # Dự đoán
        proba = float(pipeline.predict_proba(df)[0][1])  # Chuyển thành float Python
        probability = proba * 100

        risk_level, color = get_risk_assessment(probability)

        # Response – đảm bảo tất cả giá trị đều là Python native types (tránh numpy types)
        return {
            "probability": round(probability, 2),
            "risk_level": risk_level,
            "risk_color": color,
            "advice_list": generate_advice(raw, probability),
            "model_used": model_key.upper(),
            "raw_probability": round(proba, 4),
            "applied_threshold": round(threshold, 3)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý yêu cầu: {str(e)}")

# --- CHẠY SERVER ---
if __name__ == '__main__':
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)