import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

from .utils import (
    load_dataset,
    save_model_bundle,
    process_age_category,
    get_bmi_cat_code,
    get_sleep_risk_code,
    EXPECTED_FEATURES
)
from .preprocess import get_preprocessor


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['HeartDisease'] = (
        (df['HadHeartAttack'] == 'Yes') | (df['HadAngina'] == 'Yes')
    ).astype(int)

    df['AgeCategory'] = df['AgeCategory'].apply(process_age_category)
    df['BMI_cat'] = df['BMI'].apply(get_bmi_cat_code)
    df['Sleep_risk'] = df['SleepHours'].apply(get_sleep_risk_code)

    binary_cols = ["PhysicalActivities", "AlcoholDrinkers", "ChestScan", "HadArthritis"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(int)

    return df


def find_optimal_threshold_logreg(model, X_val, y_val):
    """Chiến lược Higher Recall Relaxed như notebook phần Logistic"""
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.71, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "Threshold": t,
            "Recall": recall_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "F1": f1_score(y_val, y_pred, zero_division=0)
        })

    df_res = pd.DataFrame(results)
    f1_max = df_res["F1"].max()
    best_row = df_res[df_res["F1"] == f1_max].iloc[0]
    current_precision = best_row["Precision"]
    current_f1 = best_row["F1"]

    relaxed = df_res[
        (df_res["Precision"] >= 0.85 * current_precision) &
        (df_res["F1"] >= 0.90 * current_f1)
    ]

    if relaxed.empty:
        return best_row["Threshold"]
    else:
        return relaxed.loc[relaxed["Recall"].idxmax(), "Threshold"]


def find_optimal_threshold_rf(model, X_val, y_val):
    """Chiến lược MIN_RECALL + MIN_PRECISION như notebook phần RF"""
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.51, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "Threshold": round(t, 2),
            "Recall": recall_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "F1": f1_score(y_val, y_pred, zero_division=0)
        })

    df_res = pd.DataFrame(results)
    candidates = df_res[
        (df_res["Recall"] >= 0.82) &
        (df_res["Precision"] >= 0.17)
    ]

    if candidates.empty:
        return df_res.loc[df_res["F1"].idxmax(), "Threshold"]
    else:
        return candidates.sort_values(by=["Recall", "F1"], ascending=False).iloc[0]["Threshold"]


def find_optimal_threshold_xgb(model, X_val, y_val):
    """Chiến lược MIN_RECALL + MIN_PRECISION như notebook phần XGB"""
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.51, 0.02)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "Threshold": round(t, 2),
            "Recall": recall_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "F1": f1_score(y_val, y_pred, zero_division=0)
        })

    df_res = pd.DataFrame(results)
    candidates = df_res[
        (df_res["Recall"] >= 0.83) &
        (df_res["Precision"] >= 0.15)
    ]

    if candidates.empty:
        return df_res.loc[df_res["F1"].idxmax(), "Threshold"]
    else:
        return candidates.sort_values(by=["Recall", "F1"], ascending=False).iloc[0]["Threshold"]


def train_system():
    print("=" * 80)
    print("HUẤN LUYỆN HỆ THỐNG SÀNG LỌC NGUY CƠ TIM MẠCH - REPRODUCE NOTEBOOK GỐC")
    print("=" * 80)

    raw_df = load_dataset()
    df = prepare_training_data(raw_df)

    X = df[EXPECTED_FEATURES]
    y = df['HeartDisease']

    # Split đúng như notebook: Train 60%, Val 20%, Test 20%
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    print(f"Split: Train {X_train.shape[0]:,}, Val {X_val.shape[0]:,}, Test {X_test.shape[0]:,}")

    # scale_pos_weight cho XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    preprocessor = get_preprocessor()

    models_config = {
        "logistic_regression_high_recall.pkl": (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            find_optimal_threshold_logreg
        ),
        "random_forest_high_recall.pkl": (
            RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            find_optimal_threshold_rf
        ),
        "xgboost_high_recall.pkl": (
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss', random_state=42, n_jobs=-1
            ),
            find_optimal_threshold_xgb
        )
    }

    results = []

    for filename, (estimator, thresh_func) in models_config.items():
        print(f"\nĐang huấn luyện {filename}...")

        pipeline = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
        pipeline.fit(X_train, y_train)

        optimal_thresh = thresh_func(pipeline, X_val, y_val)
        print(f"   → Threshold high-recall: {optimal_thresh:.3f}")

        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_thresh).astype(int)

        recall = recall_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_test_proba)

        results.append({
            "Model": filename.replace("_high_recall.pkl", "").title().replace("_", " "),
            "Threshold (Val)": round(optimal_thresh, 2),
            "Recall": round(recall, 3),
            "Precision": round(precision, 3),
            "F1-score": round(f1, 3),
            "ROC-AUC": round(auc, 3)
        })

        bundle = {
            "pipeline": pipeline,
            "threshold": optimal_thresh,
            "features": EXPECTED_FEATURES
        }
        save_model_bundle(bundle, filename)

    # In kết quả giống hệt notebook
    print("\n" + "="*80)
    print("TỔNG KẾT HIỆU SUẤT 3 MÔ HÌNH (TEST SET)")
    print("="*80)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=True))
    print("="*80)

if __name__ == "__main__":
    train_system()