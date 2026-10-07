"""Klasifikasi risiko diabetes dengan algoritma K-Nearest Neighbors (KNN).

Jalankan dari root project dengan:
    uv run python challanges/ch06_knn_risiko_diabetes.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "diabetes_risk_prediction_dataset.csv"
)
TARGET = "Diabetes_Risk"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Membaca dataset dan menghapus kolom ID/leakage dari fitur."""
    df = pd.read_csv(DATASET_PATH)

    leakage_columns = {
        "Patient_ID",
        "Diabetes_Risk_Score",
        "AI_Health_Recommendation",
        "Doctor_Consultation_Needed",
    }
    feature_columns = [
        column for column in df.columns if column not in leakage_columns | {TARGET}
    ]

    return df[feature_columns], df[TARGET]


def build_model(features: pd.DataFrame) -> Pipeline:
    """Membuat pipeline preprocessing dan model KNN."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    # KNN memakai jarak antar data, jadi fitur numerik wajib diskalakan.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=47,
        stratify=target,
    )

    model = build_model(features)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Dataset       : {DATASET_PATH.name}")
    print(f"Jumlah data   : {len(features)}")
    print(f"Jumlah fitur  : {features.shape[1]}")
    print("Algoritma     : KNN (k=7, weights=distance)")
    print(f"Akurasi test  : {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    labels = model.classes_
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("Confusion matrix (baris=aktual, kolom=prediksi):")
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    sample = x_test.iloc[[0]]
    sample_prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    print(f"\nContoh prediksi: {sample_prediction}")
    print("Probabilitas kelas:")
    for label, probability in zip(model.classes_, probabilities):
        print(f"  {label}: {probability:.2%}")


if __name__ == "__main__":
    main()
