import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.preprocessing import (
    load_data,
    drop_unused_columns,
    split_features_target,
    get_feature_types,
    build_preprocessor,
)
from src.features import create_risk_count


def main():
    # 1. Carregar e preparar os dados
    df = load_data("dataset/healthcare-dataset-stroke-data.csv")
    df = drop_unused_columns(df)
    df = create_risk_count(df)

    X, y = split_features_target(df)

    # 2. Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 3. Identificar tipos de variáveis e montar o preprocessor
    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 4. Pipeline com pré-processamento + SMOTE + Logistic Regression
    model_pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # 5. Treinar modelo
    model_pipeline.fit(X_train, y_train)

    # 6. Previsões com threshold padrão
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("=== Resultado com threshold padrão (0.5) ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("=== Matriz de confusão ===")
    print(cm)

    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # 7. Encontrar melhor threshold com base no F1-score
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # thresholds tem tamanho menor que precision/recall
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]

    print("\n=== Melhor threshold encontrado ===")
    print("Threshold:", best_threshold)
    print("Precision:", precision[best_idx])
    print("Recall:", recall[best_idx])
    print("F1-score:", f1_scores[best_idx])

    # 8. Previsões com threshold ajustado
    y_pred_adjusted = (y_proba >= best_threshold).astype(int)

    print("\n=== Resultado com threshold ajustado ===")
    print(classification_report(y_test, y_pred_adjusted))

    cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
    print("=== Matriz de confusão com threshold ajustado ===")
    print(cm_adjusted)

    # 9. Plot da matriz de confusão ajustada
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_adjusted, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - Threshold Ajustado")
    plt.show()


if __name__ == "__main__":
    main()