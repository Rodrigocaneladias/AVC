from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from dataset import get_train_test_data
from util import find_best_threshold, classify_risk
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    classification_report
)

X_train, X_test, y_train, y_test = get_train_test_data()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,          # evita overfitting
    min_samples_leaf=5,    # folhas com pelo menos 5 amostras
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Validação cruzada antes de avaliar no teste
scores_cv = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"AUC médio (CV 5-fold): {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")

# Probabilidades e threshold otimizado
proba     = model.predict_proba(X_test)[:, 1]
threshold = find_best_threshold(y_test, proba)
predictions = (proba >= threshold).astype(int)

print(f"Threshold otimizado: {threshold:.4f}")

risk_levels = [classify_risk(p, threshold) for p in proba]

#print("\nRisco classificado para cada paciente:")
#for i, risk in enumerate(risk_levels):
#    print(f"Paciente {i+1}: {risk}")    

print(f"\nAcurácia:  {accuracy_score(y_test, predictions):.4f}")
print(f"AUC:       {roc_auc_score(y_test, proba):.4f}")
print(f"F1-Score:  {f1_score(y_test, predictions):.4f}")
print(f"Recall:    {recall_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions, target_names=['normal', 'stroke']))