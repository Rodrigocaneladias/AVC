from sklearn.linear_model import LogisticRegression
from dataset import get_train_test_data
from util import find_best_threshold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score,
    f1_score,
    recall_score,
    classification_report
)

# -  pega os dados de treino e teste
X_train, X_test, y_train, y_test = get_train_test_data()    

# - estancia e treina o modelo
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Validação cruzada antes de avaliar no teste
scores_cv = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"AUC médio (CV 5-fold): {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")

# - faz as previsões
proba = model.predict_proba(X_test)[:, 1]
treshold = find_best_threshold(y_test, proba)
predictions = (proba >= treshold ).astype(int)

print(f"Acurácia:  {accuracy_score(y_test, predictions):.4f}")
print(f"AUC:       {roc_auc_score(y_test, proba):.4f}")
print(f"F1-Score:  {f1_score(y_test, predictions):.4f}")
print(f"Recall:    {recall_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions, target_names=['normal', 'stroke']))