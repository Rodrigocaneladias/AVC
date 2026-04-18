from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_data, preprocess
from utils import evaluate_model, plot_confusion

import matplotlib.pyplot as plt

df = load_data("../dataset/stroke_risk_dataset.csv")

X_train, X_test, y_train, y_test = preprocess(df)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Random Forest ===")
evaluate_model(y_test, y_pred)

# matriz de confusão
plot_confusion(y_test, y_pred, "Random Forest")

# 🔥 feature importance
importances = model.feature_importances_

plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Importância das Variáveis")
plt.xlabel("Features")
plt.ylabel("Importância")
plt.show()