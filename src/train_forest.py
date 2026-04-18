from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_data, preprocess
from utils import evaluate_model, plot_confusion
import matplotlib.pyplot as plt

df = load_data("../dataset/stroke_risk_dataset.csv")

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# validação
y_val_pred = model.predict(X_val)
print("\n=== Random Forest (Validação) ===")
evaluate_model(y_val, y_val_pred)

# teste
y_test_pred = model.predict(X_test)
print("\n=== Random Forest (Teste) ===")
evaluate_model(y_test, y_test_pred)

plot_confusion(y_test, y_test_pred, "forest")

# 🔥 feature importance
importances = model.feature_importances_

plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Importância das Variáveis")
plt.xlabel("Features")
plt.ylabel("Importância")

plt.savefig("feature_importance.png")
plt.close()