from sklearn.tree import DecisionTreeClassifier
from preprocessing import load_data, preprocess
from utils import evaluate_model, plot_confusion

df = load_data("../dataset/stroke_risk_dataset.csv")

X_train, X_test, y_train, y_test = preprocess(df)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Decision Tree ===")
evaluate_model(y_test, y_pred)

# gráfico
plot_confusion(y_test, y_pred, "Decision Tree")