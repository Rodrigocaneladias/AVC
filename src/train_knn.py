from sklearn.neighbors import KNeighborsClassifier
from preprocessing import load_data, preprocess
from utils import evaluate_model, plot_confusion

df = load_data("../dataset/stroke_risk_dataset.csv")

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

# validação
y_val_pred = model.predict(X_val)
print("\n=== KNN (Validação) ===")
evaluate_model(y_val, y_val_pred)

# teste
y_test_pred = model.predict(X_test)
print("\n=== KNN (Teste) ===")
evaluate_model(y_test, y_test_pred)

plot_confusion(y_test, y_test_pred, "knn")