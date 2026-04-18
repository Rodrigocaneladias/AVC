from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {acc}")
    print(f"Recall: {rec}")
    print(f"F1-score: {f1}")
    print("\nRelatório completo:\n")
    print(classification_report(y_true, y_pred))

def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Matriz de Confusão - {name}")
    plt.savefig(f"confusion_{name}.png")
    plt.close()