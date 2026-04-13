import numpy as np
from sklearn.metrics import precision_recall_curve

def find_best_threshold(y_true, proba):

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def classify_risk(proba, threshold):
    """
    Classifica o risco usando o threshold ótimo como referência.
    - alto:  probabilidade >= threshold (o modelo classifica como AVC)
    - medio: zona de atenção — abaixo do threshold mas acima da metade dele
    - baixo: probabilidade baixa, longe do ponto de corte
    """
    if proba >= threshold:
        return "alto"
    elif proba >= threshold / 2:
        return "medio"
    else:
        return "baixo"
