from sklearn.linear_model import LinearRegression
from dataset import get_train_test_data
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)

# -  pega os dados de treino e teste
X_train, X_test, y_train, y_test = get_train_test_data()

# - treina o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Validação cruzada antes de avaliar no teste
scores_cv = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"AUC médio (CV 5-fold): {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")

# - faz as previsões
predictions = model.predict(X_test)   

# - avalia o modelo
#MSE: média dos quadrados dos erros (quanto menor, melhor)
mse = mean_squared_error(y_test, predictions)
lin_mse = np.sqrt(mse)  # Raiz do MSE para interpretar na mesma escala dos dados

# MAE: média dos erros absolutos (quanto menor, melhor)
mae = mean_absolute_error(y_test, predictions) 

# R2: proporção da variância dos dados explicada pelo modelo (quanto mais próximo de 1, melhor)
r2 = r2_score(y_test, predictions)

#MAPE: média dos erros percentuais absolutos (quanto menor, melhor)
def calculate_mape(labels, predict):
     print (predict)
     error = np.abs(labels - predict)
     relative_error = error / np.abs(labels)
     mape = np.mean(relative_error) * 100
     return mape 
mape = calculate_mape(y_test, predictions)

print('predctions: ', model.predict(X_test))
print('model score: ', model.score(X_test, y_test))
print('MSE: ', lin_mse)
print('MAE: ', mae)
print('R2: ', r2)
