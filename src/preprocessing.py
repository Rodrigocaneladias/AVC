import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    print("Colunas originais:", df.columns)

    # 🔥 REMOVER coluna que causa vazamento de dados
    if 'Stroke Risk (%)' in df.columns:
        df = df.drop(columns=['Stroke Risk (%)'])

    # 🔹 Separar target
    y = df['At Risk (Binary)']

    # 🔹 Separar features (REMOVER target)
    X = df.drop(columns=['At Risk (Binary)'])

    print("Colunas usadas no modelo:", X.columns)

    # 🔹 Renomear colunas (remove espaços e caracteres especiais)
    X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    # 🔹 Normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 🔹 Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test