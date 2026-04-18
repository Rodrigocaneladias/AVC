import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    print("Colunas originais:", df.columns)

    # remover coluna que causa data leakage
    df = df.drop(columns=['Stroke Risk (%)'])

    # separar X e y
    y = df['At Risk (Binary)']
    X = df.drop(columns=['At Risk (Binary)'])

    print("Colunas usadas no modelo:", X.columns)

    # normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 🔥 divisão em treino + teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 divisão em treino + validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test