import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# 1. Carregar o dataset
ds = pd.read_csv("./dataset/healthcare-dataset-stroke-data.csv")

# 2. Pré-processamento inicial (operações globais que não dependem de treino/teste)
ds['smoking_status'] = ds['smoking_status'].replace('Unknown', 'never smoked')
ds = ds.drop(ds[ds['gender'] == 'Other'].index).reset_index(drop=True)
ds = ds.drop(columns=['id'])

# 3. Separar X e y ANTES de qualquer transformação
# Isso evita data leakage: o pipeline não pode "ver" o teste durante o fit
X = ds.drop(columns=['stroke'])
y = ds['stroke']

# 4. Dividir em treino e teste com os dados ainda brutos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Definir o ColumnTransformer
# Cada tupla: (nome, transformador, lista_de_colunas)
transformer = ColumnTransformer([
    # Imputa bmi com mediana — calculada apenas no treino
    ('bmi_imputer', SimpleImputer(strategy='median'), ['bmi']),

    # Variáveis binárias sem ordem: OneHotEncoder com drop='first'
    # drop='first' remove a coluna redundante (se não é Male é Female)
    ('binary_encoder',
     OneHotEncoder(drop='first', sparse_output=False),
     ['ever_married', 'Residence_type', 'gender']),

    # Status de fumo: tem ordem real (nunca < ex-fumante < fumante)
    # OrdinalEncoder com ordem explícita é correto aqui
    ('smoking_encoder',
     OrdinalEncoder(categories=[['never smoked', 'formerly smoked', 'smokes']]),
     ['smoking_status']),

    # work_type: sem ordem → OneHotEncoder
    ('work_type_encoder',
     OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),
     ['work_type']),

], remainder='passthrough')  # age, hypertension, heart_disease, glucose, etc.

# 6. Pipeline: pré-processamento + normalização
pipeline = Pipeline(steps=[
    ('preprocessor', transformer),
    ('scaler', StandardScaler()),
])

# 7. Fit APENAS no treino, transform nos dois
# fit_transform no treino: aprende as transformações
# transform no teste: aplica as transformações já aprendidas (sem reaprender)
X_train = pipeline.fit_transform(X_train)
X_test  = pipeline.transform(X_test)

# 8. Recuperar nomes das colunas para análise
feature_names = [
    'bmi',
    'ever_married_Yes',
    'Residence_type_Urban',
    'gender_Male',
    'smoking_status',
    'Never_worked',
    'Private',
    'Self-employed',
    'children',
    'age',
    'hypertension',
    'heart_disease',
    'avg_glucose_level'
]
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df  = pd.DataFrame(X_test,  columns=feature_names)

#---------------------------------------------------------------------------------------------------
#PCA
# diminuir a quantidade de colunas sem perder muita informação, ou seja, sem perder muita variância
# nao fez muita diferença ja que nao temos tantas colunas assim.
#guardar os resultados para comparação posterior
def plot_pca_variance():
    results = []
    #dados de exemplo para teste
    x = X_train_df
    #loop de 2 a 10 componentes
    for n in range(2, 13):
        "cria o PCA com n componentes, ajusta aos dados e guarda a variância explicada acumulada"
        pca = PCA(n_components=n)
        #cria as componentes principais e calcula a variância explicada acumulada
        pca.fit(x)
        #Calcula a variança explicada acumulada e guarda o resultado
        explained_variance = np.sum(pca.explained_variance_ratio_)
        results.append((explained_variance)) 

    #Gerando grafico
    plt.figure(figsize=(15, 8))
    plt.plot(range(2, 13), results , marker='o', linestyle='-', color='b')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada %')
    plt.title('PCA - Variância Explicada por Número de Componentes')
    plt.grid(True)
    # adiciona rotulo aos pontos do grafico
    for i, (n_component, explained_variance) in enumerate(zip(range(2, 13), results)):
        plt.text(n_component, explained_variance, f'Component {n_component}', ha='center', va='bottom')
    plt.show()

#---------------------------------------------------------------------------------------------------

df_corr = pd.DataFrame(X_train, columns=feature_names)
df_corr['stroke'] = y_train.reset_index(drop=True)


def print_data_info():
    print("\n--------------------------------------------------")
    print("Correlação com stroke (X_train):")
    print(df_corr.corr()['stroke'].sort_values(ascending=False))
    print("\n5 linhas do dataset:")
    print(df_corr.head())
    print("\nnumero de nulls na tabela:")
    print(df_corr.isnull().sum())

    print("\n--------------------------------------------------")
    print("Dimensões do X_train:", X_train_df.shape)
    print("Dimensões do X_test:",  X_test_df.shape)
    print("\nProporção de stroke em y_train:\n", y_train.value_counts(normalize=True))
    print("\nProporção de stroke em y_test:\n",  y_test.value_counts(normalize=True))
print_data_info()

def get_pipeline():
    return pipeline

def get_train_test_data():
    return X_train_df, X_test_df, y_train, y_test