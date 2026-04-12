import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Carregar o dataset
ds = pd.read_csv("./dataset/healthcare-dataset-stroke-data.csv")

# 2. Pré-processamento inicial (fora do ColumnTransformer, pois são remoções ou substituições globais)
# Tratar 'Unknown' em smoking_status
ds['smoking_status'] = ds['smoking_status'].replace('Unknown', 'never smoked')

# Remover a linha com 'Other' em gender
ds = ds.drop(ds[ds['gender'] == 'Other'].index).reset_index(drop=True)

# Remover a coluna 'id'
ds = ds.drop(columns=['id']).reset_index(drop=True)

# 3. Definir o ColumnTransformer para as transformações específicas
transformer = ColumnTransformer([
    ('bmi_imputer', SimpleImputer(strategy='median'), ['bmi']), # Imputação de BMI
    ('binary_categorical_encoder', OrdinalEncoder(), ['ever_married', 'Residence_type', 'gender']), # Categorias binárias
    ('smoking_encoder', OrdinalEncoder(categories=[['never smoked', 'formerly smoked', 'smokes']]), ['smoking_status']), # Status de fumo ordinal
    ('work_type_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), ['work_type']) # One-Hot Encoding para work_type
], remainder='passthrough') # Manter as colunas não transformadas (como 'age', 'hypertension', 'stroke', etc.)

# 4. Criar o Pipeline, incluindo o ColumnTransformer
pipeline = Pipeline(steps=[('preprocessor', transformer),
                           ('Standard_Scaler', StandardScaler())
                           ])

# 5. Aplicar o pipeline ao dataset original
ds_processed_pipeline = pipeline.fit_transform(ds)

# 6. Obter os nomes das colunas transformadas e criar um DataFrame
pipeline_feature_names = pipeline.get_feature_names_out()
ds_pipeline_df = pd.DataFrame(ds_processed_pipeline, columns=pipeline_feature_names)

# Renomear a coluna 'remainder__stroke' de volta para 'stroke', se existir
# (a coluna 'stroke' é passada por 'remainder__passthrough' e terá esse prefixo)
if 'remainder__stroke' in ds_pipeline_df.columns:
    ds_pipeline_df.rename(columns={'remainder__stroke': 'stroke'}, inplace=True)

# 7. Calcular a matriz de correlação com a coluna 'stroke'
corr_matrix_pipeline = ds_pipeline_df.corr()

# 8. Exibir as correlações com a coluna 'stroke' ordenadas
print(corr_matrix_pipeline['stroke'].sort_values(ascending=False))

#--------------------------------------------------------------------------------
# Separar as features (X) da variável alvo (y)
X = ds_pipeline_df.drop('stroke', axis=1)
y = ds_pipeline_df['stroke']

# Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
# Usar stratify=y para manter a proporção da classe 'stroke' em ambos os conjuntos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Exibir as dimensões dos conjuntos resultantes
print("Dimensões do X_train:", X_train.shape)
print("Dimensões do X_test:", X_test.shape)
print("Dimensões do y_train:", y_train.shape)
print(
"Dimensões do y_test:", y_test.shape)

# Exibir a proporção da variável alvo em cada conjunto para verificar a estratificação
print("\nProporção de 'stroke' em y_train:\n", y_train.value_counts(normalize=True))
print("\nProporção de 'stroke' em y_test:\n", y_test.value_counts(normalize=True))


def get_train_test_data():
    return X_train, X_test, y_train, y_test