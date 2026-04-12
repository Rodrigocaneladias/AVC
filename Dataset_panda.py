import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split   

# importa o dataset
ds = pd.read_csv("./dataset/healthcare-dataset-stroke-data.csv")

#normalizando os dados 

# 1- retirando a coluna ID
ds = ds.drop(columns=['id'])

# 2- Gender
# Retirando a unica ocorrencia de gender que esta com valor outros
# reset index renumera as linhas apos a remoçao
ds = ds[ds['gender'] != 'Other'].reset_index(drop=True)

# 3- retirando valores nulos de bmi
ds['bmi'] = ds['bmi'].fillna(ds['bmi'].median())

# 4- tratando valores binarios
ds['ever_married'] = ds['ever_married'].map({'Yes': 1, 'No': 0})
ds['Residence_type'] = ds['Residence_type'].map({'Urban': 1, 'Rural': 0})
ds['gender'] = ds['gender'].map({'Male': 1, 'Female': 0})

# 5- tratando o smoking com valores ordinais
ds['smoking_status'] = ds['smoking_status'].map(
    {'never smoked': 0, 
     'Unknown': 0, 
     'formerly smoked': 1, 
     'smokes': 2
    })
    #OBS: poderiamos utilizar o OrdinalEncoder do sklearn, mas como temos apenas 4 categorias, achei mais simples fazer a substituição manualmente. 

# 6- Tratando work_type por aparentemente nao ter nada a ver com o avc
ds = pd.get_dummies(ds, columns=['work_type'], drop_first=True)

# 7- Tratando as novas colunas criadas com o get_dummies
ds['work_type_Private'] = ds['work_type_Private'].map({True: 1, False: 0})
ds['work_type_Self-employed'] = ds['work_type_Self-employed'].map({True: 1, False: 0})
ds['work_type_children'] = ds['work_type_children'].map({True: 1, False: 0})
ds['work_type_Never_worked'] = ds['work_type_Never_worked'].map({True: 1, False: 0})

ds.info()

#Procurando correlaçao
corr_matrix = ds.corr()
print(corr_matrix['stroke'].sort_values(ascending=False))

# As 6 maiores correlações são:
#  - Idade com 24%
#  - Doença cardiaca com 13%
#. - Diabetes com 13%
#  - Hipertensão com 12%
#  - Fumar com 11%
#  - Casado com 10%


