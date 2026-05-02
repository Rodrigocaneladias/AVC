# Previsão de Risco de AVC — Tech Challenge Fase 1
**PosTech FIAP — IA para Devs**

## Descrição do Projeto

Sistema inteligente de suporte ao diagnóstico para previsão de risco de AVC (Acidente Vascular Cerebral), desenvolvido como parte do Tech Challenge Fase 1 da pós-graduação PosTech FIAP.

O AVC é uma das principais causas de morte e incapacidade no mundo. Este projeto aplica técnicas de Machine Learning supervisionado para classificar pacientes em risco, auxiliando equipes médicas na triagem e apoio à decisão clínica.

## Objetivo

Construir modelos preditivos de classificação binária (AVC: sim/não) utilizando o Stroke Prediction Dataset, com foco em maximizar o **Recall** — minimizando falsos negativos, que representam risco clínico grave.

## Estrutura do Projeto

```
stroke-project/
├── data/
│   └── healthcare-dataset-stroke-data.csv    # Dataset
├── notebooks/
│   └── stroke_prediction.ipynb               # Notebook principal
├── outputs/
│   ├── 01_distribuicao_target.png
│   ├── 02_distribuicao_numericas.png
│   ├── 03_categoricas_vs_stroke.png
│   ├── 04_correlacao.png
│   ├── 05_comparacao_modelos.png
│   ├── 06_curvas_roc.png
│   ├── 07_matrizes_confusao.png
│   ├── 08_feature_importance_rf.png
│   ├── 09_feature_importance_comparacao.png
│   ├── 10_threshold_analysis.png
│   └── metrics.json
├── Dockerfile
├── requirements.txt
└── README.md
```

## Dataset

**Stroke Prediction Dataset** — Kaggle/UCI  
- 5.110 registros, 11 features + 1 variável alvo
- Prevalência de AVC: ~4,87% (dataset desbalanceado)
- Download: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

### Features
| Feature | Tipo | Descrição |
|---|---|---|
| gender | Categórica | Gênero do paciente |
| age | Numérica | Idade |
| hypertension | Binária | Hipertensão (0/1) |
| heart_disease | Binária | Doença cardíaca (0/1) |
| ever_married | Categórica | Estado civil |
| work_type | Categórica | Tipo de trabalho |
| Residence_type | Categórica | Tipo de residência |
| avg_glucose_level | Numérica | Glicose média no sangue |
| bmi | Numérica | Índice de Massa Corporal |
| smoking_status | Categórica | Status de tabagismo |
| **stroke** | **Binária** | **Alvo: AVC (0/1)** |

## Modelos Utilizados

1. **Regressão Logística** — Baseline interpretável
2. **Random Forest** — Modelo principal (melhor equilíbrio recall/AUC)
3. **Gradient Boosting** — Modelo desafiante

## Como Executar

### Opção 1: Docker (recomendado)

```bash
# Build da imagem
docker build -t stroke-prediction .

# Executar o notebook
docker run -p 8888:8888 stroke-prediction
```

Acesse: `http://localhost:8888`

### Opção 2: Ambiente local


```bash
# Utilizar python 3.11
py -3.11 -m venv venv

# Instalar dependências
pip install -r requirements.txt

# Iniciar Jupyter (Ter o Jupyter instalado previamente)
jupyter notebook notebooks/AVC_prediction.ipynb
```

## Requisitos

Ver `requirements.txt`. Principais:
- Python 3.11
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- jupyter

## Resultados Principais

| Modelo | Recall | F1-Score | ROC-AUC |
|---|---|---|---|
| Regressão Logística | 0.89 | 0.26 | 0.89 |
| Random Forest | 0.49 | 0.28 | 0.86 |
| Gradient Boosting | 0.57 | 0.30 | 0.86 |

**Modelo final:** Random Forest com threshold=0.3  
**Recall no teste:** 0.81 | **ROC-AUC:** 0.82

## Equipe

PosTech FIAP — IA para Devs — Fase 1
