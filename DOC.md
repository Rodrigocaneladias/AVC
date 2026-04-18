# Documentação Técnica - Projeto AVC

## Objetivo
Construir um modelo de Machine Learning para prever risco de AVC.

## Dataset
Dataset com aproximadamente 70.000 registros contendo sintomas clínicos e idade.

## Pré-processamento
- Remoção da coluna Stroke Risk (%)
- Normalização com StandardScaler
- Split treino/teste (80/20)

## Modelos
- Logistic Regression
- Decision Tree
- Random Forest

## Resultados
Random Forest foi o melhor modelo com alto recall (~0.97).

## Conclusão
O modelo pode ser usado como suporte à decisão clínica, mas não substitui médicos.
