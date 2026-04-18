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


## 📊 Comparação dos Modelos

| Modelo              | Accuracy | Recall | F1-score | Observações |
|--------------------|---------|--------|----------|-------------|
| Logistic Regression | 1.00    | 1.00   | 1.00     | Resultado perfeito (possível simplificação do dataset, não utilizado como modelo final) |
| Decision Tree       | 0.82    | 0.88   | 0.86     | Alto número de erros, especialmente falsos negativos |
| Random Forest       | **0.95**| **0.97** | **0.96** | Melhor desempenho geral, escolhido como modelo principal |
| KNN                 | 0.89    | 0.93   | 0.92     | Bom desempenho, utilizado como modelo de comparação |
