# Model Card - Purchase Propensity Classifier

## Resumo

- **Problema:** classificação binária de sessões de e-commerce;
- **Alvo:** `Revenue`;
- **Campeão:** Random Forest;
- **Critério de seleção:** média da PR-AUC em validação cruzada;
- **Registry:** `purchase-propensity-classifier@champion`;
- **Threshold avaliado:** 0,5.

## Dados

O Online Shoppers Purchasing Intention Dataset contém 12.330 sessões e 17 atributos. A divisão treino/teste é estratificada em 80/20 com `random_state=42`. O treino possui 9.864 linhas e o teste, 2.466.

## Pré-processamento

As transformações são ajustadas dentro da Pipeline Scikit-Learn:

- imputação por mediana e padronização para variáveis numéricas;
- imputação pela moda e One-Hot Encoding para variáveis categóricas;
- categorias desconhecidas são ignoradas na inferência.

## Desempenho

| Métrica de teste | Valor |
|---|---:|
| Accuracy | 0,8637 |
| Precision | 0,5405 |
| Recall | 0,8037 |
| F1 | 0,6463 |
| ROC-AUC | 0,9190 |
| PR-AUC | 0,7066 |

## Uso pretendido

Ordenar ou segmentar sessões para análises e ações comerciais assistidas. A saída é um sinal de apoio e precisa ser combinada com capacidade operacional, custo de abordagem e revisão de negócio.

## Usos inadequados

- decisões que produzam efeitos jurídicos ou financeiros sem revisão humana;
- inferência sobre populações ou períodos não validados;
- interpretação causal das variáveis;
- uso de probabilidades como se estivessem calibradas.

## Limitações e riscos

- prevalência baixa da classe positiva;
- threshold não escolhido por uma função de custo;
- ausência de calibração e validação temporal;
- possível mudança de distribuição entre a base histórica e o e-commerce real;
- disponibilidade de `PageValues` no momento da decisão precisa ser confirmada;
- a base não contém atributos suficientes para uma auditoria ampla de fairness.

## Governança

Parâmetros e métricas são rastreados no MLflow. O campeão é registrado com assinatura e exemplo de entrada, recebe alias `champion` e pode ser substituído sem alterar consumidores que usem o alias. Dados, código, parâmetros e outputs do pipeline são relacionados por Git, `params.yaml`, `dvc.yaml` e `dvc.lock`.

