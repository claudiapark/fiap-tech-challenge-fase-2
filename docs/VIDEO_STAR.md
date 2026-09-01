# Roteiro do vídeo STAR - até 5 minutos

## Preparação

1. Execute `uv run dvc repro`.
2. Abra o GitHub no README.
3. Execute `uv run dvc dag` e `uv run dvc metrics show`.
4. Inicie o MLflow conforme o README e abra <http://127.0.0.1:5000>.
5. Deixe abertas as telas **Experiments** e **Models**.
6. Construa a imagem Docker antes da gravação para não esperar durante o vídeo.

## 0:00-0:40 - Situation

> Uma empresa de e-commerce precisa identificar sessões com maior propensão de compra para apoiar ações comerciais. Utilizamos uma base pública da UCI com 12.330 sessões e 17 atributos comportamentais. Apenas cerca de 15,5% das sessões terminam em compra, portanto a classe positiva é desbalanceada.

## 0:40-1:05 - Task

> Nossa tarefa foi construir um pipeline completo de Engenharia de Machine Learning: código limpo e tipado, dependências reproduzíveis, dados e estágios versionados com DVC, treinamento rastreado no MLflow, melhor modelo no Model Registry e execução containerizada com Docker.

## 1:05-3:35 - Action

Enquanto fala, mostre a estrutura, `dvc dag`, as runs e o Registry.

> O download parte da fonte oficial e valida o arquivo por SHA-256. Em seguida, realizamos um split estratificado de 80% para treino e 20% para teste, com semente 42. O pré-processamento permanece dentro das pipelines para evitar vazamento de dados.
>
> Comparamos Regressão Logística e Random Forest em cinco folds estratificados. Como somente 15,5% das sessões resultam em compra, escolhemos PR-AUC como métrica principal. A Regressão Logística obteve PR-AUC média de 0,6571 e a Random Forest, 0,7222. O teste ficou isolado até essa escolha.
>
> Cada candidato gera uma run no MLflow com parâmetros, métricas, assinatura, exemplo de entrada e artefato. A Random Forest vencedora foi registrada como `purchase-propensity-classifier` e recebeu o alias `champion`. O DVC conecta os estágios download, prepare e train, enquanto o Docker reproduz o ambiente e inicia a interface do MLflow.

## 3:35-4:35 - Result

> No teste final, a Random Forest atingiu ROC-AUC de 0,9190, PR-AUC de 0,7066, recall de 0,8037 e F1 de 0,6463. A automação também valida o código com lint, 20 testes, cobertura mínima de 80%, reprodução do DVC e build do Docker.

## 4:35-4:55 - Fechamento

> Como limitações, o threshold ainda não considera custos reais, as probabilidades não foram calibradas e não há validação temporal. O deploy em nuvem é opcional e ficou fora do escopo. O resultado é um pipeline local reproduzível, rastreável e pronto para evolução. Obrigado.

## Checklist

- [ ] duração inferior a 5 minutos;
- [ ] STAR identificável;
- [ ] DVC, Docker, MLflow Tracking e Registry demonstrados;
- [ ] métricas e campeão apresentados;
- [ ] limitações mencionadas;
- [ ] nenhum `.env`, token ou dado pessoal visível.

