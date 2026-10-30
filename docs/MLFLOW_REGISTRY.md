# Evidências do MLflow Tracking e Model Registry

## Experimentos

O pipeline registra uma run para cada candidato (`logistic_regression`, `random_forest` e
`mlp_classifier`) no experimento `purchase-propensity`. As runs armazenam o nome e os
hiperparâmetros do modelo, as médias e desvios da validação cruzada e, somente para o campeão,
as métricas do holdout e o threshold de decisão. A comparação tabular também é exportada em
`reports/model_comparison.csv`.

![Runs de treinamento no experimento purchase-propensity](images/mlflow-experiments.png)

Na execução validada em 27/10/2026, a run campeã registrou:

- `model=random_forest`, `n_estimators=300`, `max_depth=12`,
  `min_samples_leaf=3` e `class_weight=balanced`;
- `cv_pr_auc_mean=0,7222` e `test_pr_auc=0,7066`;
- `test_recall=0,8037` e `test_f1=0,6463`;
- `decision_threshold=0,5`;
- tag `stage=champion`.

Após a ampliação da comparação, o experimento passou a registrar três candidatos. O
`MLPClassifier` obteve `cv_pr_auc_mean=0,7083`, ficando entre a Random Forest (`0,7222`) e a
Regressão Logística (`0,6571`). A seleção continua sendo feita somente pela PR-AUC média da
validação cruzada, antes de consultar o holdout.

## Model Registry

O artefato selecionado é registrado como `purchase-propensity-classifier`. A versão validada
recebe a tag `status=validated` e o alias mutável `champion`, permitindo promover outra versão
sem mudar o identificador usado pelos consumidores.

![Modelo registrado com o alias champion](images/mlflow-model-registry.png)

## Como conferir localmente

Depois de `uv run dvc repro`, execute:

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Acesse <http://127.0.0.1:5000>, entre em **Experiments > purchase-propensity > Training
runs** e depois em **Models > purchase-propensity-classifier**.
