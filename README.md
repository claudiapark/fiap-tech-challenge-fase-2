# Purchase Propensity MLOps Pipeline

[![CI](https://github.com/claudiapark/fiap-tech-challenge-fase-2/actions/workflows/ci.yml/badge.svg)](https://github.com/claudiapark/fiap-tech-challenge-fase-2/actions/workflows/ci.yml)

Tech Challenge - Fase 2 da FIAP Pós-Tech Machine Learning Engineering. O projeto prevê se uma sessão de e-commerce terminará em compra e demonstra um ciclo de Engenharia de Machine Learning com Clean Code, `uv`, DVC, MLflow Tracking, Model Registry, Docker, testes e CI.

## Problema de negócio

Uma empresa de e-commerce precisa priorizar sessões com maior propensão de compra. O alvo `Revenue` indica se a sessão terminou em transação. O modelo apoia decisões como segmentação e abordagem comercial; não deve tomar decisões automáticas sobre pessoas.

## Dataset

Foi utilizado o [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), disponibilizado pela UCI Machine Learning Repository sob licença CC BY 4.0 ([DOI 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)).

- 12.330 sessões;
- 17 atributos comportamentais e contextuais;
- alvo binário `Revenue`;
- 15,47% de compras no treino e 15,49% no teste;
- arquivo validado por SHA-256 antes do uso.

## Arquitetura

```mermaid
flowchart LR
    UCI["UCI Dataset"] --> D["Download + SHA-256"]
    D --> R["Dados brutos"]
    R --> P["Split estratificado"]
    P --> T["Treino e validação cruzada"]
    T --> M["MLflow Tracking"]
    T --> E["Teste holdout"]
    M --> G["Model Registry"]
    G --> A["Alias champion"]
```

O DVC coordena `download -> prepare -> train`. As transformações ficam dentro das pipelines Scikit-Learn para evitar vazamento entre validação e treino.

## Resultados

A classe positiva é minoritária; por isso, **PR-AUC de validação cruzada** foi definida como critério de seleção. Os candidatos Linear (Regressão Logística), Ensemble (Random Forest) e Rede Neural (`MLPClassifier`) são comparados com 5 folds estratificados. O teste permanece isolado até a escolha do campeão.

| Modelo | CV PR-AUC | CV ROC-AUC | CV Precision | CV Recall | CV F1 |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **0,7222** | **0,9275** | **0,5597** | **0,8329** | **0,6691** |
| MLP Classifier | 0,7083 | 0,9161 | 0,7115 | 0,5754 | 0,6357 |
| Regressão Logística | 0,6571 | 0,9044 | 0,5173 | 0,7647 | 0,6169 |

A tabela completa (média e desvio) está em [`reports/model_comparison.csv`](reports/model_comparison.csv). O modelo campeão é salvo em `models/purchase_propensity.joblib`.

Métricas finais da Random Forest no holdout:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 0,8637 | 0,5405 | 0,8037 | 0,6463 | 0,9190 | 0,7066 |

Matriz de confusão com threshold `0,5`:

| Real \ Predito | Não compra | Compra |
|---|---:|---:|
| Não compra | 1.823 | 261 |
| Compra | 75 | 307 |

Os 261 falsos positivos podem gerar abordagens ou incentivos sem conversão; os 75 falsos
negativos representam compras reais que deixariam de ser priorizadas. O threshold `0,5` é um
baseline explícito em `params.yaml`, ainda não otimizado por uma função de custo do negócio.

A Random Forest é registrada como `purchase-propensity-classifier` e recebe o alias `champion`. Os valores completos estão em [`reports/metrics.json`](reports/metrics.json).

## Estrutura

```text
.
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MLFLOW_REGISTRY.md
│   ├── MODEL_CARD.md
│   ├── REPRODUCIBILITY.md
│   └── VIDEO_STAR.md
├── models/
├── reports/
│   ├── metrics.json
│   └── model_comparison.csv
├── src/purchase_propensity/
│   ├── data/
│   ├── modeling/
│   └── training/
├── tests/
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── pyproject.toml
└── uv.lock
```

## Execução local

Pré-requisitos: Git, Python 3.12 e [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/claudiapark/fiap-tech-challenge-fase-2.git
cd fiap-tech-challenge-fase-2
python -m pip install uv
uv sync --locked
cp .env.example .env
uv run dvc repro
```

O dataset não precisa ser baixado manualmente. O estágio `download` obtém a versão oficial da UCI, confere o checksum fixado em `params.yaml` e deixa o DVC registrar os hashes em `dvc.lock`.

Comandos úteis:

```bash
uv run dvc dag
uv run dvc metrics show
uv run dvc repro
```

## MLflow Tracking e Model Registry

O pipeline usa SQLite como backend local porque o Model Registry requer um backend baseado em banco de dados.

Depois de executar `dvc repro`, inicie a interface:

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Acesse <http://127.0.0.1:5000>. Em **Experiments**, compare as runs; em **Models**, abra `purchase-propensity-classifier` e confira o alias `champion`.

### Evidências da execução

![Runs do experimento purchase-propensity](docs/images/mlflow-experiments.png)

![Model Registry com alias champion](docs/images/mlflow-model-registry.png)

Os parâmetros, métricas e passos de conferência estão documentados em
[`docs/MLFLOW_REGISTRY.md`](docs/MLFLOW_REGISTRY.md).

## Docker

```bash
docker build -t purchase-propensity:latest .
docker run --rm -p 5000:5000 purchase-propensity:latest
```

O contêiner reproduz o pipeline, inicia o MLflow em <http://127.0.0.1:5000> e expõe
um `HEALTHCHECK`. A CI constrói a imagem, inicia o contêiner e aguarda o estado
`healthy`, validando não apenas o build, mas também a execução.

## Qualidade e testes

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

A CI verifica sintaxe, formatação, lint, 26 testes com cobertura mínima de 80%, `dvc repro`,
quality gate do modelo, build do Docker e um smoke test do contêiner em pushes e pull
requests. Execuções concorrentes da mesma branch são canceladas para economizar tempo de CI.

O quality gate bloqueia o registro do campeão quando uma métrica obrigatória fica abaixo de
`quality_gate.minimum_metric_score`, atualmente `0,50` em `params.yaml`.

## Configuração

Variáveis locais ficam no `.env`, nunca no Git. Os nomes e valores seguros estão em `.env.example`:

| Variável | Finalidade |
|---|---|
| `MLFLOW_TRACKING_URI` | backend de tracking e registry |
| `MLFLOW_EXPERIMENT_NAME` | experimento que agrupa as runs |
| `MLFLOW_REGISTERED_MODEL` | nome governado do modelo |
| `MLFLOW_MODEL_ALIAS` | alias da versão campeã |

## Limitações

- o dataset representa um recorte histórico e pode não refletir o comportamento atual;
- não há dimensão temporal suficiente para uma validação fora do tempo;
- o threshold padrão de 0,5 não foi otimizado por custos de negócio;
- as probabilidades ainda não foram calibradas;
- `PageValues` é muito informativa, mas sua disponibilidade precisa ser confirmada no instante real da decisão;
- linhas com atributos idênticos foram mantidas porque a base não possui identificador que prove duplicidade de sessão;
- o deploy em nuvem é opcional no desafio e não faz parte desta entrega.

Consulte o [Model Card](docs/MODEL_CARD.md) para detalhes de uso responsável.

## Documentação da entrega

- [Arquitetura e decisões](docs/ARCHITECTURE.md)
- [Model Card](docs/MODEL_CARD.md)
- [Relatório de reprodutibilidade](docs/REPRODUCIBILITY.md)
- [Evidências do MLflow e Model Registry](docs/MLFLOW_REGISTRY.md)
- [Roteiro do vídeo STAR](docs/VIDEO_STAR.md)

## Vídeo STAR

O vídeo de até cinco minutos é parte obrigatória da entrega. O roteiro completo,
com comandos, telas e divisão por tempo, está em [`docs/VIDEO_STAR.md`](docs/VIDEO_STAR.md).

> **Status:** gravação e publicação pendentes. Após publicar como **não listado**,
> substitua este aviso pelo link final e confirme que ele abre em uma janela anônima.

## Equipe

| GitHub |
|---|
| [@claudiapark](https://github.com/claudiapark) |
| [@SamuelCambui](https://github.com/SamuelCambui) |
| [@LincolnLN](https://github.com/LincolnLN) |
| [@Leoosantoszl](https://github.com/Leoosantoszl) |
| [@Lucaskjkk](https://github.com/Lucaskjkk) |

As contribuições devem ser feitas em branches individuais e integradas por Pull
Request, mantendo no histórico a responsabilidade real de cada integrante.

## Checklist de entrega

- [x] ambiente reproduzível com `uv` e lock file;
- [x] pipeline DVC `download -> prepare -> train`;
- [x] MLflow Tracking e Model Registry com alias `champion`;
- [x] Docker com verificação de saúde;
- [x] lint, testes, cobertura e pipeline automatizados na CI;
- [x] quality gate antes da promoção do modelo;
- [x] README, arquitetura e Model Card;
- [x] reprodução validada em clone limpo e evidências do MLflow;
- [ ] contribuições reais dos integrantes registradas por Pull Requests;
- [ ] vídeo STAR publicado e link adicionado acima;
- [ ] link do repositório e do vídeo enviados na plataforma da FIAP.
