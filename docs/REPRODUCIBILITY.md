# Relatório de reprodutibilidade

## Objetivo

Comprovar que uma pessoa sem o ambiente de desenvolvimento original consegue clonar o
repositório, instalar exatamente as dependências travadas e reconstruir os dados, o modelo e
as métricas por meio do DVC.

## Teste realizado

- **Data:** 27/10/2026;
- **Sistema:** macOS ARM64;
- **Python:** 3.12.14, provisionado pelo `uv`;
- **uv:** 0.13.6;
- **DVC:** 3.67.1;
- **origem:** clone limpo da branch, sem reutilizar `.venv`, cache DVC ou arquivos gerados do
  diretório de desenvolvimento.

Comandos executados:

```bash
git clone --branch evidencias-reproducibilidade-modelo \
  https://github.com/claudiapark/fiap-tech-challenge-fase-2.git
cd fiap-tech-challenge-fase-2
uv sync --locked
uv run dvc repro
uv run dvc metrics show
uv run pytest
```

O executável `uv` precisa estar disponível no `PATH`, pois cada estágio do `dvc.yaml` também
executa seus comandos com `uv run`. Ao instalar pelo instalador oficial, abra um novo terminal
ou aplique a instrução de atualização do `PATH` exibida pelo instalador antes de rodar o DVC.

## Resultado

O teste terminou com sucesso e produziu, a partir do zero:

- dataset bruto obtido da UCI e validado pelo SHA-256 fixado em `params.yaml`;
- divisões estratificadas de treino e teste;
- comparação entre Regressão Logística e Random Forest;
- modelo campeão em `models/purchase_propensity.joblib`;
- métricas versionadas em `reports/metrics.json`;
- runs no experimento `purchase-propensity`;
- modelo `purchase-propensity-classifier` registrado com o alias `champion`;
- 26 testes aprovados e cobertura total de 80,53%.

A Random Forest foi novamente selecionada por `cv_pr_auc_mean=0,7222`. No holdout, o
resultado reproduzido foi `PR-AUC=0,7066`, `ROC-AUC=0,9190`, `recall=0,8037` e `F1=0,6463`.
Os números completos e hashes dos outputs permanecem em `reports/metrics.json` e `dvc.lock`.

## Escopo da garantia

O lock file fixa as versões das dependências Python, e o checksum protege a versão da fonte de
dados. Pequenas diferenças de ponto flutuante ainda podem ocorrer em outra arquitetura ou
implementação de bibliotecas numéricas. O teste não representa garantia de desempenho futuro
em produção; drift e mudanças no comportamento dos usuários exigem monitoramento e novo ciclo
de validação.
