# Arquitetura e decisões técnicas

## Separação de responsabilidades

- `data/download.py`: aquisição, extração e validação de checksum;
- `data/prepare.py`: validação de esquema e split estratificado;
- `modeling/candidates.py`: pré-processamento e candidatos Scikit-Learn;
- `modeling/metrics.py`: métricas de classificação desbalanceada;
- `modeling/registry.py`: registro e alias de governança;
- `training/train.py`: orquestração do experimento.

As funções de domínio são pequenas, tipadas e independentes da CLI. Os comandos `main` apenas convertem argumentos em chamadas para essas funções.

## Decisões

### PR-AUC como critério

Somente cerca de 15,5% das sessões terminam em compra. A PR-AUC enfatiza a qualidade da classe positiva e evita que a predominância de sessões sem compra torne a avaliação excessivamente otimista.

### Holdout isolado

Regressão Logística e Random Forest são comparadas por validação cruzada estratificada de cinco folds somente no treino. O teste é consultado uma vez, após a escolha do campeão.

### Duplicidades aparentes

Há linhas com atributos iguais, mas não existe identificador de sessão. Elas foram mantidas: igualdade de atributos não prova que sejam a mesma observação.

### DVC sem credencial remota

O pipeline baixa a fonte pública e valida seu SHA-256. Os dados e modelos são outputs do DVC, os hashes ficam em `dvc.lock` e o cache permanece local. Isso mantém a reprodução pública sem publicar credenciais de storage.

### Registry baseado em SQLite

O tracking URI padrão é `sqlite:///mlflow.db`, pois o Model Registry do MLflow precisa de backend baseado em banco. Cada execução registra parâmetros, métricas, assinatura, exemplo de entrada e artefato; somente o vencedor recebe o alias `champion`.

