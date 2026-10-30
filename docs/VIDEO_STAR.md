# Roteiro didático do vídeo STAR - até 5 minutos

Este roteiro foi preparado para uma gravação objetiva, com evidências visuais do
pipeline funcionando. Não mostre instalação de dependências, downloads demorados,
arquivos `.env`, tokens ou informações pessoais.

## 1. Preparação antes de gravar

Abra o terminal na raiz do repositório e execute:

```bash
uv sync --locked
uv run dvc repro
uv run dvc dag
uv run dvc metrics show
docker build -t purchase-propensity:demo .
docker run --detach --name purchase-propensity-demo -p 5000:5000 purchase-propensity:demo
docker ps
```

Espere o contêiner ficar saudável:

```bash
docker inspect --format='{{.State.Health.Status}}' purchase-propensity-demo
```

Quando aparecer `healthy`, abra <http://127.0.0.1:5000> e deixe preparadas estas
telas:

1. GitHub com o README aberto;
2. terminal na raiz do projeto;
3. MLflow na aba **Experiments**;
4. MLflow na aba **Models**, com `purchase-propensity-classifier` aberto;
5. GitHub Actions mostrando a CI verde.

Para evitar espera durante a gravação, use `uv run dvc repro` novamente. Como os
artefatos já estarão atualizados, o DVC comprovará a reprodutibilidade rapidamente.

## 2. Ordem exata da gravação

### 0:00-0:35 - Situation

**Mostrar:** título, problema de negócio e seção Dataset do README.

**Falar:**

> Uma empresa de e-commerce precisa identificar sessões com maior propensão de
> compra para direcionar ações comerciais. Utilizamos o Online Shoppers Purchasing
> Intention Dataset da UCI, com 12.330 sessões e 17 atributos comportamentais.
> Somente cerca de 15,5% das sessões terminam em compra, portanto trabalhamos com
> uma classificação binária desbalanceada.

### 0:35-1:00 - Task

**Mostrar:** árvore de diretórios do README.

**Falar:**

> Nossa tarefa foi construir um pipeline completo de Engenharia de Machine Learning:
> código limpo e tipado, ambiente reproduzível com uv, dados e etapas versionados
> com DVC, experimentos rastreados no MLflow, modelo campeão no Registry e execução
> containerizada com Docker.

### 1:00-1:45 - Action: arquitetura e DVC

**Mostrar:** no terminal, executar os comandos abaixo, um de cada vez:

```bash
uv run dvc dag
uv run dvc repro
uv run dvc metrics show
```

**Falar:**

> O DVC coordena três etapas: download, preparação e treinamento. O download parte
> da fonte oficial e valida o arquivo por SHA-256. Depois realizamos um split
> estratificado de 80% para treino e 20% para teste, com semente 42. Como nenhuma
> dependência mudou, o DVC confirma que o pipeline já está reproduzido e atualizado.

### 1:45-2:35 - Action: modelagem

**Mostrar:** tabela de resultados no README ou saída de `dvc metrics show`.

**Falar:**

> O pré-processamento permanece dentro das pipelines do Scikit-Learn para evitar
> vazamento de dados. Comparamos Regressão Logística, Random Forest e MLPClassifier
> usando cinco folds estratificados. Como a classe positiva é minoritária, escolhemos
> PR-AUC como métrica principal. A Random Forest venceu com 0,7222, seguida do MLP
> com 0,7083 e da Regressão Logística com 0,6571. O conjunto de teste ficou isolado
> até a seleção do campeão.

### 2:35-3:25 - Action: MLflow Tracking e Registry

**Mostrar:** MLflow **Experiments**, abrir as runs e depois acessar **Models**.

**Falar:**

> Cada candidato gera uma run no MLflow com parâmetros, métricas, assinatura,
> exemplo de entrada e artefato do modelo. A Random Forest vencedora foi registrada
> como purchase-propensity-classifier e recebeu o alias champion. Assim, o histórico
> dos experimentos e a promoção do modelo permanecem auditáveis.

Na tela **Models**, aponte visualmente:

- nome `purchase-propensity-classifier`;
- versão registrada;
- alias `champion`;
- tags `task` e `status`, se estiverem visíveis.

### 3:25-4:05 - Action: Docker, testes e CI

**Mostrar:** `docker ps` com o contêiner saudável e depois GitHub Actions verde.

```bash
docker ps
docker inspect --format='{{.State.Health.Status}}' purchase-propensity-demo
```

**Falar:**

> O Docker empacota o ambiente, reproduz o pipeline e inicia o servidor MLflow. O
> health check confirma que o serviço está respondendo. Na integração contínua,
> validamos formatação, lint, 27 testes, cobertura de 81,07%, reprodução do DVC,
> construção da imagem e inicialização real do contêiner.

### 4:05-4:40 - Result

**Mostrar:** métricas finais no README.

**Falar:**

> No teste final, a Random Forest atingiu ROC-AUC de 0,9190, PR-AUC de 0,7066,
> recall de 0,8037 e F1 de 0,6463. O resultado é um pipeline local reproduzível,
> rastreável e pronto para receber novos dados e novas versões de modelo.

### 4:40-4:58 - Limitações e fechamento

**Mostrar:** seção Limitações do README ou Model Card.

**Falar:**

> Como limitações, o threshold ainda não considera custos reais de negócio, as
> probabilidades não foram calibradas e não há validação temporal. O deploy em nuvem
> é opcional e ficou fora do escopo. Obrigado.

## 3. Apresentação definida

A gravação pode ser apresentada integralmente por **Claudia Park**, mantendo o projeto como
trabalho em grupo e citando as contribuições técnicas registradas no GitHub. A divisão para a
apresentação solo é:

| Tempo | Parte |
|---|---|
| 0:00-0:55 | Situation e Task |
| 0:55-2:20 | Arquitetura, DVC e reprodutibilidade |
| 2:20-3:35 | Comparação dos modelos, MLflow e Model Registry |
| 3:35-4:40 | Docker, CI e resultados |
| 4:40-4:58 | Limitações e fechamento |

Não é necessário que todos os integrantes apareçam no vídeo; a autoria técnica permanece
demonstrada pelos commits e Pull Requests do repositório.

## 4. Depois de gravar

1. Corte silêncios e esperas, mantendo o vídeo abaixo de cinco minutos.
2. Publique no YouTube como **não listado**.
3. Abra o link em uma janela anônima para validar o acesso.
4. Adicione o link na seção **Vídeo STAR** do README.
5. Faça a alteração em uma branch e abra uma Pull Request.
6. Envie na FIAP o link do repositório e o link do vídeo.
7. Pare e remova o contêiner local:

```bash
docker rm --force purchase-propensity-demo
```

## Checklist de gravação

- [ ] duração inferior a cinco minutos;
- [ ] STAR claramente identificável;
- [ ] DVC executado e DAG exibido;
- [ ] MLflow Experiments e Model Registry demonstrados;
- [ ] modelo `champion` visível;
- [ ] Docker saudável e CI verde;
- [ ] métricas e limitações mencionadas;
- [ ] nenhum `.env`, token ou dado pessoal visível;
- [ ] link testado em janela anônima.
