# Contribuição

1. Crie uma branch com nome descritivo.
2. Instale o projeto com `uv sync --locked`.
3. Faça alterações pequenas e com responsabilidade única.
4. Execute `uv run ruff format .`, `uv run ruff check .` e `uv run pytest`.
5. Quando código ou parâmetros do pipeline mudarem, execute `uv run dvc repro`.
6. Abra uma Pull Request explicando objetivo, validações e impacto nas métricas.

Commits devem descrever a alteração realizada e não devem conter dados gerados, caches, `.env`, banco local do MLflow ou credenciais de storage.

