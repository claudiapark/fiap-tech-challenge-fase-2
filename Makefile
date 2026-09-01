.PHONY: install pipeline test lint format mlflow docker-build docker-run

install:
	uv sync --locked

pipeline:
	uv run dvc repro

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

mlflow:
	uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

docker-build:
	docker build -t purchase-propensity:latest .

docker-run:
	docker run --rm -p 5000:5000 purchase-propensity:latest

