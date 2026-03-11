# ─────────────────────────────────────────────
# MelodyTab AI — Makefile
# ─────────────────────────────────────────────

.PHONY: run test lint format install install-dev clean help

# Variáveis
PYTHON = .venv/bin/python
PIP = .venv/bin/pip

help: ## Exibe esta mensagem de ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Instala as dependências de produção
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Instala as dependências de desenvolvimento
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

run: ## Inicia o aplicativo Streamlit
	.venv/bin/streamlit run app/main.py

test: ## Roda os testes com cobertura
	.venv/bin/pytest tests/ --cov=app --cov-report=term-missing

test-unit: ## Roda apenas os testes unitários
	.venv/bin/pytest tests/unit/ -v

test-integration: ## Roda apenas os testes de integração
	.venv/bin/pytest tests/integration/ -v

lint: ## Verifica lint com ruff
	.venv/bin/ruff check app/ tests/

format: ## Formata o código com black
	.venv/bin/black app/ tests/

format-check: ## Verifica formatação sem alterar arquivos
	.venv/bin/black --check app/ tests/

clean: ## Remove arquivos temporários e cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -name ".coverage" -delete
	@echo "Limpeza concluída!"