# Commands work from the repository root or with make -f /path/to/Makefile.
PROJECT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE_PROJECT_NAME ?= juribot
COMPOSE := docker compose -p $(COMPOSE_PROJECT_NAME) -f "$(PROJECT_DIR)ops/docker-compose.yml"
PYTHON ?= python

.PHONY: help config up down build logs logs-api logs-web shell-api ingest eval test
help:
	@echo "make config     - Valida o Docker Compose"
	@echo "make up         - Inicia API, Qdrant e Redis"
	@echo "make down       - Para os servicos, preservando volumes"
	@echo "make build      - Constroi a imagem do backend"
	@echo "make logs       - Acompanha logs dos servicos"
	@echo "make logs-api   - Acompanha logs da API"
	@echo "make logs-web   - Acompanha o frontend opcional"
	@echo "make shell-api  - Abre shell na API"
	@echo "make ingest     - Indexa os PDFs no container"
	@echo "make eval       - Executa o modulo de avaliacao"
	@echo "make test       - Executa os testes Python"

config:
	$(COMPOSE) config --quiet

up:
	$(COMPOSE) up -d --build

# Do not use --volumes: database contents must be preserved.
down:
	$(COMPOSE) down

build:
	$(COMPOSE) build api

logs:
	$(COMPOSE) logs -f

logs-api:
	$(COMPOSE) logs -f api

logs-web:
	$(COMPOSE) --profile frontend logs -f frontend

shell-api:
	$(COMPOSE) exec api /bin/bash

ingest:
	$(COMPOSE) exec api python -m backend.ingest.prepare_index

eval:
	$(COMPOSE) exec api python -m backend.eval.eval_runner

test:
	cd "$(PROJECT_DIR)" && $(PYTHON) -B -m unittest discover -s backend/tests -p "test_*.py" -v
