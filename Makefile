PY ?= python3
PIP ?= $(PY) -m pip

.PHONY: help install dev test lint fmt demo backtest paper status kill dashboard gate docker-build docker-up docker-down backup clean

help:
	@echo "install      install Sentinel (core + exchanges + ml)"
	@echo "dev          install with dev tools"
	@echo "test         run the full test suite"
	@echo "lint         ruff lint"
	@echo "demo         run the offline demo (synthetic data, no API keys)"
	@echo "backtest     demo backtest with HTML report"
	@echo "paper        paper trade on a synthetic feed for 500 bars"
	@echo "status       show bot status"
	@echo "kill         engage kill switch"
	@echo "dashboard    start the local dashboard on 127.0.0.1:8765"
	@echo "gate         run the LIVE readiness gate"

install:
	$(PIP) install -e ".[all]"

dev:
	$(PIP) install -e ".[all,dev]"

test:
	$(PY) -m pytest

lint:
	ruff check sentinel tests

fmt:
	ruff check --fix sentinel tests
	ruff format sentinel tests

demo:
	$(PY) -m sentinel demo

backtest:
	$(PY) -m sentinel backtest --demo --strategy ema_adx --report

paper:
	$(PY) -m sentinel paper --demo --max-bars 500

status:
	$(PY) -m sentinel status

kill:
	$(PY) -m sentinel kill

dashboard:
	$(PY) -m sentinel dashboard

gate:
	$(PY) -m sentinel gate

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

backup:
	./scripts/backup.sh

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
