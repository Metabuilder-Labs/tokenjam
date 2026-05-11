.PHONY: test test-unit test-e2e lint typecheck all

test:
	pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/

test-unit:
	pytest tests/unit/

test-e2e:
	pytest tests/e2e/

lint:
	ruff check tokenjam/

typecheck:
	mypy tokenjam/

all: lint typecheck test
