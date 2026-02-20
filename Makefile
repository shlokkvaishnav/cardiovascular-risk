.PHONY: install test train predict clean

install:
	pip install -e "backend[dev,viz]"
	pre-commit install

test:
	pytest backend/tests/ -v --cov=backend/src --cov-report=html

train:
	python backend/scripts/train.py --config backend/config/config.yaml

predict:
	python backend/scripts/predict.py --input backend/data/test_sample.csv

lint:
	black backend/src/ backend/tests/
	flake8 backend/src/ backend/tests/
	mypy backend/src/

docker-build:
	docker build -t cardiovascular-risk-api backend

docker-run:
	docker-compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
