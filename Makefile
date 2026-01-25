.PHONY: install test train predict clean

install:
	pip install -e ".[dev,viz]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python scripts/train.py --config config/config.yaml

predict:
	python scripts/predict.py --input data/test_sample.csv

lint:
	black src/ tests/
	flake8 src/ tests/
	mypy src/

docker-build:
	docker build -t heart-disease-api .

docker-run:
	docker-compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
