.PHONY: setup clean lint test coverage docs docker-build docker-run download-models help

# Default help message
help:
	@echo "SL-License-OCR Makefile commands:"
	@echo "make setup         - Install dependencies and prepare development environment"
	@echo "make lint          - Run code quality checks"
	@echo "make test          - Run test suite"
	@echo "make coverage      - Run tests with coverage report"
	@echo "make clean         - Clean up temporary files and caches"
	@echo "make docs          - Generate documentation"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-run    - Run the application in Docker"
	@echo "make download-models - Download required ML models"

# Setup development environment
setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Clean up temporary files and caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.eggs" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".coverage.*" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .eggs/ *.egg-info/

# Run code quality checks
lint:
	isort .
	black .
	flake8 .
	mypy src

# Run tests
test:
	pytest tests/

# Run tests with coverage
coverage:
	pytest --cov=src tests/ --cov-report=html --cov-report=term

# Generate documentation
docs:
	mkdir -p docs/build
	# appropriate documentation generator command
	# For example, if using sphinx:
	# sphinx-build docs docs/build

# Download pre-trained models
download-models:
	# python scripts/download_models.py

# Docker commands
docker-build:
	docker build -t sl-license-ocr .

docker-run:
	docker run -p 8000:8000 sl-license-ocr

# Run the API server
run-api:
	uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000