.PHONY: setup data train evaluate serve test docker-build docker-run all

setup:
	py -m pip install -r requirements.txt

data:
	py src/data_preparation.py

train:
	py src/train.py

evaluate:
	py -c "import json; print(json.load(open('reports/metrics/training_results.json')))"

serve:
	py src/app.py

test:
	py -m pytest tests/ -v

docker-build:
	docker build -t loan-approval-prediction:latest .

docker-run:
	docker run -p 8000:8000 loan-approval-prediction:latest

all: data train test
	@echo "Pipeline complete"
