.PHONY: test demo serve
test:
	python -m pytest
demo:
	python -m clinical_lab.cli run --synthetic --config configs/demo.yaml --output artifacts/demo
serve:
	python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
