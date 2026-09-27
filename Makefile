.PHONY: install run test eval
install:
	pip install -r requirements.txt
run:
	python run.py
test:
	python -m pytest -q
eval:
	python -m evaluation.evaluate