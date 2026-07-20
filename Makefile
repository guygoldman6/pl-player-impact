.PHONY: all scrape build model outputs test notebooks

all: scrape build test model outputs

scrape:
	uv run plimpact scrape

build:
	uv run plimpact build

model:
	uv run plimpact model

outputs:
	uv run plimpact outputs

test:
	uv run pytest -q

notebooks:
	uv run --group dev jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
