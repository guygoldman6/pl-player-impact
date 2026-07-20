"""Command-line interface: scrape -> build -> model -> outputs."""

from __future__ import annotations

import logging

import typer

from .config import load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def scrape() -> None:
    """Fetch raw Understat + football-data.co.uk data for all configured seasons."""
    from . import scrape as scrape_mod

    scrape_mod.fetch_all(load_config())


@app.command()
def build() -> None:
    """Parse raw JSON into tidy parquet tables (matches, appearances, shots, stints)."""
    from . import build as build_mod

    build_mod.build_all(load_config())


@app.command()
def model() -> None:
    """Fit naive plus-minus, RAPM, and xG-RAPM; write ratings to processed dir."""
    from . import model as model_mod

    model_mod.run_all(load_config())


@app.command()
def outputs() -> None:
    """Generate final rankings CSV and charts under outputs/."""
    from . import viz as viz_mod

    viz_mod.make_outputs(load_config())


@app.command(name="all")
def run_all() -> None:
    """Full pipeline: scrape (cached) -> build -> model -> outputs."""
    scrape()
    build()
    model()
    outputs()


if __name__ == "__main__":
    app()
