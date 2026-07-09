"""Typer CLI entry point. Loads config, wires components, runs the ablation, writes the report."""
from __future__ import annotations
import logging
from pathlib import Path

import typer

from ..config import ExperimentConfig

app = typer.Typer(help="Robust hedge-ratio filtering framework.")


@app.command()
def run(config: Path = typer.Option("config/default.yaml", exists=True,
                                    help="Path to the experiment YAML.")) -> None:
    """Run the ablation grid defined by CONFIG and write results/REPORT.md."""
    logging.basicConfig(level=logging.INFO)
    cfg = ExperimentConfig.from_yaml(config)
    # TODO: AblationRunner(cfg).run(); eval.reports.write_report(...)
    raise NotImplementedError("cli.run wiring")


if __name__ == "__main__":
    app()
