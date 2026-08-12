"""Run the Typer CLI, with an offline-environment fallback."""

try:
    from .cli import run
except ModuleNotFoundError as exc:
    if exc.name != "typer":
        raise
    from .argparse_cli import run


if __name__ == "__main__":
    run()
