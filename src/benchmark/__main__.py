"""Allow ``python -m src.benchmark`` to invoke the CLI."""

from .demo import main

if __name__ == "__main__":
    raise SystemExit(main())
