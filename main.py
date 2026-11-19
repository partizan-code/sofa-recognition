"""Streamlit entry point: run with `python -m streamlit run main.py`."""

from dotenv import load_dotenv

import config
from log_config import configure_logging


def main() -> int:
    load_dotenv(config.ROOT / ".env", override=False)
    configure_logging()
    from ui import render

    render()
    return 0


if __name__ == "__main__":
    main()
