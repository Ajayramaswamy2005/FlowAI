"""Allows `python -m flowai`."""

import sys

from flowai.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
