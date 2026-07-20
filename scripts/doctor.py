#!/usr/bin/env python
"""Wrapper for the package doctor command."""

from __future__ import annotations

import sys

from causal_workspace_jepa.cli import main


if __name__ == "__main__":
    sys.exit(main(["doctor", *sys.argv[1:]]))
