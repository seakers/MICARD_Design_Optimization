#!/usr/bin/env python3
"""Dev convenience: `python main.py` from a checkout, no install required.

Real CLI lives in micard_design_optimization/cli.py (also the
`micard-design-optimization` console script once this package is pip
installed); this just calls it.
"""
from micard_design_optimization.cli import main

if __name__ == "__main__":
    main()
