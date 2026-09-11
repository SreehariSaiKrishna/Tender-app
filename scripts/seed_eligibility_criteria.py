"""Seed (or update) the `eligibility_criteria` MongoDB collection from the
in-code defaults in app.processing.eligibility (transcribed from the
client's Eligibility.pdf). Idempotent - safe to re-run after editing
DEFAULT_CRITERIA there.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\seed_eligibility_criteria.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.processing.eligibility import DEFAULT_CRITERIA, seed_default_criteria  # noqa: E402


def main() -> None:
    seed_default_criteria()
    print(f"Seeded eligibility_criteria/{DEFAULT_CRITERIA['_id']} with {len(DEFAULT_CRITERIA['criteria'])} criteria rows.")


if __name__ == "__main__":
    main()
