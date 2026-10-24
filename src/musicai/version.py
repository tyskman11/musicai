"""Eine zentrale, zur Laufzeit sichtbare Projektversion."""
from pathlib import Path

VERSION = "0.2.4"
CHANGELOG_PATH = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
