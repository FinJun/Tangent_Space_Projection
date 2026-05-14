"""Path utilities."""

from pathlib import Path


def get_results_dir(subdir=None):
    """Get results directory."""
    root = Path(__file__).parent.parent.parent
    results = root / "results"
    if subdir:
        results = results / subdir
    results.mkdir(parents=True, exist_ok=True)
    return results
