"""Tests verifying the execution and database safety of demo.py and minivec.eval CLI."""

from pathlib import Path
from demo import run_demo
from minivec.eval import run_cli_evaluation


def test_demo_execution_and_isolation(monkeypatch, capsys):
    """Verifies that demo.py executes cleanly without modifying persistent development data."""
    data_db = Path("data/minivec.db")
    initial_mtime = data_db.stat().st_mtime if data_db.exists() else None
    initial_size = data_db.stat().st_size if data_db.exists() else None

    # Run demo
    run_demo()

    captured = capsys.readouterr()
    assert "MiniVec" in captured.out
    assert "Noise-Cancelling Headphones" in captured.out
    assert "Demonstration completed successfully" in captured.out

    # Verify data/minivec.db was completely untouched
    if data_db.exists():
        assert data_db.stat().st_mtime == initial_mtime
        assert data_db.stat().st_size == initial_size


def test_eval_cli_execution(capsys):
    """Verifies that python -m minivec.eval runs cleanly across Cosine, Euclidean, and Dot metrics."""
    run_cli_evaluation()

    captured = capsys.readouterr()
    assert "Recall@k & Latency Evaluation" in captured.out
    assert "Metric: cosine" in captured.out
    assert "Metric: euclidean" in captured.out
    assert "Metric: dot" in captured.out
    assert "Evaluation completed successfully" in captured.out
