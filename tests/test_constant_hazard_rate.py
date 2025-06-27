import types
import math
import sys
from pathlib import Path

# Make sure the project root is on sys.path so tests can import modules.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a minimal NumPy stub for environments without NumPy installed.
numpy_stub = types.SimpleNamespace(
    exp=math.exp,
    sqrt=math.sqrt,
    random=types.SimpleNamespace(randn=lambda n: [0.0] * n),
)
sys.modules.setdefault("numpy", numpy_stub)

from constant_hazard_rate import ConstantHazardRate


def constant_pi(rt: int, at: int, x_t: float) -> float:
    """Return a constant predictive probability."""
    return 1.0


def test_predictions_length() -> None:
    """Model returns a prediction for each input value."""

    data = [0.0, 1.0, 2.0]
    model = ConstantHazardRate(H=0.5, pi=constant_pi, Tmax=len(data))
    preds = model.run(data)

    assert len(preds) == len(data)


def test_predictions_constant_pi() -> None:
    """Predictions equal one when predictive probability is constant."""

    data = [0.0, 1.0, 2.0]
    model = ConstantHazardRate(H=0.5, pi=constant_pi, Tmax=len(data))
    preds = model.run(data)

    assert all(math.isclose(p, 1.0) for p in preds)


def test_normalize_nodes_sum() -> None:
    """Normalized node weights sum to one."""

    model = ConstantHazardRate(H=0.5, pi=constant_pi, Tmax=1)
    nodes = {(0, 0): 0.2, (1, 0): 0.3, (0, 1): 0.5}
    normalized = model.normalize_nodes(nodes, Wtotal=sum(nodes.values()))

    total = sum(normalized.values())
    assert math.isclose(total, 1.0)



def test_update_nodes_constant_pi() -> None:
    """Update nodes with constant predictive probability.

    Ensure the node weights are correctly computed at the first step.
    """
    model = ConstantHazardRate(H=0.5, pi=constant_pi, Tmax=1)
    model.initialize()
    nodes, Wtotal, Lt = model.update_nodes(1, 1.0)
    assert math.isclose(nodes[(1, 0)], 0.6)
    assert math.isclose(nodes[(0, 1)], 0.4)
    assert math.isclose(Wtotal, 1.0)
