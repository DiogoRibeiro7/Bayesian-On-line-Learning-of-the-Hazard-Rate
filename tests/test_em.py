import math
import sys
from pathlib import Path

# Make sure the project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from em import EMAlgorithm


def test_m_step_simple() -> None:
    """M_step averages values when responsibilities are uniform."""
    x_list = [1.0, 2.0]
    e_matrix = [[0.5, 0.5], [0.5, 0.5]]
    em = EMAlgorithm(x_list, e_matrix)
    means = em.M_step()
    assert all(math.isclose(m, 1.5) for m in means)


def test_e_step_idempotent() -> None:
    """E_step returns the same matrix for symmetric input."""
    x_list = [1.0, 2.0]
    e_matrix = [[0.5, 0.5], [0.5, 0.5]]
    em = EMAlgorithm(x_list, e_matrix)
    means = [1.5, 1.5]
    updated = em.E_step(means)
    for i in range(2):
        for j in range(2):
            assert math.isclose(updated[i][j], e_matrix[i][j])

