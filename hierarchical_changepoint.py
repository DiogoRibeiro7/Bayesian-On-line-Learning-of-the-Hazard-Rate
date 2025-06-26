"""Hierarchical changepoint model example.

This module defines :class:`ThreeLevelChangePointHierarchy`, showcasing a
three-level hazard rate hierarchy for Bayesian on-line changepoint detection.
The example at the bottom illustrates typical usage.
"""

from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np

__all__ = ["ThreeLevelChangePointHierarchy"]

class ThreeLevelChangePointHierarchy:
    """Three-level hierarchy for Bayesian online changepoint detection.

    The model propagates weights through a hierarchy of hazard rates,
    capturing multiple levels of changepoint structure.

    Args:
        H0 (float): Top-level hazard rate.
        pi (Callable[[int, int, int, float], float]): Predictive probability
            function.
        Tmax (int): Maximum number of time steps.
    """

    def __init__(
        self, H0: float, pi: Callable[[int, int, int, float], float], Tmax: int
    ) -> None:
        """Initialize hierarchy parameters."""

        self.H0 = H0
        self.pi = pi
        self.Tmax = Tmax
        self.nodes = {}
        self.data = []

    def initialize(self) -> None:
        """Initialize hierarchy state."""

        self.nodes = {(0, 0, 0): 1}  # w(r0=0, a0=0, b0=0, t=0) = 1
        self.Lt = set([(0, 0, 0)])  # nodelist Lt=0 = {N(0, 0, 0, 0)}

    def update_weight(
        self,
        nodes: Dict[Tuple[int, int, int], float],
        key: Tuple[int, int, int],
        value: float,
        Lt: Set[Tuple[int, int, int]],
    ) -> None:
        """Accumulate weight for a specific node.

        Args:
            nodes (dict): Dictionary of node weights.
            key (tuple): Node identifier.
            value (float): Weight contribution.
            Lt (set): Set of active nodes.
        """

        if key not in nodes:
            nodes[key] = 0
        nodes[key] += value
        Lt.add(key)

    def update_nodes(
        self, t: int, x_t: float
    ) -> Tuple[Dict[Tuple[int, int, int], float], Set[Tuple[int, int, int]]]:
        """Update all nodes for a new observation.

        Args:
            t (int): Current time step.
            x_t (float): Observed value at time ``t``.

        Returns:
            Tuple[dict, set]: Updated node dictionary and pruned set of nodes.
        """

        new_nodes = {}
        Lt = set()

        for (rt, at, bt), w in self.nodes.items():
            # Observe data xt and compute predictive probability
            pi_xt = self.pi(rt, at, bt, x_t)
            # Compute estimate of hazard rate
            h_t = (at + 1) / (at + bt + 2)

            # Send messages to children and update weights
            self.update_weight(new_nodes, (rt + 1, at, bt + 1), (1 - h_t) * (1 - self.H0) * w * pi_xt, Lt)
            self.update_weight(new_nodes, (0, at + 1, bt), h_t * (1 - self.H0) * w * pi_xt, Lt)
            self.update_weight(new_nodes, (rt + 1, 0, 0), (1 - h_t) * self.H0 * w * pi_xt, Lt)
            self.update_weight(new_nodes, (0, 0, bt + 1), h_t * self.H0 * w * pi_xt, Lt)

        new_nodes = self.normalize_nodes(new_nodes)
        Lt = self.prune_nodes(Lt)

        return new_nodes, Lt

    def normalize_nodes(
        self, nodes: Dict[Tuple[int, int, int], float]
    ) -> Dict[Tuple[int, int, int], float]:
        """Normalize node weights so they sum to one.

        Args:
            nodes (dict): Dictionary of node weights.

        Returns:
            dict: Normalized nodes.
        """

        Wtotal = sum(nodes.values())
        for key in nodes:
            nodes[key] /= Wtotal
        return nodes

    def prune_nodes(self, Lt: Set[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
        """Prune inactive nodes from ``Lt``.

        This is currently a placeholder; real pruning logic can be implemented
        depending on the application.

        Args:
            Lt (set): Set of active nodes.

        Returns:
            set: Possibly pruned set of nodes.
        """

        return Lt

    def predict(self) -> float:
        """Predict the next observation based on current node weights.

        Returns:
            float: Expected value of the next observation.
        """

        prediction = 0
        for (rt, at, bt), w in self.nodes.items():
            prediction += self.pi(rt, at, bt, self.data[-1]) * w
        return prediction

    def run(self, data: Iterable[float]) -> List[float]:
        """Execute online learning over provided data.

        Args:
            data (Iterable[float]): Sequence of observed values.

        Returns:
            List[float]: Predictions for each observation.
        """

        self.initialize()
        self.data = data  # Store the data for access in predict()

        predictions = []
        for t, x_t in enumerate(data, 1):
            self.nodes, self.Lt = self.update_nodes(t, x_t)
            predictions.append(self.predict())

        return predictions

if __name__ == "__main__":
    # Example usage

    H0 = 0.1  # Top-level constant hazard rate

    def pi(rt: int, at: int, bt: int, x_t: float) -> float:
        """Simple Gaussian predictive probability."""

        return np.exp(-((x_t - rt) ** 2) / 2) / np.sqrt(2 * np.pi)

    Tmax = 100
    data = np.random.randn(Tmax)

    model = ThreeLevelChangePointHierarchy(H0, pi, Tmax)
    predictions = model.run(data)

    print(predictions)
