"""Constant hazard rate changepoint model example.

This module provides the :class:`ConstantHazardRate` class, a simple example
of Bayesian on-line learning with a single changepoint hierarchy. The example
at the bottom demonstrates how the class can be used.
"""

from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np

__all__ = ["ConstantHazardRate"]

class ConstantHazardRate:
    """Bayesian online learning with a constant hazard rate.

    This model tracks changepoints by propagating weight among nodes in a
    hierarchy. The hazard rate is assumed constant over time.

    Args:
        H (float): Constant hazard rate hyperparameter.
        pi (Callable[[int, int, float], float]): Predictive probability
            function.
        Tmax (int): Maximum number of time steps.
    """

    def __init__(self, H: float, pi: Callable[[int, int, float], float], Tmax: int) -> None:
        """Initialize the model parameters.

        Args:
            H: Constant hazard rate hyperparameter.
            pi: Predictive probability function.
            Tmax: Maximum number of time steps.
        """

        self.H = H
        self.pi = pi
        self.Tmax = Tmax
        self.nodes = {}
        self.data = []

    def initialize(self) -> None:
        """Initialize the node weights and state variables."""

        self.nodes = {(0, 0): 1}  # w(r0=0, a0=0, t=1) = 1
        self.Wtotal = 0
        self.Lt = set([(0, 0)])  # nodelist Lt=0 = {N(0, 0, 0)}

    def update_nodes(
        self, t: int, x_t: float
    ) -> Tuple[Dict[Tuple[int, int], float], float, Set[Tuple[int, int]]]:
        """Update the node weights for a new observation.

        Args:
            t (int): Current time step.
            x_t (float): Observed value at time ``t``.

        Returns:
            Tuple[dict, float, set]: Updated nodes, total weight, and the new
            node set ``Lt``.
        """

        new_nodes = {}
        Wtotal = 0
        Lt = set()

        for (rt, at), w in self.nodes.items():
            # Observe data xt and compute predictive probability
            pi_xt = self.pi(rt, at, x_t)
            # Compute estimate of hazard rate
            h_t = (at + 1) / (at + self.H + 2)

            # Send messages to children
            new_rt, new_at = rt + 1, at
            if (new_rt, new_at) not in new_nodes:
                new_nodes[(new_rt, new_at)] = 0
            new_nodes[(new_rt, new_at)] += (1 - h_t) * w * pi_xt

            new_rt, new_at = 0, at + 1
            if (new_rt, new_at) not in new_nodes:
                new_nodes[(new_rt, new_at)] = 0
            new_nodes[(new_rt, new_at)] += h_t * w * pi_xt

            Lt.add((new_rt, new_at))
            Wtotal += w * pi_xt

        return new_nodes, Wtotal, Lt

    def normalize_nodes(
        self, nodes: Dict[Tuple[int, int], float], Wtotal: float
    ) -> Dict[Tuple[int, int], float]:
        """Normalize node weights to sum to one.

        Args:
            nodes (dict): Dictionary of node weights.
            Wtotal (float): Sum of all unnormalized weights.

        Returns:
            dict: Normalized node weights.
        """

        for key in nodes:
            nodes[key] /= Wtotal
        return nodes

    def predict(self) -> float:
        """Predict the next observation using current node weights.

        Returns:
            float: Expected value of the next observation.
        """

        prediction = 0
        for (rt, at), w in self.nodes.items():
            prediction += self.pi(rt, at, self.data[-1]) * w
        return prediction

    def run(self, data: Iterable[float]) -> List[float]:
        """Run the online learning algorithm over a sequence of data.

        Args:
            data (Iterable[float]): Sequence of observed values.

        Returns:
            List[float]: The predicted value after each observation.
        """

        self.initialize()
        self.data = data  # Store the data for access in predict()

        predictions = []
        for t, x_t in enumerate(data, 1):
            self.nodes, self.Wtotal, self.Lt = self.update_nodes(t, x_t)
            self.nodes = self.normalize_nodes(self.nodes, self.Wtotal)
            predictions.append(self.predict())

        return predictions

if __name__ == "__main__":
    # Example usage

    H = 0.1  # Constant hazard rate

    def pi(rt: int, at: int, x_t: float) -> float:
        """Simple Gaussian predictive probability."""

        return np.exp(-((x_t - rt) ** 2) / 2) / np.sqrt(2 * np.pi)

    Tmax = 100
    data = np.random.randn(Tmax)

    model = ConstantHazardRate(H, pi, Tmax)
    predictions = model.run(data)

    print(predictions)
