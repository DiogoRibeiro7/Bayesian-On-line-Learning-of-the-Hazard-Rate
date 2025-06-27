"""Utilities for a simple Expectation-Maximization algorithm.

This module implements the :class:`EMAlgorithm` class without relying on
external dependencies such as NumPy. All computations are performed using
Python's built-in ``math`` module and list comprehensions.
"""

from __future__ import annotations

import math

class EMAlgorithm:
    """Expectation-Maximization algorithm implemented using pure Python."""

    def __init__(self, x_list: list[float], e_matrix: list[list[float]]) -> None:
        """Initialize the algorithm with data and initial estimates.

        Args:
            x_list: Sequence of observed values.
            e_matrix: Initial responsibility matrix as a list of lists.
        """

        self.x_list = x_list
        self.e_matrix = e_matrix

    @staticmethod
    def exponent_func(x: float, std: float) -> float:
        """Compute the exponent term used in the E step.

        Args:
            x: Input value.
            std: Standard deviation of the component.

        Returns:
            Computed exponent value.
        """
        return math.exp((-1.0 * x) / (2 * std))

    def get_std(self, mean_list: list[float]) -> list[float]:
        """Compute the standard deviation for each component.

        Args:
            mean_list: List of mean values.

        Returns:
            List of standard deviations for each component.
        """

        std_list = []
        for j, mean in enumerate(mean_list):
            # Compute weighted variance for component ``j``.
            numerator = 0.0
            weight_sum = 0.0
            for i, x in enumerate(self.x_list):
                weight = self.e_matrix[i][j]
                numerator += ((mean - x) ** 2) * weight
                weight_sum += weight

            std_list.append(math.sqrt(numerator / weight_sum))

        return std_list

    def E_step(self, mean_list: list[float]) -> list[list[float]]:
        """Perform the expectation step for a list of means.

        Args:
            mean_list: List of mean values.

        Returns:
            Updated responsibility matrix after the E step.
        """

        std_list = self.get_std(mean_list)
        estimated_cols = []

        for mean, std in zip(mean_list, std_list):
            # Compute unnormalized responsibilities for each observation.
            col = [self.exponent_func((x - mean) ** 2, std) for x in self.x_list]
            col_sum = sum(col)
            col = [val / col_sum for val in col]
            estimated_cols.append(col)

        # Transpose so rows correspond to observations and columns to components.
        estimated = [list(row) for row in zip(*estimated_cols)]
        return estimated

    def M_step(self) -> list[float]:
        """Perform the maximization step.

        Returns:
            List of updated mean values.
        """

        means = []
        num_components = len(self.e_matrix[0])
        for j in range(num_components):
            numerator = 0.0
            denominator = 0.0
            for i, x in enumerate(self.x_list):
                weight = self.e_matrix[i][j]
                numerator += x * weight
                denominator += weight
            means.append(numerator / denominator)

        return means

    def get_theta(self) -> list[float]:
        """Calculate the theta values for the current ``e_matrix``.

        Returns:
            List of theta values for each component.
        """

        num_obs = len(self.e_matrix)
        num_components = len(self.e_matrix[0])
        thetas = []
        for j in range(num_components):
            total = sum(self.e_matrix[i][j] for i in range(num_obs))
            thetas.append(total / num_obs)

        return thetas

    def simulate_E_M(self, steps: int) -> list[list[float]]:
        """Run several iterations of the E-M algorithm.

        Args:
            steps: Number of E-M steps to perform.

        Returns:
            Matrix of mean values after each step with shape
            ``len(mean_list) x steps``.
        """

        mean_matrix = []
        for _ in range(steps):
            mean_list = self.M_step()
            mean_matrix.append(mean_list)
            self.e_matrix = self.E_step(mean_list)

        # Transpose to keep a consistent orientation with the original code.
        return [list(col) for col in zip(*mean_matrix)]

if __name__ == "__main__":
    # Example usage with plain Python lists
    x_list = [1.0, 2.0, 3.0]
    e_matrix = [
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
    ]

    em_algorithm = EMAlgorithm(x_list, e_matrix)
    mean_matrix = em_algorithm.simulate_E_M(steps=2)
    print(mean_matrix)
