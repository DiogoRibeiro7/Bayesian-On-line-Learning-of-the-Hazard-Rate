"""Utilities for a simple Expectation-Maximization algorithm."""

import numpy as np

class EMAlgorithm:
    def __init__(self, x_list: np.ndarray, e_matrix: np.ndarray):
        """Initialize the EMAlgorithm with data and initial estimates.

        Args:
            x_list: Array of x values.
            e_matrix: Initial matrix of estimated values.
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
        return np.exp((-1.0 * x) / (2 * std))

    def get_std(self, mean_list: np.ndarray) -> np.ndarray:
        """Compute the standard deviation for each component.

        Args:
            mean_list: Array of mean values.

        Returns:
            Array of standard deviations.
        """
        x_vector = self.x_list[:, np.newaxis]
        var = np.square(mean_list - x_vector) * self.e_matrix
        return np.sqrt(var.sum(axis=0) / self.e_matrix.sum(axis=0))

    def E_step(self, mean_list: np.ndarray) -> np.ndarray:
        """Perform the expectation step for a list of means.

        Args:
            mean_list: Array of mean values.

        Returns:
            Updated ``e_matrix`` after the E step.
        """
        std_list = self.get_std(mean_list)
        
        estimated = [np.vectorize(self.exponent_func)(np.square(self.x_list - mean), std) for mean, std in zip(mean_list, std_list)]
        estimated = np.array(estimated).transpose()
        
        for i, n in enumerate(estimated):
            estimated[i] /= n.sum()
        
        return estimated

    def M_step(self) -> np.ndarray:
        """Perform the maximization step.

        Returns:
            Updated mean values.
        """
        numerator = np.dot(self.x_list, self.e_matrix)
        denominator = self.e_matrix.sum(axis=0)
        return np.divide(numerator, denominator)

    def get_theta(self) -> np.ndarray:
        """Calculate the theta values for the current ``e_matrix``.

        Returns:
            Theta values.
        """
        return self.e_matrix.sum(axis=0) / len(self.e_matrix)

    def simulate_E_M(self, steps: int) -> np.ndarray:
        """Run several iterations of the E-M algorithm.

        Args:
            steps: Number of E-M steps to perform.

        Returns:
            Matrix of mean values after each step.
        """
        mean_matrix = []
        for _ in range(steps):
            mean_list = self.M_step()
            mean_matrix.append(mean_list)
            self.e_matrix = self.E_step(mean_list)
        
        return np.array(mean_matrix).transpose()

if __name__ == "__main__":
    # Example usage
    x_list = np.array([...])  # Replace with your data
    e_matrix = np.array([...])  # Replace with your initial estimates
    em_algorithm = EMAlgorithm(x_list, e_matrix)
    mean_matrix = em_algorithm.simulate_E_M(steps=10)
    print(mean_matrix)
