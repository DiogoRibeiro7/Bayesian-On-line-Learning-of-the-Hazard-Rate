import numpy as np

class ThreeLevelChangePointHierarchy:
    def __init__(self, H0, pi, Tmax):
        # H0: hazard function for the top level
        # pi: predictive probability function
        # Tmax: maximum number of time steps
        self.H0 = H0
        self.pi = pi
        self.Tmax = Tmax
        self.nodes = {}

    def initialize(self):
        self.nodes[0] = {(0, 0, 0): 1}  # w(r0=0, a0=0, b0=0, t=0) = 1
        self.Lt = set([(0, 0, 0)])  # nodelist Lt=0 = {N(0, 0, 0, 0)}

    def update_nodes(self, t, x_t):
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

    def update_weight(self, nodes, key, value, Lt):
        if key not in nodes:
            nodes[key] = 0
        nodes[key] += value
        Lt.add(key)

    def normalize_nodes(self, nodes):
        Wtotal = sum(nodes.values())
        for key in nodes:
            nodes[key] /= Wtotal
        return nodes

    def prune_nodes(self, Lt):
        # Placeholder for actual pruning logic, if required
        return Lt

    def predict(self):
        prediction = 0
        for (rt, at, bt), w in self.nodes.items():
            prediction += self.pi(rt, at, bt) * w
        return prediction

    def run(self, data):
        self.initialize()

        predictions = []
        for t, x_t in enumerate(data, 1):
            self.nodes, self.Lt = self.update_nodes(t, x_t)
            predictions.append(self.predict())

        return predictions

# Example usage:

# Define the hazard function (for simplicity, constant hazard rate for the top level)
H0 = 0.1

# Define a predictive probability function (for illustration purposes)
def pi(rt, at, bt, x_t):
    return np.exp(-((x_t - rt) ** 2) / 2) / np.sqrt(2 * np.pi)

# Maximum time steps
Tmax = 100

# Generate some example data
data = np.random.randn(Tmax)

# Instantiate and run the model
model = ThreeLevelChangePointHierarchy(H0, pi, Tmax)
predictions = model.run(data)

print(predictions)
