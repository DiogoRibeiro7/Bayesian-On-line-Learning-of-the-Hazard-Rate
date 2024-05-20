import numpy as np

class ConstantHazardRate:
    def __init__(self, H, pi, Tmax):
        # H: hazard function
        # pi: predictive probability function
        # Tmax: maximum number of time steps
        self.H = H
        self.pi = pi
        self.Tmax = Tmax
        self.nodes = {}
        self.data = []

    def initialize(self):
        self.nodes = {(0, 0): 1}  # w(r0=0, a0=0, t=1) = 1
        self.Wtotal = 0
        self.Lt = set([(0, 0)])  # nodelist Lt=0 = {N(0, 0, 0)}

    def update_nodes(self, t, x_t):
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

    def normalize_nodes(self, nodes, Wtotal):
        for key in nodes:
            nodes[key] /= Wtotal
        return nodes

    def predict(self):
        prediction = 0
        for (rt, at), w in self.nodes.items():
            prediction += self.pi(rt, at, self.data[-1]) * w
        return prediction

    def run(self, data):
        self.initialize()
        self.data = data  # Store the data for access in predict()

        predictions = []
        for t, x_t in enumerate(data, 1):
            self.nodes, self.Wtotal, self.Lt = self.update_nodes(t, x_t)
            self.nodes = self.normalize_nodes(self.nodes, self.Wtotal)
            predictions.append(self.predict())

        return predictions

# Example usage:

# Define the hazard function (for simplicity, constant hazard rate)
H = 0.1

# Define a predictive probability function (for illustration purposes)
def pi(rt, at, x_t):
    return np.exp(-((x_t - rt) ** 2) / 2) / np.sqrt(2 * np.pi)

# Maximum time steps
Tmax = 100

# Generate some example data
data = np.random.randn(Tmax)

# Instantiate and run the model
model = ConstantHazardRate(H, pi, Tmax)
predictions = model.run(data)

print(predictions)
