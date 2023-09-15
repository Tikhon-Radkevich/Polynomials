from typing import List, Tuple


class Scaler:
    def __init__(self, data: List[float]):
        self.max = max(data)

    def scale(self, data: List[float]) -> List[float]:
        """Scale the input data by dividing each value by the maximum value."""
        return [val / self.max for val in data]

    def restore(self, data: List[float]) -> List[float]:
        """Restore the scaled data by multiplying each value by the maximum value."""
        return [val * self.max for val in data]


class PowerFunc:
    def __init__(self, max_degree: int, learning_rate: float):
        self.neuron = Neuron(max_degree, learning_rate)

    def predict(self, x_data: List[float]) -> List[float]:
        """Predict output for the given input data."""
        return [self.neuron.forward(x) for x in x_data]

    def fit(self, x_data: List[float], y_data: List[float]):
        """Train the model using the given input and target data."""
        for x, y in zip(x_data, y_data):
            output = self.neuron.forward(x)
            self.neuron.backward(x, y, output)


class Neuron:
    def __init__(self, max_degree: int, learning_rate: float):
        self.learning_rate = learning_rate
        self.weights = [0 for _ in range(max_degree + 1)]
        self.bias = 0

    def backward(self, x: float, y: float, output: float):
        """Update weights and bias using backpropagation."""
        error = y - output
        self.bias += error * self.learning_rate
        for degree in range(len(self.weights)):
            x_degree = x ** degree
            self.weights[degree] += x_degree * error * self.learning_rate

    def forward(self, x: float) -> float:
        """Compute the output for the given input."""
        output = sum(weight * (x ** degree) for degree, weight in enumerate(self.weights))
        return output


class Datamining:

    def __init__(self, train_set: List[Tuple[float, float]]):
        x_data = [point[0] for point in train_set]
        y_data = [point[1] for point in train_set]

        self.x_scaler = Scaler(x_data)
        x_data = self.x_scaler.scale(x_data)

        self.y_scaler = Scaler(y_data)
        y_data = self.y_scaler.scale(y_data)

        max_degree = 3
        learning_rate = 0.6
        self.model = PowerFunc(max_degree, learning_rate)
        for _ in range(300):
            self.model.fit(x_data, y_data)

    def predict(self, x: float) -> float:
        x_data = self.x_scaler.scale([x])
        y_pred = self.model.predict(x_data)
        return self.y_scaler.restore(y_pred)[0]
