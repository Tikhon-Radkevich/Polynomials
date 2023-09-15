import math
from random import random

from tqdm import tqdm


class Scaler:
    def __init__(self):
        self.max = 0
        self.min = 0

    def fit(self, x):
        self.min = min(x)
        self.max = max(x) - self.min
        return [self.scale(i) for i in x]

    def fit_min(self, x):
        self.max = max(x)
        return list(map(self.scale_min, x))

    def scale_min(self, x):
        return x / self.max

    def scale(self, x):
        return (x - self.min) / self.max

    def restore(self, val):
        return val * self.max + self.min

    def restore_min(self, val):
        return val * self.max


class CustomNeuralNetwork:
    def __init__(self, hidden_size):
        # Initialize weights and biases with random values
        self.weights_input_hidden = [random() for _ in range(hidden_size)]
        self.bias_hidden = [random() for _ in range(hidden_size)]
        self.hidden_outputs = [0 for _ in range(hidden_size)]
        self.weights_hidden_output = [random() for _ in range(hidden_size)]
        self.bias_output = random()
        self.output = 0
        self.learning_rate = 0.01

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward(self, x):
        # Calculate the output of the hidden layer
        for i, (weight, bias) in enumerate(zip(self.weights_input_hidden, self.bias_hidden)):
            # self.hidden_outputs[i] = self.sigmoid(x * weight + bias)
            self.hidden_outputs[i] = x * weight + bias

        # Calculate the output of the output layer
        self.output = 0
        for hidden_output, weight in zip(self.hidden_outputs, self.weights_hidden_output):
            self.output += hidden_output*weight
        self.output += self.bias_output
        return self.output

    def backward(self, x, y, output):
        # Calculate the error and deltas for the output layer
        error = y - output
        delta_output = error

        # Calculate the error and deltas for the hidden layer
        error_hidden = [delta_output*weight for weight in self.weights_hidden_output]
        delta_hidden = [err*output for err, output in zip(error_hidden, self.hidden_outputs)]

        # Calculate the error and delta for the hidden layer
        for i, hidden_output in enumerate(self.hidden_outputs):
            self.weights_hidden_output[i] += hidden_output * delta_output * self.learning_rate
        self.bias_output += delta_output * self.learning_rate

        # Update weights and biases for the hidden layer
        for i, delta_h in enumerate(delta_hidden):
            self.weights_input_hidden[i] += x * delta_h * self.learning_rate
            self.bias_hidden[i] += delta_h * self.learning_rate

    def fit(self, x_data, y_data, epochs=100):
        for _ in tqdm(range(epochs)):
            for x, y in zip(x_data, y_data):
                output = self.forward(x)
                self.backward(x, y, output)
        print(self.weights_input_hidden)
        print(self.weights_hidden_output)

    def predict(self, x):
        return self.forward(x)

    @staticmethod
    def rmse(y_true, y_pred):
        n = len(y_true)
        squared_errors = [(y_true[i] - y_pred[i]) ** 2 for i in range(n)]
        mean_squared_error = sum(squared_errors) / n
        root_mean_squared_error = mean_squared_error ** 0.5
        return root_mean_squared_error


class Datamining:

    def __init__(self, train_set):
        x_data, y_data = [], []
        for x, y in train_set:
            x_data.append(x)
            y_data.append(y)

        self.x_scaler = Scaler()
        x_data = self.x_scaler.fit_min(x_data)

        self.y_scaler = Scaler()
        y_data = self.y_scaler.fit_min(y_data)

        self.model = CustomNeuralNetwork(4)
        self.model.fit(x_data, y_data, epochs=2000)

    def predict(self, x):
        x = self.x_scaler.scale_min(x)
        y = self.model.predict(x)
        return self.y_scaler.restore_min(y)


def main():
    train_data = [(-1636, 17506968515.0), (783, -1922037931.0), (2233, -44552503381.0), (-1455, 12314741699.0), (1385, -10632728101.0), (1661, -18338556193.0), (-1205, 6994410449.0), (-2064, 35158582607.0), (-815, 2163384899.0), (1590, -16086308251.0), (-1261, 8015806265.0), (2051, -34523584663.0), (-261, 70915265.0), (-1672, 18688487399.0), (1275, -8295570751.0), (1416, -11362659433.0), (-935, 3266983499.0), (-2325, 50256107249.0), (-2744, 82621612247.0), (-2479, 60919769027.0), (-1068, 4869329195.0), (-2753, 83437323845.0), (343, -161769091.0), (-1399, 10946632187.0), (1518, -13998775891.0), (-729, 1548091277.0), (138, -10570111.0), (-1648, 17895099695.0), (-1568, 15413121695.0), (2514, -63574864135.0), (452, -369996805.0), (-2954, 103081623077.0), (-2421, 56742686225.0), (-2351, 51961228355.0), (-1765, 21984151649.0), (-2137, 39023051789.0), (810, -2127736351.0), (-27, 76679.0), (1273, -8256601621.0), (159, -16155355.0), (297, -105058405.0), (-1350, 9836039249.0), (1411, -11242737943.0), (-2295, 48335499899.0), (-553, 675534845.0), (2522, -64183656655.0), (1672, -18705260905.0), (-1590, 16071139649.0), (-2882, 95725788509.0), (-1560, 15178370999.0), (-1748, 21354926195.0), (-191, 27762995.0), (-997, 3961130849.0), (-2985, 106361370749.0), (-2738, 82080768845.0), (-1910, 27860549249.0), (2072, -35594798905.0), (511, -534517243.0), (-740, 1619256899.0), (-1375, 10392772499.0), (-528, 587958095.0), (2222, -43897403155.0), (569, -737854165.0), (-1681, 18991948085.0), (1912, -27970106905.0), (-1372, 10324883099.0), (-1150, 6079538249.0), (1398, -10934897371.0), (-2404, 55555615427.0), (-2816, 89298122495.0), (-519, 558387947.0), (-835, 2326643999.0), (-374, 208836737.0), (-41, 270845.0), (-2467, 60039280319.0), (-1497, 13412446349.0), (-203, 33339095.0), (462, -395087155.0), (754, -1716353575.0), (757, -1736915305.0), (2748, -83028800221.0), (-1788, 22854961595.0), (-1633, 17410820645.0), (734, -1583407555.0), (-1644, 17765091947.0), (-1002, 4021041029.0), (-243, 57219695.0), (-203, 33339095.0), (-2361, 52627176365.0), (561, -707180893.0), (359, -185461555.0), (2162, -40436895655.0), (-2071, 35517558875.0), (991, -3895920283.0), (1348, -9803282821.0), (-2061, 35005479065.0), (-179, 22846127.0), (1485, -13105659601.0), (2896, -97178051473.0), (106, -4798303.0), (-1619, 16966783247.0), (-2029, 33399941177.0), (-2590, 69475804649.0), (-1651, 17993020655.0), (-2386, 54316886765.0), (-1195, 6821681399.0), (912, -3036701905.0), (-1578, 15709947845.0), (279, -87105475.0), (113, -5810461.0), (2129, -38613579325.0)]
    dm = Datamining(train_data)

    y_data = []
    y_pred = []
    for x, y in train_data:
        y_data.append(y)
        y_pred.append(dm.predict(x))

    err = CustomNeuralNetwork.rmse(y_data, y_pred)
    print(f"error: {len(str(err))}")


if __name__ == "__main__":
    main()

