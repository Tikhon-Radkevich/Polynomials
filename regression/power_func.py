from sklearn.model_selection import train_test_split


class Scaler:
    def __init__(self):
        self.max = 0

    def fit(self, x):
        self.max = max(x)
        return [self.scale(i) for i in x]

    def scale(self, x):
        return x / self.max

    def restore(self, val):
        return val * self.max


class PowerFunc:
    def __init__(self, max_degree, learning_rate):
        self.neuron = Neuron(max_degree, learning_rate)

    def predict(self, x_data):
        return [self.neuron.forward(x) for x in x_data]

    def fit(self, x_data, y_data):
        for x, y in zip(x_data, y_data):
            output = self.neuron.forward(x)
            self.neuron.backward(x, y, output)

    @staticmethod
    def rmse(y_true, y_pred):
        n = len(y_true)
        squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
        mse = sum(squared_errors) / n
        rmse = mse ** 0.5
        return rmse


class Neuron:
    def __init__(self, max_degree, learning_rate):
        self.learning_rate = learning_rate
        self.weights = [0 for _ in range(max_degree + 1)]
        self.bias = 0

    def backward(self, x, y, output):
        error = y - output
        self.bias += error * self.learning_rate
        for degree in range(len(self.weights)):
            x_degree = x ** degree
            self.weights[degree] += x_degree * error * self.learning_rate

    def forward(self, x):
        output = sum(weight * (x ** degree) for degree, weight in enumerate(self.weights))
        return output


def main():
    train_data = [(-1636, 17506968515.0), (783, -1922037931.0), (2233, -44552503381.0), (-1455, 12314741699.0), (1385, -10632728101.0), (1661, -18338556193.0), (-1205, 6994410449.0), (-2064, 35158582607.0), (-815, 2163384899.0), (1590, -16086308251.0), (-1261, 8015806265.0), (2051, -34523584663.0), (-261, 70915265.0), (-1672, 18688487399.0), (1275, -8295570751.0), (1416, -11362659433.0), (-935, 3266983499.0), (-2325, 50256107249.0), (-2744, 82621612247.0), (-2479, 60919769027.0), (-1068, 4869329195.0), (-2753, 83437323845.0), (343, -161769091.0), (-1399, 10946632187.0), (1518, -13998775891.0), (-729, 1548091277.0), (138, -10570111.0), (-1648, 17895099695.0), (-1568, 15413121695.0), (2514, -63574864135.0), (452, -369996805.0), (-2954, 103081623077.0), (-2421, 56742686225.0), (-2351, 51961228355.0), (-1765, 21984151649.0), (-2137, 39023051789.0), (810, -2127736351.0), (-27, 76679.0), (1273, -8256601621.0), (159, -16155355.0), (297, -105058405.0), (-1350, 9836039249.0), (1411, -11242737943.0), (-2295, 48335499899.0), (-553, 675534845.0), (2522, -64183656655.0), (1672, -18705260905.0), (-1590, 16071139649.0), (-2882, 95725788509.0), (-1560, 15178370999.0), (-1748, 21354926195.0), (-191, 27762995.0), (-997, 3961130849.0), (-2985, 106361370749.0), (-2738, 82080768845.0), (-1910, 27860549249.0), (2072, -35594798905.0), (511, -534517243.0), (-740, 1619256899.0), (-1375, 10392772499.0), (-528, 587958095.0), (2222, -43897403155.0), (569, -737854165.0), (-1681, 18991948085.0), (1912, -27970106905.0), (-1372, 10324883099.0), (-1150, 6079538249.0), (1398, -10934897371.0), (-2404, 55555615427.0), (-2816, 89298122495.0), (-519, 558387947.0), (-835, 2326643999.0), (-374, 208836737.0), (-41, 270845.0), (-2467, 60039280319.0), (-1497, 13412446349.0), (-203, 33339095.0), (462, -395087155.0), (754, -1716353575.0), (757, -1736915305.0), (2748, -83028800221.0), (-1788, 22854961595.0), (-1633, 17410820645.0), (734, -1583407555.0), (-1644, 17765091947.0), (-1002, 4021041029.0), (-243, 57219695.0), (-203, 33339095.0), (-2361, 52627176365.0), (561, -707180893.0), (359, -185461555.0), (2162, -40436895655.0), (-2071, 35517558875.0), (991, -3895920283.0), (1348, -9803282821.0), (-2061, 35005479065.0), (-179, 22846127.0), (1485, -13105659601.0), (2896, -97178051473.0), (106, -4798303.0), (-1619, 16966783247.0), (-2029, 33399941177.0), (-2590, 69475804649.0), (-1651, 17993020655.0), (-2386, 54316886765.0), (-1195, 6821681399.0), (912, -3036701905.0), (-1578, 15709947845.0), (279, -87105475.0), (113, -5810461.0), (2129, -38613579325.0)]

    x_data = [point[0] for point in train_data]
    y_data = [point[1] for point in train_data]
    y_max = max(y_data)
    x_max = max(x_data)
    x_data = list(map(lambda val: val / x_max, x_data))
    y_data = list(map(lambda val: val / y_max, y_data))

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    max_degree = 3  # Set the maximum degree for your power functions
    learning_rate = 0.6
    pf = PowerFunc(max_degree, learning_rate)
    for _ in range(20):
        pf.fit(x_train, y_train)

    y_pred = pf.predict(x_test)
    y_pred = [y*y_max for y in y_pred]
    y_test = [y*y_max for y in y_test]
    err = pf.rmse(y_test, y_pred)
    print(pf.neuron.weights)
    print(err/1000000)
    print(f"error: {len(str(err))}")


if __name__ == "__main__":
    main()
