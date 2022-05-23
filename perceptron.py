import yaml
import random
import math
import csv

class Perceptron:
    def __init__(self, path_to_file_params=None):
        if path_to_file_params:
            with open(path_to_file_params, 'r') as f:
                data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        else:
            data = None
        params = {
            'lr': data.get('lr', 0.1) if data else 0.1,
            'function': data.get('function', 'tanh') if data else 'tanh',
            'epochs': data.get('epochs', 100) if data else 100,
            'split': data.get('split', 0.2) if data else 0.2,
            'loss': data.get('loss', 'rmse') if data else 'rmse'
        }
        self.__dict__.update(params)
        random.seed(42)
    def calc_activ(self, x: float) -> float:
        if self.function == 'relu':
            res = 0 if x <= 0 else x
        elif self.function == 'sigmoid':
            res = 1 / (1 + math.exp(-x))
        elif self.function == 'tanh':
            res = math.tanh(x)
        else:
            raise ValueError('Unknown activation function')
        return res
    def calc_derivative(self, x: float) -> float:
        if self.function == 'relu':
            res = 0 if x <= 0 else 1
        elif self.function == 'sigmoid':
            res = math.exp(-x) / ((1 + math.exp(-x)) ** 2)
        elif self.function == 'tanh':
            res = 1 - math.tanh(x)**2
        else:
            raise ValueError('Unknown activation function')
        return res
    def read_data(self, path_to_file_data):
        with open(path_to_file_data, 'r') as f:
            data = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            data = list(data)
        self.data = data
    def normalize(self):
        minmax = list()
        for i in range(len(self.data[0])):
            col_values = [row[i] for row in self.data]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        for row in self.data:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        X, Y = [], []
        for line in self.data:
            X.append(line[:-1])
            Y.append(line[-1])
        self.data_X = X
        self.data_Y = Y
        self.dim = len(X[0])
        self.minmax = minmax
    def split_data(self):
        validation_size = int(self.split*len(self.data_X))
        indexes = random.choices(range(len(self.data_X)), k=validation_size)
        X_tra, Y_tra, X_val, Y_val = [], [], [], []
        for i in range(len(self.data_X)):
            if i in indexes:
                X_val.append(self.data_X[i])
                Y_val.append(self.data_Y[i])
            else:
                X_tra.append(self.data_X[i])
                Y_tra.append(self.data_Y[i])
        self.X_tra = X_tra
        self.Y_tra = Y_tra
        self.X_val = X_val
        self.Y_val = Y_val
    def calc(self, x):
        sum = 0
        for j in range(self.dim):
            sum += self.weights[j] * x[j]
        f = self.calc_activ(sum)
        return f, sum
    def calc_loss(self, y_tru, y_pred):
        if self.loss == 'rmse':
            loss = (y_tru - y_pred)**2
        if self.loss == 'mae':
            loss = abs(y_tru - y_pred)
        return loss
    def train(self):
        self.normalize()
        self.split_data()
        self.weights = [random.random() for i in range(self.dim)]
        for epoch in range(self.epochs):
            loss_train = 0
            for i in range(len(self.X_tra)):
                y_pred, sum = self.calc(self.X_tra[i])
                loss_train += self.calc_loss(self.Y_tra[i], y_pred)
                for j in range(self.dim):
                    self.weights[j] += self.lr * self.calc_derivative(sum) * \
                        (self.Y_tra[i] - y_pred) * self.X_tra[i][j]
            loss_val = 0
            Y_pred = []
            for i in range(len(self.X_val)):
                y_pred, sum = self.calc(self.X_val[i])
                Y_pred.append(y_pred)
                loss_val += self.calc_loss(self.Y_val[i], y_pred)
            print(f'Epoch no {epoch+1}: loss - {loss_train}, val_loss - {loss_val}')

        for i in range(len(self.Y_val)):
            y_val = self.Y_val[i] * (self.minmax[-1][1] - self.minmax[-1][0]) + self.minmax[-1][0]
            y_pred = Y_pred[i] * (self.minmax[-1][1] - self.minmax[-1][0]) + self.minmax[-1][0]
            print(y_val, y_pred)

if __name__ == "__main__":
    model = Perceptron('settings.yaml')
    model.read_data('perceptron-data.csv')
    model.train()
