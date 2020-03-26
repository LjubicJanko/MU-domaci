import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def read_data(path):
    sizes = []
    weights = []
    with open(path) as train_data:
        csv_reader = csv.reader(train_data, delimiter=',')
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                sizes.append(int(row[0]))
                weights.append(int(row[1]))
    return np.asarray(sizes).reshape((-1, 1)), np.asarray(weights)

def fit_and_predict():
    lm = LinearRegression()
    lm.fit(train_x, train_y)

    y_pred = lm.predict(test_x)

    print('predicted weights:', y_pred, sep='\n')
    print('actual weights:', test_y, sep='\n')

    rms = sqrt(mean_squared_error(test_y, y_pred))

    plt.scatter(test_x, test_y, label='data')
    plt.plot([min(test_x), max(test_x)], [min(y_pred), max(y_pred)], label='model', color='red')

    plt.legend()
    plt.show()

    print("RMS: ", rms)

    print('theta[0] = ', lm.intercept_)
    print('theta[1] = ', lm.coef_)


train_x, train_y = read_data('resources/train.csv')
test_x, test_y = read_data('resources/test_preview.csv')
fit_and_predict()

