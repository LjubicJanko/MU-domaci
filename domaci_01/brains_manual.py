import numpy as np
import pandas
from math import sqrt
import sys

def read(train_path, test_path):
    train_set = pandas.read_csv(train_path)
    x = np.asarray(train_set['size']).reshape((-1, 1))
    y = np.asarray(train_set['weight'])

    test_set = pandas.read_csv(test_path)
    test_x = np.asarray(test_set['size']).reshape((-1, 1))
    test_y = np.asarray(test_set['weight'])

    return x, y, test_x, test_y

def fit(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = 0
    den = 0
    for i in range(len(x)):
        num += (x[i] - x_mean)*(y[i] - y_mean)
        den += (x[i] - x_mean)**2
    m = num / den
    c = y_mean - m*x_mean

    return m, c

def predict(intercept, slope, test_x):
    return intercept + slope * test_x

def calculate_rmse(actual, predicted):
    sum_error = 0.0
    for j in range(len(actual)):
        prediction_error =  actual[j] - predicted[j]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


def remove_outliers(x, y):
    cleared_x = []
    cleared_y = []
    for i in range(len(x)):
        if x[i] < 3200 and y[i] > 1400:
            continue
        elif x[i] > 4300 and y[i] <= 1300:
            continue
        cleared_x.append(x[i])
        cleared_y.append(y[i])
    return np.asarray(cleared_x).reshape((-1, 1)), np.asarray(cleared_y)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Bad argument list, enter in following form:")
        print("python <script_name>.py <train_set_path> <test_set_path>")
        exit()

    x, y, test_x, test_y = read(sys.argv[1], sys.argv[2])
    # x, y, test_x, test_y = read('resources/train.csv', 'resources/test_preview.csv')
    x, y = remove_outliers(x, y)
    slope, intercept = fit(x, y)
    y_pred = predict(intercept, slope, test_x)
    print(calculate_rmse(test_y, y_pred))

