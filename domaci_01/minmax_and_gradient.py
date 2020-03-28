import numpy as np
import pandas
from math import sqrt
import sys

'''
    Reads training and test data set and creates numpy arrays
'''
def read(train_path, test_path):
    train_set = pandas.read_csv(train_path)
    x = np.asarray(train_set['size']).reshape((-1, 1))
    y = np.asarray(train_set['weight'])

    test_set = pandas.read_csv(test_path)
    test_x = np.asarray(test_set['size']).reshape((-1, 1))
    test_y = np.asarray(test_set['weight'])

    return x, y, test_x, test_y

'''
    Calculates root mean square error
'''
def calculate_rmse(actual, predicted):
    sum_error = 0.0
    for j in range(len(actual)):
        prediction_error =  actual[j] - predicted[j]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

'''
    Removes outlier data pairs from train data set
'''
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

'''
    Performs data normalisation with min max algorithm
'''
def min_max(x, y):
    v = x
    x = (v - v.min()) / (v.max() - v.min())

    v = y
    y = (v - v.min()) / (v.max() - v.min())
    return x, y

'''
    Calculates cost of position during gradient descent
'''
def cost(X, Y, theta):
    J = np.dot((np.dot(X, theta) - Y).T, (np.dot(X, theta) - Y)) / (2 * len(Y))
    return J

'''
    Fits train data and finds intercept and slope values
'''
def fit(x, y):
    x, y = remove_outliers(x, y)
    x, y = min_max(x, y)

    alpha = 0.1
    theta = np.array([[0, 0]]).T
    X = np.c_[np.ones(174), x]
    Y = np.c_[y]
    X_1 = np.c_[x].T
    num_iters = 880

    for i in range(num_iters):
        a = np.sum(theta[0] - alpha * (1 / len(Y)) * np.sum((np.dot(X, theta) - Y)))
        b = np.sum(theta[1] - alpha * (1 / len(Y)) * np.sum(np.dot(X_1, (np.dot(X, theta) - Y))))
        theta = np.array([[a], [b]])
    return theta/2

'''
    Performs linear regression formula
'''
def predict(test_x, theta):
    y_pred = theta[0] + test_x * theta[1]
    return y_pred

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Bad argument list, enter in following form:")
    #     print("python <script_name>.py <train_set_path> <test_set_path>")
    #     exit()
    #
    # x, y, test_x, test_y = read(sys.argv[1], sys.argv[2])
    x, y, test_x, test_y = read('resources/train.csv', 'resources/test_preview.csv')

    x, y = remove_outliers(x, y)
    theta = fit(x, y)
    y_pred = predict(test_x, theta)
    print(calculate_rmse(test_y, y_pred))




