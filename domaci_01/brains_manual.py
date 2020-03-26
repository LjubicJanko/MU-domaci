import numpy as np
import pandas
from math import sqrt


train_set = pandas.read_csv('resources/train.csv')
x = np.asarray(train_set['size']).reshape((-1, 1))
y = np.asarray(train_set['weight'])

test_set = pandas.read_csv('resources/test_preview.csv')
test_x = np.asarray(test_set['size']).reshape((-1, 1))
test_y = np.asarray(test_set['weight'])


x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0
den = 0
for i in range(len(x)):
    num += (x[i] - x_mean)*(y[i] - y_mean)
    den += (x[i] - x_mean)**2
m = num / den
c = y_mean - m*x_mean

y_pred = c + m*test_x

def get_mse(actual, predicted):
    sum_error = 0.0
    for j in range(len(actual)):
        prediction_error =  actual[j] - predicted[j]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

print(get_mse(test_y, y_pred))


# TODO: remove this import
# import matplotlib.pyplot as plt
#
# plt.scatter(test_x, test_y,label='data')
# plt.plot([min(test_x), max(test_x)], [min(y_pred), max(y_pred)], label='predicted',color='red')
# plt.plot([min(test_x), max(test_x)], [min(test_y), max(test_y)],'g-.' ,label='actual')
#
#
# plt.title('Brain SIZE-WEIGHT relation')
# plt.legend()
# plt.show()