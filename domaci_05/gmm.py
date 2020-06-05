import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
import sys

import matplotlib.pyplot as plt

switcher = {
    "Africa": 1,
    "Americas": 2,
    "Asia": 3,
    "Europe": 4
}

def mapBinary(column, value):
    return 0 if column == value else 1

def encodeRegion(y):
    encoded_region = []
    for region in y:
        encoded_region.append(switcher.get(region))
    return encoded_region

def encodeOil(data):
    for index, row in data.iterrows():
        data.at[index, 'oil'] = mapBinary(row["oil"], "no")
    return data

def meanInfant(data):
    data['infant'].fillna((data['infant'].mean()), inplace=True)
    return data

def plot(x, y, x_name, y_name):
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def read(filePath, training=False):
    # read data
    data = pd.read_csv(filePath)

    # remove rows with NAN values
    # data = meanInfant(data)
    data.dropna(inplace=True)

    ''' removing outliers - line 53 and 60'''
    if training:

        # plot(data['income'], data["region"], 'income', 'region')
        data = data[(data['region'] == 'Americas') & (data['income'] < 3000) | (data['region'] != 'Americas')]
        # data = data[(data['region'] == 'Asia') & (data['income'] < 1000) | (data['region'] != 'Asia')]
        # data = data[(data['region'] == 'Africa') & (data['income'] < 2000) | (data['region'] != 'Africa')]


        # plot(data['income'], data["region"], 'income', 'region')
        # plot(data['infant'], data["region"], 'infant', 'region')
        data = data[data['infant'] < 350]
        # plot(data['infant'], data["region"], 'infant', 'region')


    x = data.loc[:, data.columns != 'region']
    y = data['region']

    # encode string values as integer
    x = encodeOil(x)
    y = encodeRegion(y)

    return x, y

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Bad argument list, enter in following form:")
    #     print("python <script_name>.py <train_set_path> <test_set_path>")
    #     exit()
    # X_train, y_train = read(sys.argv[1], True)
    # X_test, y_test  = read(sys.argv[2])

    X_train, y_train = read("./resources/train.csv", True)
    X_test, y_test = read("./resources/test_final.csv")


    g = GaussianMixture(n_components=4, random_state=2)
    g.fit(X_train, y_train)

    y_pred = g.predict(X_test)

    v_measure = v_measure_score(y_test, y_pred)
    print(v_measure)
