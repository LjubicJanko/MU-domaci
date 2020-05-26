import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score

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


def read(filePath, training=False):
    # read data
    data = pd.read_csv(filePath)

    # plt.figure(figsize=(7, 7))
    # plt.scatter(data["infant"], data["region"])
    # plt.xlabel('infant')
    # plt.ylabel('region')
    # plt.title('Infant by region')
    # plt.show()

    # plt.figure(figsize=(7, 7))
    # plt.scatter(data["income"], data["region"])
    # plt.xlabel('income')
    # plt.ylabel('region')
    # plt.title('Income by region')
    # plt.show()

    # plt.figure(figsize=(7, 7))
    # plt.scatter(data["oil"], data["region"])
    # plt.xlabel('oil')
    # plt.ylabel('region')
    # plt.title('Data Distribution')
    # plt.show()


    # remove rows with NAN values

    data = meanInfant(data)
    data.dropna(inplace=True)

    x = data.loc[:, data.columns != 'region']
    y = data['region']

    # encode string values as integer
    x = encodeOil(x)
    y = encodeRegion(y)

    return x, y

if __name__ == '__main__':
    X_train, y_train = read("./resources/train.csv")
    # X_test, y_test = read("./resources/test_preview.csv")
    X_test, y_test = read("./resources/test_final.csv")

    maximum = 0
    max_n = 0
    iteration = 0
    for n in range(2, 80):
        for i in range(10):
            g = GaussianMixture(n_components=n)
            g.fit(X_train, y_train)

            y_pred = g.predict(X_test)
            v_measure = v_measure_score(y_test, y_pred)
            if v_measure > maximum:
                iteration = i
                maximum = v_measure
                max_n = n

    print("N components: " + str(max_n) + " Iteration: " + str(iteration))
    print("#"*20)
    print(maximum)