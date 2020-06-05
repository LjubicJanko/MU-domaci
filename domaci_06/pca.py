import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
import sys

import matplotlib.pyplot as plt

switcher_maritl = {
    "1. Never Married": 1,
    "2. Married": 2,
    "3. Widowed": 3,
    "4. Divorced": 4,
    "5. Separated": 5
}

switcher_education = {
    "1. < HS Grad": 1,
    "2. HS Grad": 2,
    "3. Some College": 3,
    "4. College Grad": 4,
    "5. Advanced Degree": 5
}
switcher_race = {
    "1. White": 1,
    "2. Black": 2,
    "3. Asian": 3,
    "4. Other": 4
}

def encodeRace(data):
    encoded_race = []
    for race in data:
        encoded_race.append(switcher_race.get(race))
    return encoded_race

def mapBinary(column, value):
    return 0 if column == value else 1
'''
    health: 1. <=Good
    jobclass: 1. Industrial
    health_ins: 1. Yes
'''
# def encodeBooleanColumn(data, columnName, cellValue):
#     for index, row in data.iterrows():
#         data.at[index, columnName] = mapBinary(row[columnName], cellValue)
#     return data

def plot(x, y, x_name, y_name):
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

def encode_x(x):
    for index, row in x.iterrows():
        x.at[index, 'health'] = mapBinary(row["health"], "1. <= Good")
        x.at[index, 'health_ins'] = mapBinary(row["health_ins"], "1. Yes")
        x.at[index, 'jobclass'] = mapBinary(row["health_ins"], "1. Industrial")
        x.at[index, 'maritl'] = switcher_maritl.get(row["maritl"])
        x.at[index, 'education'] = switcher_education.get(row["education"])
    return x


def read(filePath, training=False):
    # read data
    data = pd.read_csv(filePath)

    # remove rows with NAN values
    # data = meanInfant(data)
    data.dropna(inplace=True)

    x = data.loc[:, data.columns != 'race']
    y = data['race']

    # encode string values as integer
    x = encode_x(x)
    y = encodeRace(y)

    return x, y

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Bad argument list, enter in following form:")
    #     print("python <script_name>.py <train_set_path> <test_set_path>")
    #     exit()
    # X_train, y_train = read(sys.argv[1], True)
    # X_test, y_test  = read(sys.argv[2])

    X_train, y_train = read("./resources/train.csv", True)
    X_test, y_test = read("./resources/test_preview.csv")
    print(X_train.head())
    print(y_train)
