import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier  # Boosting Algorithm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
import sys

switcher = {
    "1-9km/h": 1,
    "10-24": 2,
    "25-39": 3,
    "40-54": 4,
    "55+": 5
}


def mapBinary(column, value):
    return 0 if column == value else 1


def mapAbcat(abcat):
    if abcat == "unavail":
        return 0
    elif abcat == "deploy":
        return 1
    else:
        return 2


def mapSpeed(y):
    encoded_speed = []
    for speed in y:
        encoded_speed.append(switcher.get(speed))
    return encoded_speed


def encodeData(data):
    for index, row in data.iterrows():
        data.at[index, 'dead'] = mapBinary(row["dead"], "dead")
        data.at[index, 'airbag'] = mapBinary(row["airbag"], "airbag")
        data.at[index, 'seatbelt'] = mapBinary(row["seatbelt"], "belted")
        data.at[index, 'sex'] = mapBinary(row["sex"], "m")
        data.at[index, 'abcat'] = mapAbcat(row["abcat"])
        data.at[index, 'occRole'] = mapBinary(row["occRole"], "driver")
    return data


def predictInjSeverity(data):
    data_injseverity_nan = data[data["injSeverity"].isnull()]

    data_injeseverity_exists = data[data['injSeverity'].notna()]
    data_injeseverity_exists = data_injeseverity_exists.dropna()

    Y_train = data_injeseverity_exists['injSeverity']
    X_train = data_injeseverity_exists[data_injeseverity_exists.columns[:-1]]

    X_test = data_injseverity_nan[data_injseverity_nan.columns[:-1]]

    linreg = LinearRegression()
    linreg.fit(X_train, Y_train)

    y_predicted = linreg.predict(X_test)
    y_rounded = [int(round(x)) for x in y_predicted]

    i = 0
    for index, row in data_injseverity_nan.iterrows():
        data.at[index, 'injSeverity'] = y_rounded[i]
        i += 1
    return data


def meanAgeOfOcc(data):
    data['ageOFocc'].fillna((data['ageOFocc'].mean()), inplace=True)
    return data


def read(filePath, training=False):
    # read data
    data = pd.read_csv(filePath)

    # remove rows with NAN values
    # data.dropna(inplace=True)

    if training:
        data = meanAgeOfOcc(data)

    for column in data.columns[:-1]:
        data = data[pd.notnull(data[column])]

    # extract speed column into y, and other columns into x
    x = data.iloc[:, 1:]
    y = data['speed']

    # encode string values as integer
    x = encodeData(x)
    y = mapSpeed(y)

    if training:
        x = predictInjSeverity(x)

    return x, y


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Bad argument list, enter in following form:")
    #     print("python <script_name>.py <train_set_path> <test_set_path>")
    #     exit()
    # X_train, y_train = read(sys.argv[1], True)
    # X_test, y_test  = read(sys.argv[2])

    X_train, y_train = read("./resources/train.csv", True)
    X_test, y_test = read("./resources/z4_test.csv")

    clf = GradientBoostingClassifier(n_estimators=100)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f1)
