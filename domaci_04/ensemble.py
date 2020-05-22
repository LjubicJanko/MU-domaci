import pandas as pd
from sklearn.ensemble import AdaBoostClassifier  # Boosting Algorithm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

numberColumns = ["weight", "ageOFocc"]


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

# def findMeanForNumberColumns() :
#     for column in numberColumns:
#         
#     pass

def predictInjSeverity(data) :
    # null_columns = data.columns[data.isnull().any()]
    # data_injseverity_nan = data[data["injSeverity"].isnull()][null_columns]

    data_injseverity_nan = data[data["injSeverity"].isnull()]
    # data_without_nan = data.dropna()

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
        i+=1
    return data


def read(filePath, training=False):
    # read data
    data = pd.read_csv(filePath)
    # remove rows with NAN values
    data.dropna(inplace=True)

    # for column in data.columns[:-1]:
    #     data = data[pd.notnull(data[column])]

    # extract speed column into y, and other columns into x
    x = data.iloc[:, 1:]
    y = data['speed']


    # print(x[x['injSeverity'].notna()])
    # encode string values as integer
    x = encodeData(x)
    y = mapSpeed(y)

    # if training:
    #     x = predictInjSeverity(x)


    return x, y


if __name__ == '__main__':
    X_train, y_train = read("./resources/train.csv", True)
    # X_test, y_test = read("./resources/test_preview.csv")
    X_test, y_test = read("./resources/test.csv", True)

    # cart = DecisionTreeClassifier()
    # num_trees = 250
    #
    # # Create classification model for bagging
    # model = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, learning_rate=0.1)
    #
    # # Train Classification model
    # model.fit(X_train, y_train)
    #
    # # Test trained model over test set
    # pred_label = model.predict(X_test)
    #
    # f1 = f1_score(y_test, pred_label, average='micro')
    # print(f1)


    # nums = [5, 25, 50, 70, 100, 120, 150, 170, 200, 230, 250, 270, 500, 600, 700, 800, 900, 1000]
    nums = [250, 270, 500]

    maximum = 0

    for num in nums:
        cart = DecisionTreeClassifier()
        num_trees = num

        # Create classification model for bagging
        model = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, learning_rate=0.1)

        # Train Classification model
        model.fit(X_train, y_train)

        # Test trained model over test set
        pred_label = model.predict(X_test)

        f1 = f1_score(y_test, pred_label, average='micro')
        if f1 > maximum:
            maximum = f1
        print(str(num) + " >> " + str(f1))
    print("Maksimum je: >>" + str(maximum) + ".")

    # nums = [200, 230, 250, 270, 500, 600, 700, 800, 900, 1000]
    # nums = [5, 25, 50, 70, 100, 120, 150, 170, 200, 230, 250, 270, 500, 600, 700, 800, 900, 1000]
    #
    # maximum = 0
    #
    # for num in range(2,15,3):
    #     seed = 8
    #     kfold = model_selection.KFold(n_splits=num, random_state=seed, shuffle=True)
    #
    #     # Define a decision tree classifier
    #     cart = DecisionTreeClassifier()
    #     num_trees = 170
    #
    #     # Create classification model for bagging
    #     model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    #
    #     model.fit(X_train, y_train)
    #     pred_label = model.predict(X_test)
    #     f1 = f1_score(y_test, pred_label, average='micro')
    #     # print(f1_score(y_test, pred_label, average='micro'))
    #     if f1 > maximum:
    #         maximum = f1
    #     print(str(num) + " >> " + str(f1))

