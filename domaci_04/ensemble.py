from idlelib import tree

import pandas as pd
from mlxtend.classifier import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
    GradientBoostingClassifier  # Boosting Algorithm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

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
    from time import time

    start_time = time()

    X_train, y_train = read("./resources/train.csv", True)
    # X_test, y_test = read("./resources/test_preview.csv")
    # X_test, y_test = read("./resources/test.csv", True)
    X_test, y_test = read("./resources/z4_test.csv")

    # nums = [5, 25, 50, 70, 100, 120, 150, 170, 200, 230, 250, 270, 500, 600, 700, 800, 900, 1000]
    # maximum = 0
    #
    # for num in range(5):
    #     cart = DecisionTreeClassifier(max_depth=13)
    #     # cart = SVC(probability=True)
    #     num_trees = 250
    #
    #     # Create classification model for bagging
    #     model = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, learning_rate=0.1)
    #
    #     # Train Classification model
    #     model.fit(X_train, y_train)
    #
    #     # Test trained model over test set
    #     pred_label = model.predict(X_test)
    #
    #
    #
    #     f1 = f1_score(y_test, pred_label, average='micro')
    #     if f1 > maximum:
    #         maximum = f1
    #     print(str(num) + " >> " + str(f1))
    # print("Maksimum je: >> " + str(maximum))



    nestim = [100, 200, 250, 300, 500, 700, 1000]

    for nest in nestim:
        '''
            test: 0.5551257253384912 - n_estimators = 100
            test_preview: 0.5906040268456376 - n_estimators = 100
        '''
        clf = GradientBoostingClassifier(n_estimators=nest)
        '''
            test: 0.5524661508704062 - n_estimators = 100, max_features = 13
            test_preview: 0.5704697986577181 - n_estimators = 200, max_features = 13
        '''
        # clf = RandomForestClassifier(n_estimators=nest, max_features=13, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='micro')
        print(f1)

    end_time = time()
    seconds_elapsed = end_time - start_time

    print(str(seconds_elapsed))


