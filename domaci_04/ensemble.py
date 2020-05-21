import pandas as pd
from sklearn.ensemble import AdaBoostClassifier  # Boosting Algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

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


def read(filePath):
    # read data
    data = pd.read_csv(filePath)
    # remove rows with NAN values
    data.dropna(inplace=True)

    # extract speed column into y, and other columns into x
    x = data.iloc[:, 1:]
    y = data['speed']

    # encode string values as integer
    x = encodeData(x)
    y = mapSpeed(y)
    return x, y


if __name__ == '__main__':
    X_train, y_train = read("./resources/train.csv")
    X_test, y_test = read("./resources/test_preview.csv")
    # X_test, y_test = read("./resources/test.csv")

    cart = DecisionTreeClassifier()
    num_trees = 25

    # Create classification model for bagging
    model = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, learning_rate=0.1)

    # Train Classification model
    model.fit(X_train, y_train)

    # Test trained model over test set
    pred_label = model.predict(X_test)

    print(f1_score(y_test, pred_label, average='micro'))