import pandas as pd


def mapBinary(column, value):
    return 0 if column == value else 1
def mapAbcat(abcat):
    if abcat == "unavail":
        return 0
    elif abcat == "deploy":
        return 1
    else:
        return 2


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

    x = encodeData(x)

    return x, y


if __name__ == '__main__':
    X_train, Y_train = read("./resources/train.csv")
    # X_test, Y_test = read("./resources/test_preview.csv")
