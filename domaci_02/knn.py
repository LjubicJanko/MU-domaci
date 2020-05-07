import numpy as np
import pandas
import scipy.spatial
import sys

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def counter(self, neighbors):
        sum = 0
        for x in range(len(neighbors)):
            response = neighbors[x]
            sum = sum + response
        return sum / self.k


    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            d = []
            votes = []
            for j in range(len(X_train)):
                dist = scipy.spatial.distance.cityblock(X_train[j], X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(y_train[j])
            ans = self.counter(votes)
            final_output.append(ans)

        return final_output

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return (predictions == y_test).sum() / len(y_test)


def mapZvanje(zvanje):
    if zvanje == 'AsstProf':
        return 0
    elif zvanje == 'Prof':
        return 1
    else:
        return 2


def mapOblast(oblast):
    if oblast == 'A':
        return 0
    else:
        return 1


def mapPol(pol):
    if pol == 'Male':
        return 0
    else:
        return 1


def mapValues(dataFrame):
    for index, row in dataFrame.iterrows():
        dataFrame.at[index, 'zvanje'] = mapZvanje(row['zvanje'])
        dataFrame.at[index, 'oblast'] = mapOblast(row['oblast'])
        # dataFrame.at[index, 'oblast'] = 0
        dataFrame.at[index, 'pol'] = mapPol(row['pol'])
    return dataFrame


def read(train_path, test_path):
    train_set = mapValues(pandas.read_csv(train_path))
    test_set = mapValues(pandas.read_csv(test_path))

    return train_set, test_set

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


'''
    Performs data normalisation with min max algorithm
'''
def min_max(y):
    v = y
    y = (v - v.min()) / (v.max() - v.min())
    return y


def normalize_data(train, test):
    cols = len(train[0])

    for col in range(cols):
        column_data = []
        for row in train:
            column_data.append(row[col])
        mean = np.mean(column_data)
        std = np.std(column_data) # standardna devijacija

        for row in train:
            val = row[col] - mean
            if val == 0:
                row[col] = val
            else:
                row[col] = val / std
        for row in test:
            val = row[col] - mean
            if val == 0:
                row[col] = val
            else:
                row[col] = val / std

    return train, test


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Bad argument list, enter in following form:")
    #     print("python <script_name>.py <train_set_path> <test_set_path>")
    #     exit()
    #
    # train_set, test_set = read(sys.argv[1], sys.argv[2])
    train_set, test_set = read("resources/train.csv", "test_skup.csv")


    X_train = train_set.iloc[:, :-1].values
    y_train = train_set.iloc[:, 5].values
    X_test = test_set.iloc[:, :-1].values
    y_test = test_set.iloc[:, 5].values

    X_train, X_test = normalize_data(X_train, X_test)
    # X_test = normalize_data(X_test)

    min = 10000000
    index = -1
    for i in range(2, 60):
        clf = KNN(i)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        rmse_val = rmse(np.array(y_test), np.array(prediction))
        if rmse_val < min:
            min = rmse_val
            index = i

    print(min)
