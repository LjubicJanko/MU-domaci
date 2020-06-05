import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import v_measure_score, f1_score
from sklearn.preprocessing import StandardScaler
import sys

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
        x.at[index, 'jobclass'] = mapBinary(row["jobclass"], "1. Industrial")
        x.at[index, 'maritl'] = switcher_maritl.get(row["maritl"])
        x.at[index, 'education'] = switcher_education.get(row["education"])
    return x


def read(filePath):
    # read data
    data = pd.read_csv(filePath)

    # remove rows with NAN values
    data.dropna(inplace=True)

    x = data.loc[:, data.columns != 'race']
    y = data['race']

    # encode string values as integer
    x = encode_x(x)
    y = encodeRace(y)

    return x, y


def get_result(true, pred):
    return v_measure_score(true, pred)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Bad argument list, enter in following form:")
        print("python <script_name>.py <train_set_path> <test_set_path>")
        exit()
    X_train, y_train = read(sys.argv[1])
    X_test, y_test  = read(sys.argv[2])

    # X_train, y_train = read("./resources/train.csv")
    # # X_test, y_test = read("./resources/test_preview.csv")
    # X_test, y_test = read("./resources/ceo_test.csv")

    scaler = StandardScaler()
    scaler.fit(X_train.astype(float))
    X_train = scaler.transform(X_train.astype(float))
    X_test = scaler.transform(X_test.astype(float))

    # Make an instance of the Model
    pca = PCA(0.99)
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    '''
        0.7936046511627906 za RandomForestClassifier(n_estimators=600, max_features=4, random_state=5)
    '''
    clf = RandomForestClassifier(n_estimators=600, max_features=4, random_state=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f1)

    # for nest in nestim:
    #     for i in range(1, 10):
    #         # clf = GradientBoostingClassifier(n_estimators=nest, random_state=i, subsample=0.8)
    #         clf = RandomForestClassifier(n_estimators=nest, max_features=4, random_state=i)
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_test)
    #         f1 = f1_score(y_test, y_pred, average='micro')
    #         print("nest: " + str(nest) + ", i: " + str(i) + ", f1: " + str(f1))
