from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def text_preprocessing(fileName):
    naslovi = []
    clickbates = []
    with open(fileName) as train_json:
        titles = train_json.read().split("},{")
        first_line = True
        for title in titles:
            # removing keyword
            if first_line:
                title = title[14:]
                first_line = False
            else:
                title = title[12:]
            clickbait = title[0]

            naslov = title[10:-1].lower()

            naslovi.append(naslov)
            clickbates.append(clickbait)

    naslovi[-1] = naslovi[-1][:-2]
    return naslovi, clickbates

def vectorisation(training, test):
    vectorizer = TfidfVectorizer()
    vector_training = vectorizer.fit_transform(training)
    vector_test = vectorizer.transform(test)
    return vector_training.toarray(), vector_test.toarray()

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Bad argument list, enter in following form:")
    #     print("python <script_name>.py <train_set_path> <test_set_path>")
    #     exit()
    # X_train, Y_train = text_preprocessing(sys.argv[1])
    # X_test, Y_test  = text_preprocessing(sys.argv[2])

    X_train, Y_train = text_preprocessing('resources/train.json')
    # X_test, Y_test  = text_preprocessing('resources/preview.json')
    X_test, Y_test  = text_preprocessing('whole_test.json')

    X_train, X_test = vectorisation(X_train, X_test)


    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, Y_train)

    Y_Pred = classifier.predict(X_test)

    print(accuracy_score(Y_test, Y_Pred))












