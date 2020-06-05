from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sys
import json

def text_preprocessing(fileName):
    naslovi = []
    clickbates = []

    with open(fileName) as f:
        data = json.load(f)
        for primer in data:
            naslovi.append(primer['text'].lower())
            clickbates.append(primer['clickbait'])

    return naslovi, clickbates

def vectorisation(training, test):
    vectorizer = TfidfVectorizer(stop_words="english")
    vector_training = vectorizer.fit_transform(training)
    vector_test = vectorizer.transform(test)
    return vector_training.toarray(), vector_test.toarray()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Bad argument list, enter in following form:")
        print("python <script_name>.py <train_set_path> <test_set_path>")
        exit()
    X_train, Y_train = text_preprocessing(sys.argv[1])
    X_test, Y_test  = text_preprocessing(sys.argv[2])

    # X_train, Y_train = text_preprocessing('resources/train.json')
    # X_test, Y_test  = text_preprocessing('resources/z3_test.json')

    X_train, X_test = vectorisation(X_train, X_test)

    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, Y_train)

    Y_Pred = classifier.predict(X_test)

    print(accuracy_score(Y_test, Y_Pred))