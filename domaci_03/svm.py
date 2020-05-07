const_interpunction = [',', '.', ';', '-', '?', '!', ':', '|', '_', '@', '~', '#', '^', '(', ')', '{', '}', '[', ']']

const_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                   "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                   'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                   'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                   'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                   'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                   'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                   'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                   'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                   'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                   'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
                   "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
                   "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                   "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                   'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                   'wouldn', "wouldn't"]


class Title:
    def __init__(self, text, clickbait):
        self.text = text
        self.clickbait = clickbait

'''
    Method finds occurrences of single character in string
'''
def find_char_occurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

'''
    Method finds occurrences of substring in string
'''
def find_substring_occurrences(text, substring):
    return [i for i in range(len(text)) if text.startswith(substring, i)]


'''
Method used to remove interpunction signs from const_interpunction,
but also to remove sign ' in places where it is used as quotation mark
'''


def remove_interpunction(text):

    # removing signs from const_interpunction list
    for interpunction in const_interpunction:
        text = text.replace(interpunction, ' ')

    # removing ' character
    c = "'"
    occurences = find_char_occurrences(text, c)
    for position in occurences:
        # if ' is appearing as first or as last character in string
        if (position == 0) or (position + 1 == len(text)):
            text = text[:position:] + ' ' + text[position + 1::]
        else:
            # if somewhere around ' is an space, before or after
            if (text[position - 1] == ' ') or (text[position + 1] == ' '):
                text = text[:position:] + ' ' + text[position + 1::]

    return text

'''
    Method used to remove stopwords from text
'''
def remove_stopwords(text):
    for stopword in const_stopwords:
        if stopword in text:
            occurences = find_substring_occurrences(text, stopword)

            # going through every start index of every occurrence of stopword in text
            for position in occurences:
                remove = False
                if position == 0:
                    # if it is the first word in string
                    if text[position + len(stopword)] == ' ':
                        remove = True
                elif position + len(stopword) == len(text):
                    # if it is the last word in string
                    if text[position - 1] == ' ':
                        remove = True
                else:
                    # if it is not part of another word
                    if (text[position - 1] == ' ') and (text[position + len(stopword)] == ' '):
                        remove = True

                if remove:
                    text = text[:position:] + text[position + len(stopword)::]

    return text

def text_preprocessing(fileName):
    with open(fileName) as train_json:
        titles = train_json.read().split("},{")
        first_line = True
        title_objects = []
        for title in titles:
            # removing keyword
            if first_line:
                title = title[14:]
                first_line = False
            else:
                title = title[12:]

            clickbait = title[0]

            lower_title_text = title[10:-1].lower()
            text = remove_stopwords(lower_title_text)
            text = remove_interpunction(text)
            title_objects.append(Title(text, clickbait))

        title_objects[-1].text = title_objects[-1].text[:-2]

    # for t in title_objects:
        # print(t.clickbait)
        # print(t.text)
    return title_objects


if __name__ == '__main__':
    training_titles = text_preprocessing('resources/train.json')
    test_titles = text_preprocessing('resources/preview.json')