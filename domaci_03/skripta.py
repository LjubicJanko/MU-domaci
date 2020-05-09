import json
from collections import OrderedDict
from jsondiff import diff

class Title:
    def __init__(self, text, clickbait):
        self.text = text
        self.clickbait = clickbait


def text_preprocessing(fileName):
    naslovi = []
    clickbates = []
    with open(fileName) as train_json:
        titles = train_json.read().split("},{")
        first_line = True
        for title in titles:
            # removing keyword
            if first_line:
                title = title[15:]
                first_line = False
            else:
                title = title[13:]
            clickbait = title[0]
            naslov = title[11:-1]
            naslovi.append(naslov)
            clickbates.append(clickbait)

    naslovi[-1] = naslovi[-1][:-2]

    # for naslov in naslovi:
    #     print(naslov)
    # for clickbait in clickbates:
    #     print(clickbait)

    return naslovi, clickbates


if __name__ == '__main__':
    # f = open("sve.txt", "r", encoding="utf8")
    # data = {}
    # data['titles'] = []
    # for line in f.readlines():
    #     # print(line[-2])
    #     clickbate = line[-2]
    #     title = line[:-3]
    #     data['titles'].append({
    #         'clickbate': clickbate,
    #         'text': title
    #     })

    # with open("data.json") as spec_file:
    #     spec1 = json.load(spec_file, object_pairs_hook=OrderedDict)
    #
    # with open("resources/train.json") as spec_file:
    #     spec2 = json.load(spec_file, object_pairs_hook=OrderedDict)
    #
    # difference = {key: [o, spec2[key]] for key, o in spec1.iteritems()
    #              if key not in spec2};
    #
    # with open('difference.json', 'w') as outfile:
    #     json.dump(difference, outfile)

    # print(diff(spec1, spec2))

    X_data, Y_data = text_preprocessing('data.json')
    # X_test, Y_test = text_preprocessing('resources/preview.json')
    X_train, Y_train= text_preprocessing('resources/train.json')



    with open('whole_test.json', 'w') as outfile:
        i = 0
        for title in X_data:
            if title not in X_train:
                obj_to_write = '{"clickbait":"' + Y_data[i] + '","text":"' + title + '"},'
                # print(obj_to_write)
                outfile.write(obj_to_write)
            i += 1
    outfile.close()






# import json
# f = open("sve.txt", "r", encoding="utf8")
# data = {}
# data['titles'] = []
# for line in f.readlines():
#     # print(line[-2])
#     clickbate = line[-2]
#     title = line[:-3]
#     data['titles'].append({
#         'clickbait': clickbate,
#         'text': title
#     })
#
#
#
# with open('data.json', 'w') as outfile:
#     json.dump(data, outfile)
