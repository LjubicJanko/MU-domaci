import json
from collections import OrderedDict
from jsondiff import diff

class Title:
    def __init__(self, text, clickbait):
        self.text = text
        self.clickbait = clickbait

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

    with open("data.json") as spec_file:
        spec1 = json.load(spec_file, object_pairs_hook=OrderedDict)

    with open("resources/train.json") as spec_file:
        spec2 = json.load(spec_file, object_pairs_hook=OrderedDict)
    #
    # difference = {key: [o, spec2[key]] for key, o in spec1.iteritems()
    #              if key not in spec2};
    #
    # with open('difference.json', 'w') as outfile:
    #     json.dump(difference, outfile)

    print(diff(spec1, spec2))


