import pandas as pd

# import csv
# with open("resources/Kag_Airbag.csv","r") as source:
#     rdr= csv.reader( source )
#     with open("result.csv","w") as result:
#         wtr= csv.writer( result )
#         for r in rdr:
#             wtr.writerow( (r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14]) )


# f = open("result.csv", 'r')
# for line in f.readlines():
#         if line.strip():
#             print(line[:-1])
# #
with open('result.csv', 'r') as t1, open('resources/train.csv', 'r') as t2:
    fileone = t1.readlines()
    filetwo = t2.readlines()

with open('resources/test.csv', 'w') as outFile:
    for line in fileone:
        if line not in filetwo:
            outFile.write(line)