
with open('whole.csv', 'r') as t1, open('train.csv', 'r') as t2:
    fileone = t1.readlines()
    fileone_noWage = []
    for line in fileone:
        line_split = line.split(",")
        line_split.pop()
        fileone_noWage.append(','.join(line_split))



    filetwo = t2.readlines()
    filetwo_noWage = []
    for line in filetwo:
        line_split = line.split(",")
        line_split.pop()
        filetwo_noWage.append(','.join(line_split))

with open('ceo_test.csv', 'a') as outFile:
    i = 0
    for line in fileone_noWage:
        if line not in filetwo_noWage:
            outFile.write(fileone[i])
        i += 1