import csv

# with open('infmort.csv', 'r') as infile, open('test.csv', 'a') as outfile:
#     # output dict needs a list for new column ordering
#     fieldnames = ['income', 'infant', 'region', 'oil']
#     # income,infant,region,oil
#     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#     # reorder the header first
#     writer.writeheader()
#     for row in csv.DictReader(infile):
#         # writes the reordered rows to the new file
#         writer.writerow(row)


# f = open("test.csv", 'r')
# for line in f.readlines():
#         if line.strip():
#             print(line[:-1])


with open('test.csv', 'r') as t1, open('train.csv', 'r') as t2:
    fileone = t1.readlines()
    filetwo = t2.readlines()

with open('test_final.csv', 'a') as outFile:
    for line in fileone:
        if line not in filetwo:
            outFile.write(line)
