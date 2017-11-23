import csv
import random
with open('data/train.csv') as f:
    reader = csv.reader(f, delimiter=',')
    allrows = []
    s = random.sample(population=[i for i in range(1,100000000)], k=10000)
    print(s)
    for n, row in enumerate(reader):
        if n==0:
            allrows.append(row)
        if n in s:
            allrows.append(row)
        if n>100000000:
            break
print(allrows)
with open('data/smalltrain.csv','w') as fw:
    writer = csv.writer(fw, delimiter=',')
    for row in allrows:
        writer.writerow(row)
fw.close()