import csv
import numpy
from pprint import pprint

def parseFile(fname, size):
    fp = 'data/' + fname
    all_data = []
    with open(fp) as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if i==0:
                all_data.append(row)
            if fname=='train.csv':
                if i>=size+50000000:
                    break
                if i>50000000:
                    all_data.append(row)
            else:
                all_data.append(row)
        f.close()
    return all_data

all_files = ['holidays_events.csv', 'items.csv', 'oil.csv', 'sample_submission.csv', 'stores.csv', 'test.csv',
             'train.csv', 'transactions.csv']

def writeData(small_csv, name):
    f = open('data/practiceData/' + name, 'w')
    writer = csv.writer(f,delimiter=',', quotechar='|')
    for row in small_csv:
        writer.writerow(row)

def createPractice(size):
    for f in all_files:
        small = parseFile(f, size)
        writeData(small, 'small' + f)

createPractice(10000)
