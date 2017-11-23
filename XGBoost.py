import csv
import numpy as np
import collections

#4100 unique items
all_files = ['smallholidays_events.csv', 'smallitems.csv', 'smalloil.csv', 'smallsample_submission.csv', 'smallstores.csv',
             'smalltest.csv', 'smalltrain.csv', 'smalltransactions.csv']

def loadcsv(fname):
    f = open(fname)
    reader = csv.reader(f, delimiter=',')
    all_data = []
    for i, row in enumerate(reader):
        if i==0:
            continue
        all_data.append(row)
    return all_data

def getUniques(l):
    n = []
    for x in l:
        if x in n:
            continue
        else:
            n.append(l)
    fd = {}
    for i, item in enumerate(n):
        fd[item] = i
    return fd

#date in format year, month, day
def parse_date(date):
    return date.split('-')

#final matrix will be allitems * all stores * other info *date
def getInfoByDate(date):
    trainInfo = []
    with open('data/smalltrain.csv') as st:
        reader = csv.reader(st, delimiter=',')
        for row in reader:
            if row[0]==date:
                trainInfo.append(row)
    oilData = []
    with open('data/oil.csv') as st:
        reader = csv.reader(st, delimiter=',')
        for row in reader:
            if row[0]==date:
                oilData.append(row)
    transactions = []
    with open('data/transactions.csv') as tf:
        reader = csv.reader(st, delimiter=',')
        for row in reader:
            if row[0]==date:
                transactions.append(row)
    return trainInfo, oilData, transactions

def getItemInfo(itemNum):
    with open('data/items.csv') as items:
        reader = csv.reader(items,delimiter=',')
        data = {}
        for row in reader:
            if row[0]==itemNum:
                data['family'] = row[1]
                data['class'] = row[2]
                data['perishable'] = row[3]
    return data

def finalMatrix(date):
    train, oil, trans = getInfoByDate(date)

