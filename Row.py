import numpy as np
import csv
import json

def loadcsv(fname):
    f = open(fname)
    reader = csv.reader(f, delimiter=',')
    all_data = []
    for i, row in enumerate(reader):
        if i==0:
            continue
        all_data.append(row)
    return all_data

def loadAllDicts():
    classes = json.loads(open('data/Dictionaries/classes.json').read())
    families = json.loads(open('data/Dictionaries/families.json').read())
    items = json.loads(open('data/Dictionaries/items.json').read())
    return classes, families, items

classes, families, items = loadAllDicts()

#date in format {year:year, month:month, day:day}
def parse_date(date):
    s = date.split('-')
    return {'year': int(s[0]), 'month':int(s[1]), 'day':int(s[2])}

def getInfoByDate(date):
    info = {'train':[],'transactions':[]}
    trainInfo = loadcsv('data/smalltrain.csv')
    oilData = loadcsv('data/oil.csv')
    transactions = loadcsv('data/transactions.csv')
    for t in trainInfo:
        if str(t[0])==date:
            info['train'].append(t)
    for o in oilData:
        if str(o[0])==date:
            info['oil'] = o
            break
    for tr in transactions:
        if str(tr[0])==date:
            info['transactions'].append(tr)
    return info

#already sorted by date
def sortByStore(train, transactions, storeNum):
    trainStore = []
    for t in train:
        if t[2]==storeNum:
            trainStore.append(t)
    for tr in transactions:
        if tr[1]==storeNum:
            transStore = tr
            break
    return trainStore, transStore

#returns two variables: a row of train and a row of transactions
def trainTrans(dated, storeNum, itemNum):
    trainStore, transStore = sortByStore(dated['train'],dated['transactions'], storeNum)
    for t in trainStore:
        if t[3]==itemNum:
            return t, transStore

#data in format {family: str, class: int, perishable: 1/O}
def getItemInfo(itemNum):
    with open('data/items.csv') as items:
        reader = csv.reader(items,delimiter=',')
        data = {}
        for row in reader:
            if row[0]==itemNum:
                data['family'] = families[row[1]]
                data['class'] = classes[row[2]]
                data['perishable'] = row[3]
    return data

#rowSparse: year, month, day,family,class,perishable,storeNum,onPromotion(maybe)
def getRow(date, itemNum, storeNum):
    row = {'dense':[], 'date':date, 'itemNum': items[itemNum], 'store':storeNum, 'sparse':[]}
    dateInfo = getInfoByDate(date)
    trainInfo, transInfo = trainTrans(dateInfo, storeNum, itemNum)
    itemInfo = getItemInfo(itemNum)
    parsed = parse_date(date)
    row['sparse']+=[parsed['year'],parsed['month'],parsed['date']]
    row['sparse']+=[itemInfo['family'],itemInfo['class'],itemInfo['perishable']]
    row['dense'].append(float(dateInfo['oil'][1]))
    row['sparse'].append(storeNum)

    #note that this value doesn't always exist
    try:
        row['sparse'].append(trainInfo[5])
    except IndexError:
        row['sparse'].append(0)
