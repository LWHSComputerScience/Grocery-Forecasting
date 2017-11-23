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

#date in format {year:year, month:month, day:day}
def parse_date(date):
    s = date.split('-')
    return {'year': int(s[0]), 'month':int(s[1]), 'day':int(s[2])}

#final matrix will be allitems * all stores * other info *date
def getInfoByDate(date):
    trainInfo = loadcsv('data/smalltrain.csv')
    oilData = loadcsv('data/oil.csv')
    transactions = loadcsv('data/transactions.csv')
    return trainInfo, oilData, transactions

#data in format {family: str, class: int, perishable: 1/O}
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


#row has item_num, oil, family, class, perishable, transactions, year, month, day, unit sales
def createRow(item_num, train, oil, trans, date):
    row = [item_num]
    row.append(oil[0][1])

    #sum over unit sales for a given date
    all_d = []
    for r in train:
        if r[3]==item_num:
            all_d.append(r[5])
    sales = np.sum(all_d)

    #add item info
    itemInfo = getItemInfo(item_num)
    row.append(itemInfo['family'])
    row.append(itemInfo['class'])
    row.append(itemInfo['perishable'])

    #sum over transactions for a given date
    all_t = []
    for t in trans:
        all_t.append(t[2])
    row.append(np.sum(all_t))

    #date
    d = parse_date(date)
    row.append(d['year'])
    row.append(d['month'])
    row.append(d['day'])

    row.append(sales)



def finalMatrix(date):
    train, oil, trans = getInfoByDate(date)

