import json
import csv
import numpy as np

"""Purpose of this file is to generate dictionaries for one-hot encoding various pieces of data. Data dealt with here
is:
item numbers
grocery family
grocery class

Note: dates are currently represented as dense representations but they could be switched over to sparse
"""

def loadcsv(fname):
    f = open(fname)
    reader = csv.reader(f, delimiter=',')
    all_data = []
    for i, row in enumerate(reader):
        if i==0:
            continue
        all_data.append(row)
    return all_data

def itemsDict():
    items = loadcsv('data/items.csv')[1:]
    nums = np.array(items)[:,0]
    uniques = list(set(nums))
    fd = {}
    for i, u in enumerate(uniques):
        fd[int(u)] = i
    with open('data/Dictionaries/items.json','w') as wf:
        wf.write(json.dumps(fd))

def groceryDict():
    items = loadcsv('data/items.csv')[1:]
    fams = np.array(items)[:,1]
    uniques = list(set(fams))
    fd = {}
    for i, u in enumerate(uniques):
        fd[u] = i
    with open('data/Dictionaries/families.json','w') as wf:
        wf.write(json.dumps(fd))

def groceryClass():
    items = loadcsv('data/items.csv')[1:]
    classes = np.array(items)[:,2]
    uniques = list(set(classes))
    fd = {}
    for i, u in enumerate(uniques):
        fd[u] = i
    with open('data/Dictionaries/classes.json','w') as wf:
        wf.write(json.dumps(fd))

itemsDict()
groceryClass()
groceryDict()