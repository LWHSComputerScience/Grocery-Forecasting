import numpy as np
import tensorflow as tf
import pandas as pd
from os import listdir
from os.path import isfile, join
import json
import seaborn as sns
from multiprocessing import Pool


# classes = pd.read_json("data/Dictionaries/classes.json")


def buildDataSet(mainTrain):
    # mainTrain = mainTrain.loc[mainTrain['store_nbr'] == 39]
    oil = pd.read_pickle("data/temp/oil.pickle")
    stores = pd.read_pickle("data/temp/stores.pickle")
    # families = pd.read_json("data/Dictionaries/families.json")
    # items2 = pd.read_json("data/Dictionaries/items.json")

    # holidays_events = pd.read_pickle("data/temp/holiday_events.pickle")
    transactions = pd.read_pickle("data/temp/transactions.pickle")
    items = pd.read_pickle("data/temp/items.pickle")
    mainTrain = pd.merge(left=mainTrain, right=stores, how="left", left_on="store_nbr", right_on="store_nbr")
    mainTrain = pd.merge(left=mainTrain, right=transactions, how="left", left_on=["date", "store_nbr"], right_on=["date", "store_nbr"])
    mainTrain = pd.merge(left=mainTrain, right=oil, how="left", left_on=["date"], right_on=["date"])
    mainTrain = pd.merge(left=mainTrain, right=items, how="left", left_on=["item_nbr"], right_on=["item_nbr"])
    mainTrain["date"] = mainTrain["date"].apply(lambda x: pd.to_datetime(x))
    mainTrain["month"] = mainTrain["date"].apply(lambda x: x.month)
    mainTrain["year"] = mainTrain["date"].apply(lambda x: x.year)
    mainTrain["day"] = mainTrain["date"].apply(lambda x: x.day)
    # mainTrain["items"] = mainTrain["items"].apply(lambda x: items2[x])
    # mainTrain["family"] = mainTrain["family"].apply(lambda x: families[x])
    # mainTrain["class"] = mainTrain["class"].apply(lambda x: classes[x])
    return mainTrain

print("parrrl")
num_partitions = 16 #number of partitions to split dataframe
num_cores = 16 #number of cores on your machine
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

if __name__ == '__main__':
    #only uncomment if first time running dataprep
    # oil = pd.read_csv("data/oil.csv")
    # stores = pd.read_csv("data/stores.csv")
    # holidays_events = pd.read_csv("data/holidays_events.csv")
    # transactions = pd.read_csv("data/transactions.csv")
    # items = pd.read_csv("data/items.csv")
    # print(oil.isnull().sum())
    # oil = oil.fillna(method="pad")
    # oil = oil.fillna(value=93.14, limit=1)
    # oil.to_pickle("data/temp/oil.pickle")
    # stores.to_pickle("data/temp/stores.pickle")
    # holidays_events.to_pickle("data/temp/holiday_events.pickle")
    # transactions.to_pickle("data/temp/transactions.pickle")
    # items.to_pickle("data/temp/items.pickle")
    mypath = "data/train/"
    trainFileList = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


    num = 1

    arrays = []
    for x in range(23,23+num):
        print(x)
        arrays.append(parallelize_dataframe(pd.read_csv("data/train/output_%s.csv" %str(x)),buildDataSet))
    trainArray = pd.concat(arrays)
    print(trainArray.head(100))

    trainArray.to_pickle("trainArray50MilOnlyStore39.pickle")
    testArray = parallelize_dataframe(pd.read_csv("data/train/output_%s.csv" %str(23+num)),buildDataSet)
    testArray.to_pickle("testArray50MilOnlyStore39.pickle")
