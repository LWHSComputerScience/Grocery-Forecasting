import tensorflow as tf
import pandas as pd
import numpy as np

#Note: subtracting one from store number to make it zero indexed
train_data = pd.read_pickle('trainArray5mil.pickle')[:25000]
tf_dict = {True:1,False:0}
train_data["store_nbr"] = train_data["store_nbr"].apply(lambda x: x-1)
train_data["cluster"] = train_data["cluster"].apply(lambda x:x-1)
train_data.apply(pd.to_numeric,errors='ignore')
del train_data['date']
del train_data['id']
del train_data['onpromotion']
train_data["day"] = train_data['day'].apply(lambda x: (x-train_data['day'].mean())/train_data['day'].std())
train_data["transactions"] = train_data['transactions'].apply(lambda x: (x-train_data['transactions'].mean())/train_data['transactions'].std())
train_data["dcoilwtico"] = train_data['dcoilwtico'].apply(lambda x: (x-train_data['dcoilwtico'].mean())/train_data['dcoilwtico'].std())
y = pd.Series(train_data['unit_sales'])
print(y)
del train_data['unit_sales']
print(train_data.dtypes)
print(train_data.columns.tolist())


num_stores = 54
num_items=4100
num_states = 16
num_cities=22
num_clusters = 17
train_epochs = 1000000

#note: assuming that each store is done separately, so each piece of data is just sorted by item, so 2d
#feature columns, so each of these is for a given item
day = tf.feature_column.numeric_column('day')
month = tf.feature_column.categorical_column_with_hash_bucket('month',12, dtype=tf.int64)
store_city = tf.feature_column.categorical_column_with_hash_bucket('city',num_cities, dtype=tf.int64)
store_type = tf.feature_column.categorical_column_with_hash_bucket('type',5, dtype=tf.int64)
store_state = tf.feature_column.categorical_column_with_hash_bucket('state',num_states, dtype=tf.int64)
store_nbr = tf.feature_column.categorical_column_with_hash_bucket('store_nbr',num_stores, dtype=tf.int64)
item_nbr = tf.feature_column.categorical_column_with_hash_bucket('item_nbr',num_items, dtype=tf.int64)
#onpromotion = tf.feature_column.categorical_column_with_vocabulary_list('onpromotion', vocabulary_list=[0,1],dtype=tf.int32)
cluster = tf.feature_column.categorical_column_with_hash_bucket('cluster', num_clusters, dtype=tf.int64)
oil = tf.feature_column.numeric_column('dcoilwtico')
transactions = tf.feature_column.numeric_column('transactions')
# classes = tf.feature_column.categorical_column_with_hash_bucket('class', num_classes)
#more features!

base_features = [oil, cluster, store_type] #we should decide on these

crossed_columns = [
    tf.feature_column.crossed_column(['store_nbr', 'transactions'],2500)
    #more features like this
]

#should be dense data
deep_columns = [
    #indicator columns make sparse data denser
    tf.feature_column.embedding_column(month, dimension=12),
    #add dense features here

    tf.feature_column.embedding_column(item_nbr, dimension=num_items)
]
model = tf.estimator.DNNLinearCombinedRegressor(
    model_dir='some dir',#directory to save model
    linear_feature_columns=base_features + crossed_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[5,5])

input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_data,
    y=y,
    num_epochs=1,
    shuffle=True,batch_size=1)
print('hi')
for epoch in range(5):
    model.train(input_fn=lambda: input_fn(),steps=1)
    print('yes!')
    if epoch%100==0:
        print(model.evaluate(input_fn=lambda:input_fn(),steps=1))