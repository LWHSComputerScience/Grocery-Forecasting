import tensorflow as tf
import pandas as pd
import numpy as np
print(tf.__version__)
#Note: subtracting one from store number to make it zero indexed
train_data = pd.read_pickle('trainArray40Mil39.pickle')
for x in ["month", "city", "type", "state", "cluster", "store_nbr", "item_nbr", "onpromotion", "perishable"]:
    train_data[x] = train_data[x].apply(str)
train_data["day"] = train_data["day"].apply(int)
print(train_data.head(10))
print(train_data.dtypes)
tf_dict = {True:1,False:0}
# train_data["store_nbr"] = train_data["store_nbr"].apply(lambda x: x-1)


num_stores = 1
num_items=4100
num_states = 1
num_cities=100
num_clusters = 1
train_epochs = 1000000

#note: assuming that each store is done separately, so each piece of data is just sorted by item, so 2d
#feature columns, so each of these is for a given item
day = tf.feature_column.numeric_column('day')
month = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('month',hash_bucket_size=15), 1)
store_city = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('city',hash_bucket_size=100),1)
store_type = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('type', hash_bucket_size=40),1)
store_state = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('state', hash_bucket_size=40),1)
store_nbr = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('store_nbr', hash_bucket_size=40),1)
item_nbr = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('item_nbr', hash_bucket_size=40),1)
onpromotion = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('onpromotion', hash_bucket_size=10), 1)
cluster = tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('cluster', hash_bucket_size=40),1)
#more features!
feature_cols = [day, month, store_city, store_type, store_state, store_nbr, item_nbr, cluster, onpromotion]
model = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[64, 32, 16, 8,4,2],
                                      model_dir="tmp/model7", optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=.1,
#       l1_regularization_strength=0.01,
#         l2_regularization_strength=0.01

    ))

CATEGORICAL_COLUMNS =["store_nbr", "type", "cluster", "month", "state", "item_nbr", "city", "onpromotion"]# ["month", "perishable", "onpromotion", "type", 'class']
CONTINUOUS_COLUMNS = ["transactions", "dcoilwtico", "day"]
LABEL_COLUMN = "unit_sales"

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in CATEGORICAL_COLUMNS+CONTINUOUS_COLUMNS}),
      y = pd.Series(data_set[LABEL_COLUMN].values),
      num_epochs=num_epochs,
      shuffle=shuffle)

# train_input_fn = get_input_fn(train_data[:-10000])
# model.fit(input_fn=train_input_fn, steps=10000)
print(len(train_data))
for epoch in range(train_epochs):
    model.train(get_input_fn(train_data[:-100000]), steps=10000)
    print("Succes")
    if epoch%1==0:
        print(model.evaluate(get_input_fn(train_data[-100000:], num_epochs=1, shuffle=False)))