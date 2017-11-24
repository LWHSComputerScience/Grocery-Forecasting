import tensorflow as tf
import pandas as pd
import numpy as np

#Note: subtracting one from store number to make it zero indexed
train_data = pd.read_pickle('trainArray1Mil.pickle')
train_data["month"] = train_data["month"].apply(str)
print(train_data.head(10))
tf_dict = {True:1,False:0}
train_data["store_nbr"] = train_data["store_nbr"].apply(lambda x: x-1)


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
# store_city = tf.feature_column.categorical_column_with_hash_bucket('city',hash_bucket_size=100)
# store_type = tf.contrib.layers.embedding_column(tf.feature_column.categorical_column_with_hash_bucket('type', hash_bucket_size=40),1)
# store_state = tf.feature_column.categorical_column_with_identity('state',num_states)
# store_nbr = tf.feature_column.categorical_column_with_identity('store_nbr',num_stores)
# # item_nbr = tf.feature_column.categorical_column_with_identity('item_nbr',num_items)
# # onpromotion = tf.feature_column.categorical_column_with_vocabulary_list('onpromotion', vocabulary_list=["True","False"])
# cluster = tf.feature_column.categorical_column_with_identity('cluster', num_clusters)
#more features!
feature_cols = [day, month]
model = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[10, 10],
                                      model_dir="/tmp/boston_model")

CATEGORICAL_COLUMNS =["store_nbr", "type", "cluster", "month"]# ["month", "perishable", "onpromotion", "type", 'class']
CONTINUOUS_COLUMNS = ["transactions", "dcoilwtico", "day"]
LABEL_COLUMN = "unit_sales"

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in CATEGORICAL_COLUMNS+CONTINUOUS_COLUMNS}),
      y = pd.Series(data_set[LABEL_COLUMN].values),
      num_epochs=num_epochs,
      shuffle=shuffle)

train_input_fn = get_input_fn(train_data[:100])
# model.fit(input_fn=train_input_fn, steps=10000)

for epoch in range(train_epochs):
    model.train(get_input_fn(train_data[:100]), steps=10000)
    print("SUcces")
    if epoch%1==0:
        print(model.evaluate(get_input_fn(train_data[:100], num_epochs=1, shuffle=False)))