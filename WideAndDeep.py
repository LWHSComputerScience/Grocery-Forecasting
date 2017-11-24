import tensorflow as tf
import pandas as pd
import numpy as np

num_stores = 1
num_items=4100
num_states = 1
num_cities=1
train_epochs = 1000000

#note: assuming that each store is done separately, so each piece of data is just sorted by item, so 2d
#feature columns, so each of these is for a given item
day = tf.feature_column.numeric_column('day')
month = tf.feature_column.categorical_column_with_hash_bucket('month',hash_bucket_size=12)
store_city = tf.feature_column.categorical_column_with_hash_bucket('store_city',hash_bucket_size=num_cities)
store_type = tf.feature_column.categorical_column_with_hash_bucket('store_type',hash_bucket_size=5)
store_state = tf.feature_column.categorical_column_with_hash_bucket('store_state',hash_bucket_size=num_states)
store_nbr = tf.feature_column.categorical_column_with_hash_bucket('store_nbr',hash_bucket_size=num_stores)
item_nbr = tf.feature_column.categorical_column_with_hash_bucket('item_nbr',hash_bucket_size=num_items)
#more features!

base_features = [] #we should decide on these

crossed_columns = [
    tf.feature_column.crossed_column(['store_nbr', 'transactions'],2500)
    #more features like this
]

#should be dense data
deep_columns = [
    #indicator columns make sparse data denser
    tf.feature_column.indicator_column(month),
    #add dense features here

    #this is another way to densify stuff
    tf.feature_column.embedding_column(item_nbr, dimension=12)
]
model = tf.estimator.DNNLinearCombinedClassifier(
    model_dir='some dir',#directory to save model
    linear_feature_columns=base_features + crossed_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])

def input_fn():
    pass

for epoch in range(train_epochs):
    model.train(input_fn())
    if epoch%1000==0:
        print(model.evaluate(input_fn()))