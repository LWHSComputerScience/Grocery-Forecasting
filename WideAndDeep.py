import tensorflow as tf
import pandas as pd
import numpy as np

#Note: subtracting one from store number to make it zero indexed
train_data = pd.read_pickle('trainArray5mil.pickle')
tf_dict = {True:1,False:0}
train_data["store_nbr"] = train_data["store_nbr"].apply(lambda x: x-1)


num_stores = 1
num_items=4100
num_states = 1
num_cities=1
num_clusters = 1
train_epochs = 1000000

#note: assuming that each store is done separately, so each piece of data is just sorted by item, so 2d
#feature columns, so each of these is for a given item
day = tf.feature_column.numeric_column('day')
month = tf.feature_column.categorical_column_with_identity('month',12)
store_city = tf.feature_column.categorical_column_with_identity('store_city',num_cities)
store_type = tf.feature_column.categorical_column_with_identity('store_type',5)
store_state = tf.feature_column.categorical_column_with_identity('store_state',num_states)
store_nbr = tf.feature_column.categorical_column_with_identity('store_nbr',num_stores)
item_nbr = tf.feature_column.categorical_column_with_identity('item_nbr',num_items)
onpromotion = tf.feature_column.categorical_column_with_vocabulary_list('onpromotion', vocabulary_list=[True,False])
cluster = tf.feature_column.categorical_column_with_identity('cluster', num_clusters)
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