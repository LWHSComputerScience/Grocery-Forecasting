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
store_city = tf.feature_column.categorical_column_with_identity('city',num_cities)
store_type = tf.feature_column.categorical_column_with_identity('type',5)
store_state = tf.feature_column.categorical_column_with_identity('state',num_states)
store_nbr = tf.feature_column.categorical_column_with_identity('store_nbr',num_stores)
item_nbr = tf.feature_column.categorical_column_with_identity('item_nbr',num_items)
onpromotion = tf.feature_column.categorical_column_with_vocabulary_list('onpromotion', vocabulary_list=[0,1],dtype=tf.int32)
cluster = tf.feature_column.categorical_column_with_identity('cluster', num_clusters)
oil = tf.feature_column.numeric_column('dcoilwtico')
transactions = tf.feature_column.numeric_column('transactions')
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

    #this is another way to densify stuff
    tf.feature_column.embedding_column(item_nbr, dimension=num_items)
]
model = tf.estimator.DNNLinearCombinedClassifier(
    model_dir='some dir',#directory to save model
    linear_feature_columns=base_features + crossed_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[256,128],
    dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.1))

input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_data,
    y=pd.Series(train_data["unit_sales"]),
    num_epochs=train_epochs,
    shuffle=False)

for epoch in range(train_epochs):
    model.train(input_fn())
    if epoch%1000==0:
        print(model.evaluate(input_fn()))