#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data9 = Liver disorders

"""
from graph_parallel_functions import *

data_name = "Liver disorders"
result_name = "Result_Data9_parallel_64.csv"
runtime_name = "Runtime_data9_parallel_64.csv"

url = 'https://raw.githubusercontent.com/je-nunez/testing_WEKA_ml_on_BUPA/master/bupa_liver_disorders.csv'
data1 = pd.read_csv(url)
data = data1.copy()
data.info()

# setting data precision  
precisionX = 'float64'
precisionY = 'int64'

X = data.loc[:,'mcv' : 'gammagt']
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))
X = X.astype(precisionX)
X_type = X.dtypes

y = data.loc[:,'drinks'].copy()
y[y<3] = -1
y[y>=3] = 1
y = pd.DataFrame(y)
y = y.astype(precisionY)
y_type = y.dtypes

print("The {} dataset has {} samples and its bit precision is {} for the X data and {} for y data".format(data_name, X.shape[0], X_type[1], y_type ))

# Computing the Adjacency Matrix:
Adj_matrix = get_adjacency(X, int_type = precisionY, float_type = precisionX)

# Removing noise in the data
X_new, y_new = remove_noise(X, y, Adj_matrix, float_type = precisionX)

# Implementing kfold cross validation:
k = 4

kf = KFold(n_splits=k, shuffle = True, random_state = 1)
results = []
runtime = []

for train_index, test_index in kf.split(X_new):
    start = time.time()   

    X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
    y_train, y_test = y_new.iloc[train_index], y_new.iloc[test_index]

    # define the slot size
    if X_train.shape[0] > 400:
        split_size = round(X_train.shape[0]/100)
    else:
        split_size = round(X_train.shape[0]/50)
    
    
    if split_size > 1:    
        # Find the new data set from the support edges 
        X_train_new, y_train_new = parallel_graph(X_train, y_train, split_size, precisionX, precisionY)
    else:
        print("Data too small")
        X_train_new = X_train
        y_train_new = y_train
        
    # Finding the new set of support edges
    arestas_suporte = support_edges(X_train_new, y_train_new, precisionX, precisionY)

    # Classification:
    y_hat = classify_data(X_test, y_test, arestas_suporte, int_type = precisionY)

    AUC = compute_AUC(y_test, y_hat)
    print("The AUC for the {} bit precision is: {}".format(precisionX, AUC))

    end = time.time()
    final_time = end-start
    print("The overall model running time is {0:.2f} seconds \n".format(final_time))
    
    runtime.append(final_time)
    results.append(AUC)

print("The {} data was divided in {} slots \n".format(data_name, split_size))

results = pd.DataFrame(results)
results.to_csv(result_name, sep='\t', encoding='utf-8')

runtime = pd.DataFrame(runtime)
runtime.to_csv(runtime_name , sep='\t', encoding='utf-8')