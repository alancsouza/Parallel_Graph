#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data8 = German Credit.

"""

from functions_64 import *

data_name = "German Credit."
result_name = "Result_Data8_parallel_64.csv"
runtime_name = "Runtime_data8_64_parallel.csv"

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric'
data = pd.read_fwf(url, header = None)

# setting data precision  
float_type = 'float64'
int_type = 'int64'

X = data.iloc[:,:-1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))
X = X.astype(float_type)
X_type = X.dtypes

y = data.iloc[:,-1].copy()
y[y == 2] = -1
y = y.astype(int_type)
y_type = y.dtypes

print("The {} dataset has {} samples and its bit precision is {} for the X data and {} for y data".format(data_name, X.shape[0], X_type[1], y_type ))

# Filtering data:
X_new, y_new = remove_noise(X, y)

# Implementing kfold cross validation:
k = 4

kf = KFold(n_splits=k, shuffle = True, random_state = 1)
results = []
runtime = []

for train_index, test_index in kf.split(X_new):
    start = time.time()   

    X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
    y_train, y_test = y_new.iloc[train_index], y_new.iloc[test_index]

    # Splitting the data for concurrent computing
    data_split, split_size = split(X_train, y_train)
    
    # list for adding the support edges from each slot
    Support = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, S in zip(data_split, executor.map(support_edges, data_split)):
            Support.append(S)
        
    Support_arr = np.vstack(Support) # transform list to array

    data_train_new = pd.DataFrame(Support_arr)
        
    # Finding the new set of support edges
    arestas_suporte = support_edges(data_train_new)

    # Classification:
    y_hat = classify_data(X_test, y_test, arestas_suporte)

    AUC = compute_AUC(y_test, y_hat)
    print("The AUC for the {} bit precision is: {}".format(float_type, AUC))

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
