#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data1 = Banknote Auth.

"""

from graph_functions import *

data_name = "Banknote Auth."
result_name = "Result_Data1_16.csv"
runtime_name = "Runtime_data1_16.csv"

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
data = pd.read_csv(url, header = None)

precisionX = 'float16'
precisionY = 'int8'

X = data.iloc[:,:-1]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))
X = X.astype(precisionX)
X_type = X.dtypes

y = data.iloc[:,-1].copy()

y[y == 0] = -1
y = y.astype(precisionY)
y_type = y.dtypes

print("This dataset has {} samples and its bit precision is {} for the X data and {} for y data".format(X.shape[0], X_type[1], y_type ))


# Computing the Adjacency Matrix:
print("Computing the Adjacency Matrix: ")
start_train = time.time()
start_Adj = time.time()
Adj_matrix = get_adjacency(X, int_type = precisionY, float_type = precisionX)
end_Adj = time.time()


print("The execution code of the Adj Matrix took {0:.2f} seconds to run and its data type is {1:} \n".format((end_Adj-start_Adj), Adj_matrix.dtype))

# Removing noise in the data
print("Removing noise in the data...")
X_new, y_new = remove_noise(X, y, Adj_matrix, float_type = precisionX)

# Implementing kfold cross validation:
k = 10 

kf = KFold(n_splits=k, shuffle = True, random_state = 1)
results = []
runtime = []

for train_index, test_index in kf.split(X_new):
    start = time.time()
    start_train = time.time()

    X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
    y_train, y_test = y_new.iloc[train_index], y_new.iloc[test_index]
    
    X_new_type = X_train.dtypes[0]
    y_new_type = y_train.dtypes

    print("The new training data has {} samples and its bit precision is {} for the X data and {} for y data \n".format(X_train.shape[0], X_new_type, y_new_type))

    # Computing the new Adjacency matrix for the removed noise data:
    print("Computing the new Adjacency matrix for the removed noise data: ")
    start_Adj = time.time()
    Adj_matrix = get_adjacency(X_train, int_type = precisionY, float_type = precisionX)
    end_Adj = time.time()

    print("The new Adj Matrix took {0:.2f} seconds to run and its data type is {1:} \n".format((end_Adj-start_Adj), Adj_matrix.dtype))

    # Finding the separation border:
    print("Finding the separation border..... \n")
    borda = get_borda(y_train, Adj_matrix, int_type = precisionY)

    # Finding the support edges:
    print("Finding the support edges: \n")
    X_suporte, y_suporte = get_arestas_suporte(X_train, y_train, borda, Adj_matrix, int_type = precisionY)
                                                                
    end_train = time.time()  

    print("The overall training process took {0:.2f} seconds to run \n".format((end_train-start_train)))

    ################################# Classification #########################
    print("Start classification")

    start_class = time.time()
    y_hat = classify_data(X_test, y_test, X_suporte, y_suporte, int_type = precisionY)
    end_class = time.time()

    print("The overall classification process took {0:.2f} seconds to run \n".format((end_class-start_class)))

    ############################ Results ####################

    # getting the false and true positive rate

    fpr, tpr, _ = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)
    results.append(roc_auc)

    print("The AUC for the {} bit precision is: {}".format(precisionX, roc_auc))

    end = time.time()
    print("The overall model running time is {0:.2f} seconds \n".format((end-start)))
    final_time = end-start
    runtime.append(final_time)

results = pd.DataFrame(results)
results.to_csv(result_name, sep='\t', encoding='utf-8')

runtime = pd.DataFrame(runtime)
runtime.to_csv(runtime_name , sep='\t', encoding='utf-8')