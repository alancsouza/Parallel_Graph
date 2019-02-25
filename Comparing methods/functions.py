import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.matlib # use repmat function
from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import random
import time
import copy
import concurrent.futures
random.seed(1)

# Compute Adjacency matrix for the Grabriel Graph
def get_adjacency(X):
  dist_matrix = distance_matrix(X,X)
  Adj_matrix = np.zeros(shape = dist_matrix.shape)
  nrow = dist_matrix.shape[0]
  for i in range(nrow):
    for j in range(nrow):
      if (i != j):          
        d1 = (dist_matrix[i,j])/2
        dist = pd.DataFrame((X.iloc[i,:]+X.iloc[j,:])/2).T 
        d = distance_matrix(dist, X)      
        d[0,i] = float("inf")
        d[0,j] = float("inf")      
        compara = (d<d1)
        
        if not compara.any():
          Adj_matrix[i,j] = 1
          Adj_matrix[j,i] = 1
          
  return Adj_matrix

def separate_labels(data):
  c1 = data[data.iloc[:,-1] ==  1]
  c2 = data[data.iloc[:,-1] == -1]

  return c1, c2


# Removing overlapping samples:
def remove_noise(X, y):
  
  data = pd.concat([X, y], axis = 1)
  Adj_matrix = get_adjacency(X)
  
  # getting the indices from each class
  c1_idx = np.asarray(data.index[data.iloc[:,-1] == 1])
  c2_idx = np.asarray(data.index[data.iloc[:,-1] == -1])
  
  # Computing the vertex degree (A) for each class
  A1 = Adj_matrix[:,c1_idx].sum(axis = 0) 
  A2 = Adj_matrix[:,c2_idx].sum(axis = 0)
  
  # Computing A_hat which is the number of edges connected to the vertex that are from the same class
  Adj_matrix = pd.DataFrame(Adj_matrix)
  Adj_1 = Adj_matrix.iloc[c1_idx,c1_idx]
  Adj_2 = Adj_matrix.iloc[c2_idx,c2_idx]

  A1_hat = Adj_1.sum(axis = 0)
  A2_hat = Adj_2.sum(axis = 0)
  
  #Computing the quality coefficient Q for each class
  Q1 = A1_hat / A1
  Q2 = A2_hat / A2

  # Computing the threshold value t for each class
  t1 = sum(Q1) / Q1.shape[0]
  t2 = sum(Q2) / Q2.shape[0]

  # getting the indices the are below the threshold
  noise_c1 = np.asarray(np.where(Q1 < t1)).ravel()
  noise_c2 = np.asarray(np.where(Q2 < t2)).ravel()

  noise_c1_idx = c1_idx[noise_c1]
  noise_c2_idx = c2_idx[noise_c2]

  # removing the samples that can be considered a noise
  noise_idx = np.r_[noise_c1_idx, noise_c2_idx]
  new_data = data.drop(noise_idx)
  
  new_X = new_data.iloc[:,:-1]
  new_y = new_data.iloc[:,-1]
  
  print("{} samples where removed from the data. \n".format(data.shape[0]-new_data.shape[0]))
  print("The data set now has {} samples ".format(new_data.shape[0]))
  
  return new_X, new_y

# Split the data for concurrent computing
def split(X_train, y_train):
    if X_train.shape[0] > 400: # define the slot size
        split_size = round(X_train.shape[0]/100)
    else:
        split_size = round(X_train.shape[0]/50)

    data_train = np.c_[X_train, y_train]
    np.random.shuffle(data_train)

    data_split = np.array_split(data_train, split_size)

    return data_split, split_size


# Finding the separation border:
def get_borda(y, Adj_matrix):
  y_t = pd.DataFrame(y).T
  
  ncol = y_t.shape[1]
  mask = np.matlib.repmat(y_t, ncol, 1)
  mask2 = pd.DataFrame(mask*Adj_matrix)  
  borda = pd.DataFrame(np.zeros(ncol)).T

  for idx in range(ncol):
    a1 =  sum(-y_t.iloc[0, idx] == mask2.iloc[idx,:]) # check if the labels are different
    if a1 > 0:
      borda[idx] = 1
    
  return borda

# Finding the support edges:
def get_arestas_suporte(X, y, borda, Adj_matrix):
  X = np.asarray(X)
  y_t = pd.DataFrame(y).T
  ncol = y_t.shape[1]
  mask = np.matlib.repmat(y_t, ncol, 1)
  nrow = Adj_matrix.shape[0]
  maskBorda = np.matlib.repmat(borda == 1, nrow, 1)
  maskBorda = np.asarray(maskBorda)

  # Removing the lines that not belong to the margin
  aux = maskBorda * np.transpose(maskBorda)

  # Removing edges that do not belong to the graph
  aux = Adj_matrix * aux

  # Removing edges from same labels vertices
  aux1 = aux + (mask * aux)
  aux2 = aux - (mask * aux)
  aux1 = np.asarray(aux1)
  aux2 = np.asarray(aux2)
  aux = aux1 * np.transpose(aux2)

  # converting matrix to binary
  aux  = (aux != 0)

  arestas = np.where(aux == 1)

  arestas = np.transpose(np.asarray(arestas))
  nrow_arestas = arestas.shape[0]
  ncol_arestas = arestas.shape[1]

  arestas_suporte = []
  y_suporte = []

  y_arr = np.asarray(y)

  for i in range(nrow_arestas):
    for j in range(ncol_arestas):
    
      idx = arestas[i,j]
      arestas_suporte.append(X[idx,:])
      y_suporte.append(y_arr[idx])

  
  
  X_suporte = np.asarray(arestas_suporte)
  y_suporte = np.asarray(y_suporte)
  
  return X_suporte, y_suporte

# Another support edges function that contains the other functions
def support_edges(data):  

  if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)   
  X_train = data.iloc[:,:-1]
  y_train = data.iloc[:, -1]
  Adj_matrix = get_adjacency(X_train)
  
  borda = get_borda(y_train, Adj_matrix)
  X_suporte, y_suporte = get_arestas_suporte(X_train, y_train, borda, Adj_matrix)
  
  arestas_suporte = np.c_[X_suporte, y_suporte]
  if arestas_suporte.shape[0] > 0:
    arestas_suporte = np.unique(arestas_suporte, axis = 0)
  
  return arestas_suporte

# Classification
def classify_data(X_test, y_test, arestas_suporte):
  
  X_suporte = arestas_suporte[:,:-1]
  #y_suporte = arestas_suporte[:,-1]
  nrow = X_test.shape[0]
  dist_test = distance_matrix(X_test, X_suporte) # compute the distance from the sample to the support egdes

  y_hat = np.zeros(nrow)

  for idx in range(nrow):
    dist = dist_test[idx,:]
    min_idx = np.argmin(dist)
    y_hat[idx] = arestas_suporte[min_idx, -1] 
  
  return y_hat

# Performance measure using AUC
def compute_AUC(y_test, y_hat):
  fpr, tpr, _ = roc_curve(y_test, y_hat)
  if fpr.shape[0] < 2 or tpr.shape[0] < 2:
      roc_auc = float('nan')
  else:
      roc_auc = auc(fpr, tpr)
  
  return roc_auc

# Parallel graph method:
def parallel_graph(X_train, y_train, split_size):
  data_train = np.c_[X_train, y_train]
  np.random.shuffle(data_train)

  data_split = np.array_split(data_train, split_size)
  arestas_suporte_final = []
  
  for i in range(split_size):
    data = pd.DataFrame(data_split[i])
    X_train = data.iloc[:,:-1]
    y_train = data.iloc[:, -1]

    # Finding the support edges from this slot of data:
    arestas_suporte = support_edges(data)

    arestas_suporte_final.append(arestas_suporte)
    
  arr = arestas_suporte_final[0]

  for i in range(split_size-1):
    i = i+1
    arr = np.concatenate((arr, arestas_suporte_final[i]), axis = 0)  
    
  data_train_new = pd.DataFrame(arr)
  X_train_new = data_train_new.iloc[:,:-1]
  y_train_new = data_train_new.iloc[:,-1]
  
  return X_train_new, y_train_new

def compute_extreme_search(data, scale_factor = 10, k = 5):
  
  '''
  parameters:
  scale_factor: integer that will divide the data set and choose the amount of samples from each class
  k: how many samples will be saved from the data (Nearest and distant samples)
  '''
  # separating the lables
  c1 = data[data.iloc[:,-1] ==  1]
  c2 = data[data.iloc[:,-1] == -1]

  c1_x = c1.iloc[:,:-1]
  c2_x = c2.iloc[:,:-1]

  # Choosing one random reference sample from each class
  c1_reference = c1_x.sample(n = int(c1.shape[0]/scale_factor))
  c2_reference = c2_x.sample(n = int(c2.shape[0]/scale_factor))

  # Compute the distance matrix between each sample and the opposite class
  dist_c1 = distance_matrix(c2_reference, c1_x)
  dist_c2 = distance_matrix(c1_reference, c2_x)

  dist_c1_idx = np.argsort(dist_c1, axis = 1)
  dist_c2_idx = np.argsort(dist_c2, axis = 1)

  idx1_min = dist_c1_idx[:, :k].ravel() # ravel transforms a 2D column into 1D
  idx1_max = dist_c1_idx[:, -k:].ravel() 

  idx2_min = dist_c2_idx[:, :k].ravel() # ravel transforms a 2D column into 1D
  idx2_max = dist_c2_idx[:, -k:].ravel() 

  idx1 = np.r_[idx1_min, idx1_max]
  idx1 = np.unique(idx1)

  idx2 = np.r_[idx2_min, idx2_max]
  idx2 = np.unique(idx2)

  new_c1 = c1.iloc[idx1,:]
  new_c2 = c2.iloc[idx2,:]

  new_data = pd.concat([new_c1, new_c2], axis = 0)
  
  return new_data

# Gabriel Graph classifier using nn_clas method
def nn_clas(X_train, y_train, X_test, y_test):

  data_train = np.c_[X_train, y_train]
  arestas_suporte = support_edges(data_train)
  y_hat = classify_data(X_test, y_test, arestas_suporte)

  return y_hat


def parallel_concurrent(X_train, y_train, X_test, y_test):
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

  return y_hat

def extreme_search(X_train, y_train, X_test, y_test):
  
  data_train = pd.concat([X_train, y_train], axis = 1)
  new_data = compute_extreme_search(data_train, scale_factor=10, k = 5)
  support = support_edges(new_data)
  y_hat = classify_data(X_test, y_test, support)

  return y_hat


def chip_clas(X, y, method , kfold = 10, test_size = 0.2):

  """
    Available methods:
    "parallel": Implements concurrent futures and parallelization technique
    "nn_clas": Implements nn_clas classification
    "extreme_search" = Implements data reduction technique

  """

  runtime = []

  if kfold > 0 : 

    kf = KFold(n_splits = kfold, shuffle = True, random_state = 1)

    results = []

    for train_index, test_index in kf.split(X):

      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      if method == "parallel" :
        start = time.time() 
        y_hat = parallel_concurrent(X_train, y_train, X_test, y_test)      
        end = time.time()

      elif method == "nn_clas":
        start = time.time()
        y_hat  = nn_clas(X_train, y_train, X_test, y_test)
        end = time.time()

      elif method == "extreme_search" :
        start = time.time()
        y_hat = extreme_search(X_train, y_train, X_test, y_test)
        end = time.time()

      else :
        print("Method not available")
        return None

      AUC = compute_AUC(y_test, y_hat)
      results.append(AUC)
      runtime.append(end-start)

    results = pd.DataFrame(results)
    runtime = pd.DataFrame(runtime)
      


  elif kfold == 0:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
    start = time.time()
    if method == "parallel" :
        y_hat = parallel_concurrent(X_train, y_train, X_test, y_test)      

    elif method == "nn_clas":
      y_hat  = nn_clas(X_train, y_train, X_test, y_test)

    elif method == "extreme_search":
      y_hat = extreme_search(X_train, y_train, X_test, y_test)

    else :
      print("Method not available")
      return None
    end =  time.time()
    runtime = end - start
    results = compute_AUC(y_test, y_hat)
  else :
    print("Error: kfold number invalid")


  return y_hat, y_test, results, runtime


def generate_data(d, nrow, mean1, mean2, sd1, sd2, plot=False):
  
  cov1 = np.diag(np.repeat(sd1,d))  # diagonal covariance
  cov2 = np.diag(np.repeat(sd2,d))
  X1 = np.random.multivariate_normal(np.repeat(mean1, d), cov1, nrow)
  X2 = np.random.multivariate_normal(np.repeat(mean2, d), cov2, nrow)
  
  y1 = np.repeat(1, nrow)
  y2 = np.repeat(-1, nrow)

  X = pd.DataFrame(np.r_[X1,X2])
  y = pd.DataFrame(np.r_[y1,y2])

  if plot:
    plot_2d(X, y)
  
  return X, y

def plot_2d(X, y):
  data = pd.concat([X, y], axis = 1)  
  c1, c2 = separate_labels(data)
  plt.scatter(c1.iloc[:,0],c1.iloc[:,1], marker='v', color='r')
  plt.scatter(c2.iloc[:,0],c2.iloc[:,1])
  plt.title('Data distribuition')
  plt.axis('equal')
  plt.show()
  
  return None  
