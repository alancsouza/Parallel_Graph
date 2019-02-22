from functions import chip_clas, remove_noise
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt  

def generate_data(d, nrow, mean1, mean2, sd1, sd2, plot=False):
  
  cov1 = np.diag(np.repeat(sd1,d))  # diagonal covariance
  cov2 = np.diag(np.repeat(sd2,d))
  x1, x2 = np.random.multivariate_normal(np.repeat(mean1, d), cov1, nrow).T
  x3, x4 = np.random.multivariate_normal(np.repeat(mean2, d), cov2, nrow).T
  
  X1 = np.c_[x1,x2]
  X2 = np.c_[x3,x4]
  
  
  y1 = np.repeat(1, nrow)
  y2 = np.repeat(-1, nrow)

  X = np.r_[X1,X2]
  y = np.r_[y1,y2]

  #data = pd.DataFrame(np.c_[X,y], columns = ['x1', 'x2', 'y'])
  
  if plot:
    plt.scatter(x1,x2, marker='v', color='r')
    plt.scatter(x3,x4)
    plt.title('Data distribuition')
    plt.axis('equal')
    plt.show()
  
  return pd.DataFrame(X), pd.DataFrame(y)


results = []
runtimes = []

for d in range(2,5,2):
    X, y = generate_data(d = d, nrow = 100, mean1 = 3, mean2 = 6, sd1 = 0.5, sd2 = 0.5)

    # Filtering data:
    X_new, y_new = remove_noise(X, y)

    print("The data has now {} samples".format(X_new.shape[0]))

    # Comparing methods:
    method = ["nn_clas", "parallel", "extreme_search"]

    for model in method:
        y_hat, y_test, result, runtime = chip_clas(X_new, y_new, method = model, kfold = 10)
  
        print(" \n Dimension: {0}\n Method: {1} \n Avarege AUC: {2:.4f} \n Std. Deviation {3:.4f} \n Avarege Runtime: {4:.4f} \n".format(d, model, result.mean()[0], result.std()[0], runtime.mean()[0]))

        results.append(result.mean)
        runtimes.append(runtime)

results = pd.DataFrame(results)
results.to_csv("results.csv", sep='\t', encoding='utf-8')

runtimes = pd.DataFrame(runtimes)
runtimes.to_csv("runtimes.csv" , sep='\t', encoding='utf-8')
