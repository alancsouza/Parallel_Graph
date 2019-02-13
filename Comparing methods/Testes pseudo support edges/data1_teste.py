from functions import chip_clas, remove_noise
import pandas as pd
from sklearn import preprocessing

'''
data = pd.read_csv("./data1.csv")

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
'''
data_name = "Banknote Auth."

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
data = pd.read_csv(url, header = None)

X = data.iloc[:,:-1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))

y = data.iloc[:,-1].copy()
y[y == 0] = -1

# Filtering data:
X_new, y_new = remove_noise(X, y)

y_hat, y_test, result, runtime = chip_clas(X, y, method = "pseudo_support_edges", kfold = 10)
  
print(" \n Method: {0} \n Avarege AUC: {1:.4f} \n Std. Deviation {2:.4f} \n Avarege Runtime: {3:.4f} \n".format("pseudo_support_edges", result.mean()[0], result.std()[0], runtime.mean()[0]))
