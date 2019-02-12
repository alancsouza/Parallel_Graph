from functions import *

data = pd.read_csv("./data1.csv")

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

y_hat, y_test, result, runtime = chip_clas(X, y, method = "pseudo_support_edges", kfold = 10)
  
print(" \n Method: {0} \n Avarege AUC: {1:.4f} \n Std. Deviation {2:.4f} \n Avarege Runtime: {3:.4f} \n".format("pseudo_support_edges", result.mean()[0], result.std()[0], runtime.mean()[0]))
