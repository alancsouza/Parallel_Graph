from functions import *

data_name = "Australian credit"

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'

data = pd.read_csv(url, sep='\s+', header=None, skiprows=1)

X = data.iloc[:,:-1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))

y = data.iloc[:,-1].copy()
y[y == 0] = -1

# Filtering data:
X_new, y_new = remove_noise(X, y)

# Comparing methods:
method = ["nn_clas", "parallel", "extreme_search"]

print("Dataset: {}".format(data_name))

f = open("results.txt", "a")
f.write("Dataset: %s\r\n" % data_name)

for model in method:
  y_hat, y_test, result, runtime = chip_clas(X_new, y_new, method = model, kfold = 10)
  
  print(" \n Method: {0} \n Avarege AUC: {1:.4f} \n Std. Deviation {2:.4f} \n Avarege Runtime: {3:.4f} \n".format(model, result.mean()[0], result.std()[0], runtime.mean()[0]))

  results = {'Method': model,
          'Avarege AUC': result.mean()[0],
          'Std deviation': result.std()[0],
          'Avarege runtime': runtime.mean()[0]}
  
  json.dump(results, f)
  f.write('\n')
  
f.close()