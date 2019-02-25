from functions import chip_clas, remove_noise, generate_data
import pandas as pd
import numpy as np

results = []
runtimes = []

for d in range(2,15,2):
  
    X, y = generate_data(d = d, nrow = 100, mean1 = 3, mean2 = 6, sd1 = 0.5, sd2 = 0.5)

    # Filtering data:
    X_new, y_new = remove_noise(X, y)

    # Comparing methods:
    method = ["nn_clas", "parallel", "extreme_search"]

    for model in method:
        y_hat, y_test, result, runtime = chip_clas(X_new, y_new, method = model, kfold = 5)
  
        print(" \n Dimension: {0}\n Method: {1} \n Avarege AUC: {2:.4f} \n Std. Deviation {3:.4f} \n Avarege Runtime: {4:.4f} \n".format(d, model, result.mean()[0], result.std()[0], runtime.mean()[0]))

        results.append(result.mean)
        runtimes.append(runtime)

results = pd.DataFrame(results)
results.to_csv("results.csv", sep='\t', encoding='utf-8')

runtimes = pd.DataFrame(runtimes)
runtimes.to_csv("runtimes.csv" , sep='\t', encoding='utf-8')