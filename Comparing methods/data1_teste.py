from functions import *

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

data = pd.concat([X_new, y_new], axis=1)

data.to_csv('data1.csv')
