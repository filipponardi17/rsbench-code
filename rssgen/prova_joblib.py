"/home/filippo.nardi/rsbench-code/rssgen/MNIST_LOGIC_OUT_FOLDER/train/0.joblib"

import joblib

data = joblib.load("/home/filippo.nardi/rsbench-code/rssgen/MNIST_LOGIC_OUT_FOLDER/train/0.joblib")
print(type(data))
# Se è un dizionario, fai:
print(data.keys())
print(data)