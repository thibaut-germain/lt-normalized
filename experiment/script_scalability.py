import sys
sys.path.append("./src/")
from sklearn.model_selection import ParameterGrid
from neighborhood import KNN
import pickle
import time
from random import shuffle
import pandas as pd

DATASET = "./dataset/computation-time/dataset.pkl"
SAVE = "./experiment/results/scalability.csv"

with open(DATASET, "rb") as f: 
    dataset = pickle.load(f)

configs = list(ParameterGrid(dict(n_neighbors = [1],wlen= [100], distance_name = ["LTNormalizedEuclidean", "NormalizedEuclidean"], n_jobs=[10]) ))


if __name__ == "__main__": 

    df = pd.DataFrame(columns=["distance", "wlen", "tslen", "duration"])
    shuffle(dataset)
    for ts in dataset: 
        shuffle(configs)
        for config in configs:
            knn = KNN(**config)
            start = time.time()
            knn.fit(ts)
            end = time.time()
            duration = end - start
            dist = config["distance_name"]
            wlen = config["wlen"]
            tslen = ts.shape[0]
            tdf = pd.DataFrame([[dist,wlen,tslen,duration]],columns=["distance", "wlen", "tslen", "duration"])
            df = pd.concat([df,tdf])
            df.to_csv(SAVE)
            print(f"tslen -- {dist} -- wlen: {wlen} -- tslen: {tslen} -- time: {duration}")