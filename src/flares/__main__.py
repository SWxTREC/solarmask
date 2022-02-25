from flares.data import get_dates, get_hnums
from flares.ARDataSet import ARDataSet
import numpy as np
import pickle
import os
import pandas as pd

root = "/srv/data/thli2739"
df = pd.read_csv("./labels.csv")
hnums = df["hnum"].unique()

out_dir = os.path.join(root, "data")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

print(f"Total number of hnums = {len(hnums)}")

hnums = hnums[0:5]
print(hnums)

for hnum in hnums:
    hnum_dir = os.path.join(out_dir, "sharp_" + str(hnum))
    if not os.path.isdir(hnum_dir):
        os.mkdir(hnum_dir)

    dates = get_dates(hnum, root, sort = False)
    print(f"Starting {hnum}")
    print(f"{len(dates)} total dates")

    if len(dates) == 0:
        continue

    
    ar = ARDataSet(hnum, root, dates[0:3], verbose = True, frame_dir = "./frames/sharp_" + str(hnum))

    if ar.len == 0:
        continue

    if not os.path.isdir("./data/sharp_7115"):
        os.mkdir("./data/sharp_7115")

    ar.segmented.to_csv("./data/segmented.csv")
    ar.baseline.to_csv("./data/baseline.csv")
    ar.sharps.to_csv("./data/sharps.csv")

    graph = ar.graphs
    hnum, labels = ar.graph_labels
    graph_data = (hnum, labels, graph)

    pickle.dump(graph_data, open("./graph.txt", "wb"))
