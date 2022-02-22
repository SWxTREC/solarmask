from flares.data import get_dates
from flares.ARDataSet import ARDataSet
import numpy as np



root = "/srv/data/thli2739"
dates = get_dates(7115, root)

ar = ARDataSet(7115, root, dates[0:30], verbose = True)

ar.segmented.to_csv("./segmented.csv")
ar.baseline.to_csv("./baseline.csv")
ar.sharps.to_csv("./sharps.csv")

for graph in ar.graphs:
    print(graph)
