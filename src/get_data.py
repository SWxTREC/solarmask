from flares.ARDataSet import ARDataSet
from flares.data import get_dates

hnum = 1449
root = "/srv/data/thli2739"

dates = get_dates(hnum, root, sort = True)
print(len(dates))

ds = ARDataSet(hnum, root, dates, verbose = True)

ds.segmented.to_csv("./segmented_1449.csv")
ds.baseline.to_csv("./baseline_1449.csv")
ds.sharps.to_csv("./sharps_1449.csv")
