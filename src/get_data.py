from flares.ARDataSet import ARDataSet
from flares.data import get_dates

hnum = 401
root = "/srv/data/thli2739"

dates = get_dates(hnum, root, sort = True)
print(len(dates))

ds = ARDataSet(hnum, root, dates, verbose = True)

ds.segmented.to_csv("./segmented_401.csv")
ds.baseline.to_csv("./baseline_401.csv")
ds.sharps.to_csv("./sharps_401.csv")
