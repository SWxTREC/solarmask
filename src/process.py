import os
from astropy.io import fits
import warnings
from astropy.io.fits.verify import VerifyWarning
from flares.active_region import ActiveRegion
from flares.data import get_dates
import pandas as pd
import time
import networkx as nx
import cProfile

warnings.filterwarnings('ignore')

root = "/srv/data/thli2739"

# Get the list of sharps 
sharps_m = set()
for f in os.listdir(os.path.join(root, "magnetogram")):
    if "sharp" in f:
        sharps_m.add(int(f.split("_")[1]))

sharps_c = set()
for f in os.listdir(os.path.join(root, "continuum")):
    if "sharp" in f:
        sharps_c.add(int(f.split("_")[1]))

sharps = list(sharps_c & sharps_m)



sharps = pd.DataFrame()
segmented = pd.DataFrame()
baseline = pd.DataFrame()
noaa_ars = pd.DataFrame()

i = 0
total = 289000
for sharp in [7115]:
    if not os.path.isdir(os.path.join(root, "graph", "sharp_" + str(sharp))):
        os.mkdir(os.path.join(root, "graph", "sharp_" + str(sharp)))
    
    r = os.path.join(root, "magnetogram", "sharp_" + str(sharp))
    dates = get_dates(sharp, root, sort = True) 
    total = len(dates)
    
    i = 0
    for date in dates[200:]:
        start_ = time.time()
        start = time.time()
        ar = ActiveRegion(sharp, date, root)
        
        read_data = time.time() - start

        if ar.valid:
            # Update NOAA Dict
            for noaa_ar in ar.meta["NOAA_ARS"].split(","):
                noaa_ars = pd.concat([noaa_ars, pd.DataFrame({"hnum" : sharp, "noaa_ar" : int(noaa_ar)}, index = [0])])


            start = time.time()
            # Update the three vector dataframes
            sharps = pd.concat([sharps, pd.DataFrame(ar.sharps_dataset, index=[date])])

            sharps_time = time.time() - start
            start = time.time()

            segmented = pd.concat([sharps, pd.DataFrame(ar.segmented_dataset, index=[date])])
            segmented_time = time.time() - start
            start = time.time()
            
            baseline = pd.concat([sharps, pd.DataFrame(ar.baseline_dataset, index=[date])])
            baseline_time = time.time() - start
            start = time.time()

            # Save the graph
            flname = date.strftime("%m%d%Y_%H%M%S")
            nx.write_gpickle(ar.graph_dataset["graph"], os.path.join(root, "graph", "sharp_" + str(sharp), flname + ".graph"))
            graph_time = time.time() - start

            i += 1
        if i > 3:
            break

        total = time.time() - start_
        print("Total, sharps, segmented, baseline, graph: ", total, sharps_time, segmented_time, baseline_time, graph_time)

print(noaa_ars)

