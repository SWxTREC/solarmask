from solarmask.data import *
import warnings
import pandas as pd
from skimage.morphology import square, binary_dilation
import matplotlib.pyplot as plt
from bitstring import BitArray
import sys
from datetime import datetime
import matplotlib.pyplot as plt


# For opening fits files
warnings.filterwarnings('ignore', category=VerifyWarning, append=True)

# A list of all possible harpnumbers from magnetograms
hnums = get_hnums("/srv/data/thli2739")

# Sharps data (might as well)
sharps_df = pd.DataFrame()

# Kernel for dilation of neutral line
kernel = square(5)

# Generate sharps dataset
for n, hnum in enumerate(hnums[0:10]):

    dates = get_dates(hnum, "/srv/data/thli2739")
    
    for nn, date in enumerate(dates):
        date_str = date.strftime("%Y%m%d_%H%M%S")
        sys.stdout.write(f"\rHnum {hnum}: {(n*100)/len(hnums):.2f}% -- Date {date_str}: {(nn*100)/len(dates):.2f}%")
        filename = os.path.join("/srv/data/thli2739/magnetogram", f"sharp_{hnum}", f"hmi.sharp_cea_720s.{hnum}.{date_str}_TAI.Br.fits")

        with fits.open(filename) as hdul:
            hdul.verify("fix")

            # GET SHARPS KEYS
            new_dict = {"hnum" : [hnum], "date" : [date]}
            for key in hdul[1].header.keys():
                value = hdul[1].header[key]
                new_dict[key] = [hdul[1].header[key]]
            sharps_df = pd.concat([sharps_df, pd.DataFrame(new_dict)])        

            # GET NEUTRAL LINE
            Bz = np.array(hdul[1].data)

        

        nl_mask = binary_dilation(Bz < -150, kernel) & binary_dilation(Bz > 150, kernel)

        if hnum == 3344 and date == datetime(2013, 11, 10, 2, 0, 0):
            plt.imshow(nl_mask)
            plt.savefig("orig.png")

        write_mask(nl_mask, hnum, date, "/srv/data/thli2739")

        sys.stdout.flush()

    if n > 10:
        break

sys.stdout.write("Done\n")

sharps_df.to_csv("./sharps.csv", index = False)
nl_df.to_csv("./nl.csv", index = False)
