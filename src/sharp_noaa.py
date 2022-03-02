import os
from astropy.io import fits
import warnings
from astropy.io.fits.verify import VerifyWarning

warnings.filterwarnings('ignore', category=VerifyWarning, append = True)

root = "/srv/data/thli2739"


sharps_m = set()
for f in os.listdir(os.path.join(root, "magnetogram")):
    if "sharp" in f:
        sharps_m.add(f.split("_")[1])

sharps_c = set()
for f in os.listdir(os.path.join(root, "continuum")):
    if "sharp" in f:
        sharps_c.add(f.split("_")[1])

sharps = sharps_c & sharps_m

i = 0
for sharp in sharps:
    r = os.path.join(root, "magnetogram", "sharp_" + str(sharp))
    
    noaa_ar = []
    
    j = 0
    for ff in os.listdir(r):
        if "Br" in ff:
            j += 1
            with fits.open(os.path.join(root, "magnetogram", "sharp_" + str(sharp), ff)) as hdul:
                hdul.verify('fix')
                noaa_ar.append(hdul[1].header["NOAA_ARS"])
        if j >= 3:
            break

    print(i, r, noaa_ar)
    i += 1

print(j)
