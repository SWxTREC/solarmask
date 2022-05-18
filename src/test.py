from solarmask.data import *
import warnings

warnings.filterwarnings('ignore', category=VerifyWarning, append=True)

hnums = get_hnums("/srv/data/thli2739")

keys = set()

for hnum in hnums:
    dates = get_dates(hnum, "/srv/data/thli2739")

    for i in os.listdir("/srv/data/thli2739/magnetogram/sharp_" + hnum):
        filename = os.path.join("/srv/data/thli2739/magnetogram/sharp_" + hnum, i)

        with fits.open(filename) as hdul:
            hdul.verify("fix")
            for i in hdul[1].header.keys():
                print(i)
    break
break
        

