from datetime import datetime
import os
import re
from astropy.io import fits
import numpy as np
import warnings
from astropy.io.fits.verify import VerifyWarning
from bitstring import BitArray
import time

# If set to true, includes the sharps features errors
# as well as the physical values
INCLUDE_ERRORS = False

def get_hnums(root):
    """Returns a list of harpnumbers found in root/magnetogram

    Args:
        root (string): The root directory for magnetogram and continuum folders

    Returns:
        list: A list of harpnumbers as ints
    """
    folders = os.listdir(os.path.join(root, "magnetogram"))
    hnums = []
    for file in folders:
        hnums.append(int(file.split("_")[1]))
    return hnums

###### DATA
# Get a list of dates active for a specific harpnumber
def get_dates(harpnum, root, sort = False):
    """Extracts dates of a specified harpnumber
    
    Args:
        harpnum (int): The harpnumber
        root (string): The root file directory (see ActiveRegion for a description)
        sort (bool): If true, sorts by date, otherwise just grabs the way they are sorted in memory
    """
    base = os.path.join(root, "magnetogram", "sharp_" + str(harpnum))
    assert os.path.exists(base)
    ret_m = set()
    for i in os.listdir(base):
        if "Br" in i: # Testing for radial coords
            pattern =   re.compile(rf"""hmi\.sharp_cea_720s\.
                        (?P<region>[0-9]+)\.
                        (?P<date>[0-9]+\_[0-9]+)\_TAI\.Br\.fits""", re.VERBOSE)
            match = pattern.match(i)
            ret_m.add(datetime.strptime(match.group("date"), "%Y%m%d_%H%M%S"))
    ret_m = list(ret_m)
    return sorted(ret_m) if sort else ret_m

sharps_features = [ "USFLUX",\
                    "MEANGAM",\
                    "MEANGBT",\
                    "MEANGBZ",\
                    "MEANGBH",\
                    "MEANJZD",\
                    "TOTUSJZ",\
                    "MEANALP",\
                    "MEANJZH",\
                    "TOTUSJH",\
                    "ABSNJZH",\
                    "SAVNCPP",\
                    "MEANPOT",\
                    "TOTPOT",\
                    "MEANSHR",\
                    "SHRGT45",\
                    "R_VALUE"]

sharps_features_errors = ["ERRVF",\
                    "ERRGAM",\
                    "ERRBT",\
                    "ERRBZ",\
                    "ERRBH",\
                    "ERRJZ",\
                    "ERRUSI",\
                    "ERRALP",\
                    "ERRMIH",\
                    "ERRUSI",\
                    "ERRTAI",\
                    "ERRJHT",\
                    "ERRMPOT",\
                    "ERRTPOT",\
                    "ERRMSHA"]

if INCLUDE_ERRORS:
    sharps_features += sharps_features_errors

def get_data(harpnum, date, root):
    """Gets data from root for harpnumber date

    Args:
        harpnum (int): The harpnumber to extract
        date (datetime): The date
        root (str): The root of the folders (see ActiveRegion)

    Returns:
        dict: {"Bz" : , "By" : , "Bx" : , "cont" : , "sharp" : }
        "sharp" is the default header values (16 of them or 32 if including errors) - see Varad's paper or read
        JSOC documentation for a reference
    """

    # JSOC has an invalid format for the fits files. Later in this code I include the line
    # hdul.verify('fix') to fix this invalid formatting, but there is still a warning letting the 
    # user know that the image was invalidly formatted. This supresses that warning. If you would
    # like to read the warning, delete this line
    warnings.filterwarnings('ignore', category=VerifyWarning, append=True)

    date_str = date.strftime("%Y%m%d_%H%M%S")

    files = [("magnetogram", "Br"), ("magnetogram", "Bt"), ("magnetogram", "Bp"), ("continuum", "continuum")]
    data = []
    sharps = {}
    noaa_ars = None

    for f1, f2 in files:
        filename = os.path.join(root, f1, f"sharp_{harpnum}", f"hmi.sharp_cea_720s.{harpnum}.{date_str}_TAI.{f2}.fits")
        if not os.path.isfile(filename):
            return None
        with fits.open(filename) as hdul:
            hdul.verify('fix')
            # Gather sharps info
            if f2 == "Br":
                for label in sharps_features:
                    sharps[label] = np.float(hdul[1].header[label])
                noaa_ars = hdul[1].header["NOAA_ARS"]
            
            # Add to array
            data.append(np.array(hdul[1].data))
    
    ret = {"Bz" : data[0], "By" : data[1], "Bx" : data[2], "cont" : data[3], "sharps" : sharps, "NOAA_ARS" : noaa_ars}
    return ret


def write_mask(mask: np.array, hnum: int, date: str, root: str):
    date_str = date.strftime("%Y%m%d_%H%M%S")
    shape = mask.shape

    mask = mask.flatten()


    sharp_folder = os.path.join(root, f"sharp_{hnum}")

    if not os.path.exists(sharp_folder):
        os.makedirs(sharp_folder)

    outputfile = os.path.join(sharp_folder, f"mask_{hnum}_{date_str}.bin")

    if np.count_nonzero(mask) == 0:
        mask = b""
    else:
        
        mask = np.pad(mask, (8 - len(mask) % 8, 0), "constant", constant_values = (0,0))
        mask = mask.reshape(-1, 8)
        mask = np.packbits(mask).tobytes()


    # First 8 bytes specifies the shape of the mask
    prefix = shape[0].to_bytes(4, "big") + shape[1].to_bytes(4, "big")

    with open(outputfile, "wb") as fp:
        fp.write(prefix + mask)


def read_mask(hnum: int, date: datetime, root: str):
    date_str = date.strftime("%Y%m%d_%H%M%S")
    sharp_folder = os.path.join(root, f"sharp_{hnum}")
    if not os.path.exists(sharp_folder):
        raise Exception(f"Invalid harpnumber {hnum}")

    readfile = os.path.join(sharp_folder, f"mask_{hnum}_{date_str}.bin")

    if not os.path.exists(readfile):
        raise Exception(f"Invalid harpnumber {hnum}")

    with open(readfile, "rb") as fp:
        data = fp.read()

    prefix = data[0:8]
    shape = (int.from_bytes(prefix[0:4], "big"), int.from_bytes(prefix[4:], "big"))

    if len(data) <= 8:
        return np.zeros(shape, dtype = bool)

    mask = data[8:]

    return np.array(BitArray(mask))[-shape[0]*shape[1]:].reshape(shape)




















