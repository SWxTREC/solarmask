from datetime import datetime
import os
import re
from astropy.io import fits
import numpy as np
import warnings
from astropy.io.fits.verify import VerifyWarning

# If set to true, includes the sharps features errors
# as well as the physical values
INCLUDE_ERRORS = True

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
    ret = []
    for i in os.listdir(base):
        if "Br" in i: # Testing for radial coords
            pattern =   re.compile(rf"""hmi\.sharp_cea_720s\.
                        (?P<region>[0-9]+)\.
                        (?P<date>[0-9]+\_[0-9]+)\_TAI\.Br\.fits""", re.VERBOSE)
            match = pattern.match(i)
            ret.append(datetime.strptime(match.group("date"), "%Y%m%d_%H%M%S"))
    return sorted(ret) if sort else ret

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
                    "SHRGT45"]

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

    for f1, f2 in files:
        filename = os.path.join(root, f1, f"sharp_{harpnum}", f"hmi.sharp_cea_720s.{harpnum}.{date_str}_TAI.{f2}.fits")
        assert os.path.isfile(filename)
        with fits.open(filename) as hdul:
            hdul.verify('fix')
            # Gather sharps info
            if f2 == "Br":
                for label in sharps_features:
                    sharps[label] = np.float(hdul[1].header[label])
            
            # Add to array
            data.append(np.array(hdul[1].data))
    
    ret = {"Bz" : data[0], "By" : data[1], "Bx" : data[2], "cont" : data[3], "sharps" : sharps}
    return ret
