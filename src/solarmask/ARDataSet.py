from .active_region import ActiveRegion
from .data import get_dates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ARDataSet:
    def __init__(self, hnum, root, dates = None, verbose = False):
        """Active region range

        Args:
            hnum ([type]): [description]
            date_range ([type]): [description]
            root ([type]): [description]
        """
        if dates is None:
            dates = get_dates(hnum, root, sort = True)

        self.segmented = pd.DataFrame()
        self.sharps = pd.DataFrame()
        self.baseline = pd.DataFrame()

        for date in dates:
            if verbose:
                print(f"Working on {hnum}, {date}")

            ar = ActiveRegion(hnum, date, root)
            if not ar.valid:
                if verbose:
                    print(f"skipping: {date}")
                continue

    
            self.segmented = pd.concat([self.segmented, pd.DataFrame(ar.segmented_dataset, index=[date])])
            self.baseline = pd.concat([self.baseline, pd.DataFrame(ar.baseline_dataset, index=[date])])
            self.sharps = pd.concat([self.sharps, pd.DataFrame(ar.sharps_dataset, index=[date])])
