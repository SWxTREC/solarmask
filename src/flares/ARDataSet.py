from flares.active_region import ActiveRegion
from flares.data import get_dates
import numpy as np
import pandas as pd


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
        self.graphs = []

        self.times = []
        self.t = []

        prev_time = None
        time = None

        for date in dates:
            if verbose:
                print(f"Working on {hnum}, {date}")

            ar = ActiveRegion(hnum, date, root)
            if not ar.valid:
                if verbose:
                    print(f"skipping: {date}")
                continue

            self.segmented = self.segmented.append(ar.get_segmented(), ignore_index = True)
            self.baseline = self.segmented.append(ar.get_segmented(), ignore_index = True)
            self.sharps = self.segmented.append(ar.get_segmented(), ignore_index = True)

            data = ar.get_graph()
            if not hasattr(self, "graphs_labels"):
                self.graph_labels = data[1]