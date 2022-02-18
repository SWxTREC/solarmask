from flares.active_region import ActiveRegion
from flares.data import get_dates
import numpy as np


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

        self.segmented, self.segmented_labels = [], []
        self.sharps, self.sharps_labels = [], []
        self.baseline, self.baseline_labels = [], []
        self.graphs, self.graphs_labels = [], []

        for date in dates:
            if verbose:
                print(f"Working on {hnum}, {date}")

            ar = ActiveRegion(hnum, date, root)
            if not ar.valid:
                if verbose:
                    print(f"skipping: {date}")
                continue

            data = ar.get_segmented()
            self.segmented.append(list(data.values()))
            if len(self.segmented_labels) == 0:
                self.segmented_labels = list(data.keys())

            data = ar.get_baseline()
            self.baseline.append(list(data.values()))
            if len(self.baseline_labels) == 0:
                self.baseline_labels = list(data.keys())

            data = ar.get_sharps()
            self.sharps.append(list(data.values()))
            if len(self.sharps_labels) == 0:
                self.sharps_labels = list(data.keys())

            data = ar.get_graph()
            self.graphs.append(data[0])
            if len(self.graphs_labels) == 0:
                self.graphs_labels = list(data[1])


        self.segmented = np.array(self.segmented)
        self.baseline = np.array(self.segmented)
        self.sharps = np.array(self.segmented)
