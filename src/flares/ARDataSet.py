from flares.active_region import ActiveRegion
from flares.data import get_dates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class ARDataSet:
    def __init__(self, hnum, root, dates = None, verbose = False, frame_dir = None):
        """Active region range

        Args:
            hnum ([type]): [description]
            date_range ([type]): [description]
            root ([type]): [description]
        """

        if frame_dir is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        if not os.path.isdir(frame_dir):
            os.mkdir(frame_dir)

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
        
        self.len = 0

        for date in dates:
            if verbose:
                print(f"Working on {hnum}, {date}")

            ar = ActiveRegion(hnum, date, root)
            if not ar.valid:
                if verbose:
                    print(f"skipping: {date}")
                continue

    
            data = ar.get_segmented()
            data.update({"date" : date, "hnum" : hnum})
            self.segmented = pd.concat([self.segmented, pd.DataFrame(data, index=[0])])

            data = ar.get_baseline()
            data.update({"date" : date, "hnum" : hnum})
            self.baseline = pd.concat([self.baseline, pd.DataFrame(data, index=[0])])

            data = ar.get_sharps()
            data.update({"date" : date, "hnum" : hnum})
            self.sharps = pd.concat([self.sharps, pd.DataFrame(data, index=[0])])

            data = ar.get_graph()
            if not hasattr(self, "graph_labels") or self.graph_labels == None:
                self.graph_labels = (hnum, data[1])
            self.graphs.append((date, data[0]))


            if frame_dir is not None:

                ar.show_graph(ax1, ax2)
                ar.draw_graph(ax3)

                fig.set_figwidth(20)

                fig.set_figwidth(20)
                ax1.set_title("Original Continuum")
                ax2.set_title("Segmented Umbras")
                ax3.set_title("Raw Graph")
                plt.savefig(os.path.join(frame_dir, str(self.len) + ".png"))
                fig.clf()
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

            self.len += 1
