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
        
        i = 0

        for date in dates:
            if verbose:
                print(f"Working on {hnum}, {date}")

            ar = ActiveRegion(hnum, date, root)
            if not ar.valid:
                if verbose:
                    print(f"skipping: {date}")
                continue

    
            self.segmented = pd.concat([self.segmented, pd.DataFrame(ar.get_segmented(), index=[0])])
            self.baseline = pd.concat([self.baseline, pd.DataFrame(ar.get_baseline(), index=[0])])
            self.sharps = pd.concat([self.sharps, pd.DataFrame(ar.get_sharps(), index=[0])])

            data = ar.get_graph()
            if not hasattr(self, "graphs_labels"):
                self.graph_labels = data[1]
            self.graphs.append(data[0])


            if frame_dir is not None:

                ar.show_graph(ax1, ax2)
                ar.draw_graph(ax3)

                fig.set_figwidth(20)

                fig.set_figwidth(20)
                ax1.set_title("Original Continuum")
                ax2.set_title("Segmented Umbras")
                ax3.set_title("Raw Graph")
                plt.savefig(os.path.join(frame_dir, str(i) + ".png"))
                fig.clf()
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

            i+=1
