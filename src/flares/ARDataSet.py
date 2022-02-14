from flares.active_region import ActiveRegion


class ARDataSet:
    def __init__(self, hnum, date_range, root):
        """Active region range

        Args:
            hnum ([type]): [description]
            date_range ([type]): [description]
            root ([type]): [description]
        """

        self.segmented = []
        self.baseline = []
        self.sharps = []
        self.graphs = []

        for date in date_range:
            ar = ActiveRegion(hnum, date, root)
            self.segmented.append(ar.get_segmented()[0])
            self.baseline.append(ar.get_baseline()[0])
            self.sharps.append(ar.get_sharps()[0])
            self.sharps.append(ar.get_graph()[0])
