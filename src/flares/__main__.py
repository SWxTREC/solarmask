from flares.data import get_dates
from flares.ARDataSet import ARDataSet
import numpy as np

if __name__ == '__main__':
    root = "../docs/example_data/raw"
    ar = ARDataSet(7115, root, verbose = True)
