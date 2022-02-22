from flares.data import get_dates
from flares.ARDataSet import ARDataSet
import numpy as np

if __name__ == '__main__':
    root = "/srv/data/thli2739"
    dates = get_dates(7115, root)
    

    dates = get_dates(7115, root, sort = True)
    
    from memory_profiler import profile
    
    def main():
        ar = ARDataSet(7115, root, dates[23:25], verbose = True)
        
    main()
