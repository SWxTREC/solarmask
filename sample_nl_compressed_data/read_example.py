from solarmask.data import read_mask
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Read mask takes in harpnumber (3344) datetime and location of neutral_line folder. On swami, you can use "/srv/thli2739" for now
mask = read_mask(3344, datetime(2013, 11, 3, 3, 0, 0), ".")

print(mask)
print(np.count_nonzero(mask))
print(mask.shape)
