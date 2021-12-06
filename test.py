import numpy as np
x = np.array([1,1,1,2,2,2,5,25,1,1])
unique, counts = np.unique(x, return_counts=True)

print (np.asarray((unique, counts)).T)