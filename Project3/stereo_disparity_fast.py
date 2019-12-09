import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *
import matplotlib.pyplot as plt

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Ids  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoIds
    #    using values outsIdse of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    #--- FILL ME IN ---

    # Your code goes here.
    #parameter - window size
    ws = 5
    #initialization
    ker = np.ones((ws,ws))
    last = np.ones(Il.shape)*float('inf')
    Ids = np.zeros(Il.shape)

    for i in range(1,maxd+1):
        #loop through maxd and shift the image
        I = -10*np.ones(Il.shape)
        I[:,i:] = Ir[:,:-i]
        #generate AD
        raw_diff = np.abs(Il-I)
        #generate SAD
        sad = convolve(raw_diff,ker)
        for j in range(bbox[1][0],bbox[1][1]):
            for k in range(bbox[0][0],bbox[0][1]):
                #update best guess and disparities
                if sad[j,k] < last[j,k]:
                    Ids[j,k] = i
                    last[j,k] = sad[j,k]

    Id = Ids
    #------------------

    return Id

