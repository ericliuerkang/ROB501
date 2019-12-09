import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    #--- FILL ME IN ---

    # Your code goes here.
    
    #------------------
    #parameter - radius of the rank window
    r = 3
    #generate left and right images after rank tsf
    Ilr, Irr = rank_tsf(Il,Ir, r)
    #get initial Id
    Id = stereo_disparity_fast(Ilr, Irr, bbox, maxd)
    #smooth Id
    Id = gaussian_filter(Id,sigma = 0.7)
    return Id

def rank_tsf(Il, Ir, r):
    Rl = np.zeros(Il.shape)
    Rr = np.zeros(Ir.shape)
    for j in range(r,len(Il)-r):
        for i in range(r,len(Il[0])-r):
            #make windows and find center of the windows
            wl = Il[j-r:j+r,i-r:i+r]
            wr = Ir[j-r:j+r,i-r:i+r]
            cl = Il[j,i]
            cr = Ir[j,i]
            #generate left and right images after rank tsf
            Rl[j,i]= rank(wl, cl)
            Rr[j,i]= rank(wr,cr)
    return Rl, Rr

def rank(w,c):
    #returns number of neighbours larger than P
    w = w>c
    return np.sum(w)

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    #same function from part1 with different window size
    ws = 25

    ker = np.ones((ws,ws))
    last = np.ones(Il.shape)*float('inf')
    Ids = np.zeros(Il.shape)

    for i in range(1,maxd+1):
        I = -10*np.ones(Il.shape)
        I[:,i:] = Ir[:,:-i]
        raw_diff = np.abs(Il-I)
        sad = convolve(raw_diff,ker)
        for j in range(bbox[1][0],bbox[1][1]):
            for k in range(bbox[0][0],bbox[0][1]):
                if sad[j,k] < last[j,k]:
                    Ids[j,k] = i
                    last[j,k] = sad[j,k]

    #Idsn = 255*Ids / np.linalg.norm(Ids) 
    Id = Ids
    #------------------

    return Id



