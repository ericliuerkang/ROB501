import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---

    #------------------

    a = []
    b = []
    for pt in range(4):
        x = I1pts[0][pt]
        y = I1pts[1][pt]
        u = I2pts[0][pt]
        v = I2pts[1][pt]

        #making up a, which is 8x8 for h1 to h8 (h9 set to 1)
        a.append([-x,-y,-1,-0,-0,-0,x*u,y*u])
        a.append([-0,-0,-0,-x,-y,-1,x*v,y*v])

        #making up b, which is 8x1 for h1 to h8
        b.append(-u)
        b.append(-v)

    #convert a and b to np arrays
    A = np.array(a)
    B = np.array(b)

    #solve for h
    h = np.linalg.solve(A, B) 
    #append the 1 and reshaping h from 9x1 to 3x3
    h = np.append(h,1)
    H = np.reshape(h,(-1,3))

    #since A is needed, append negative B as the last column of A
    for i in range(len(a)):
        a[i].append(-b[i])

    A = np.array(a)

    return H, A

