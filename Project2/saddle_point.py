import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
    #--- FILL ME IN ---

    m, n = I.shape

    #compute the inputs to the function lstsq

    #get sci
    sci = I.reshape(m*n, 1)
    #get A
    A = []
    for y in range(n):
        for x in range(m):
            #print((x,y))
            #print([x*x, x*y, y*y, x, y, 1])
            A.append([x*x, x*y, y*y, x, y, 1])

    A = np.array(A)
    print(A.shape)
    
    parms = np.linalg.lstsq(A,sci)[0]
    #print(parms)
    r1 = np.array([[2*parms[0][0], parms[1][0]], 
                   [parms[1][0], 2*parms[2][0]]])
    r1 = np.linalg.inv(r1)
    r2 = np.array([[parms[3][0]], 
                   [parms[4][0]]])

    pt = np.negative(np.matmul(r1, r2))

    #------------------

    return pt
