# Billboard hack script file.
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    J = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H, A = dlt_homography(Iyd_pts, Ist_pts)
    a = H.dot(np.array([410,349,1]))
    #print (a/a[-1])

    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!

    #make the path    
    path = []
    for i in range(4):
        tup = (Iyd_pts[1][i],Iyd_pts[0][i])
        path.append(tup)
    path = Path(path)

    #loop through the bounding box
    for i in range(404,490):
        for j in range(38,354):
            #only treat points within the shape
            if Path.contains_points(path, [[j, i]]):
                #computing the homogeneous representation of the point
                new_coord = H.dot(np.array([i,j,1]))
                #transfering the homogeneous form to normal form
                new_coord = np.array([[new_coord[0]/new_coord[-1]],[new_coord[1]/new_coord[-1]]])
                #perform bilinear interpolation on J
                new_pt = bilinear_interp(J, new_coord)
                #write the point in
                Ihack[j][i] = [new_pt,new_pt,new_pt]
    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    #------------------

    #plt.imshow(Ihack)
    #plt.show()
    #imwrite(Ihack, 'billboard_hacked.png');
    
    return Ihack

billboard_hack()