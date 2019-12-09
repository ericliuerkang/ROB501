import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0  ).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    #l/r = left/right, u/d = up/down, m is middle(interpolated middle not actual middle)

    lu = float(I[int(pt[1][0]), int(pt[0][0])])
    ru = float(I[int(pt[1][0])+1, int(pt[0][0])])
    ld = float(I[int(pt[1][0]), int(pt[0][0])+1])
    rd = float(I[int(pt[1][0])+1, int(pt[0][0])+1])    


    #first compute the two horrizontal interpolations
    mu = lu+(ru-lu)*(pt[1][0]-int(pt[1][0]))
    md = ld+(rd-ld)*(pt[1][0]-int(pt[1][0]))

    #perform the vertical interpolation with the results from the horrizontal ones
    b = mu+(md-mu)*(pt[0][0]-int(pt[0][0]))

    #round b into a whole number as required
    b = round(b)

    #------------------

    return b
