import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    #generate histogram as a hash table, 256 because 0~255 is 256 numbers
    h = np.array([0]*256)

    for i in range(len(I)):
        for j in range(len(I[0])):
            h[I[i][j]] += 1

    #create a J that's the same shape as I
    J = I.copy()
    
    #loop through the image
    for i in range(len(I)):
        for j in range(len(I[0])):
            #define variables to be used in the while loop
            value = I[i][j]
            v = 0
            total = 0
            #loop from 0 to value adding the entries into total
            while (v <= value):
                total += h[v]
                v += 1 
            #using the equation on page 108 of Szeliski Text
            J[i][j] = round(total*255.0/float(len(I)*len(I[0])))

    #convert type
    J.astype(np.int8)
    #------------------

    return J
