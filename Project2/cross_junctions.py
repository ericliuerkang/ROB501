import numpy as np
from scipy.ndimage.filters import *

def cross_junctions(I, bounds, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I       - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bounds  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts    - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of I. These should be floating-point values.
    """
    #--- FILL ME IN ---

    Ipts = np.zeros((2, 48))

#parameters
    alpha = 0.15 #typically 0.04 to 0.06
    threshold = 1500 #default 2000
    sigma = 2
    ws = 12 #window size for saddle point

#building Harris Detecter
    I = I/255.0
    gradx, grady = np.gradient(I)
    IxIx = gaussian_filter(gradx*gradx,sigma)
    IxIy = gaussian_filter(gradx*grady,sigma)
    IyIy = gaussian_filter(grady*grady,sigma)
    print(I.shape)

    #get harris score
    cand_score = []
    cand_index = []
    cand = []
    s_cand = []

    for j in range(len(I)):
        for i in range(len(I[0])):
            a11 = IxIx[j][i]
            a12 = IxIy[j][i]
            a21 = a12
            a22 = IyIy[j][i]
            A = np.array([[a11, a12],[a21, a22]])
            ev0, ev1 = np.linalg.eigvals(A)
            h_score = ev0*ev1 - alpha*(ev0+ev1)**2
            cand_score.append(-h_score)
            cand_index.append([i, j])

    #get the coordinates of the top 5000 scores
    sorted_ind = np.argsort(cand_score)
    sorted_score = np.sort(cand_score).tolist()

    for ind in sorted_ind[:threshold]:
        cand.append(cand_index[ind])
    s_cand = sorted_score[:threshold]


#clustering
    #using homography to project candidate points to a up-front view
    new_bbox = np.array([[0, 100, 100, 0],[0, 0, 80, 80]])
    H = dlt_homography(bounds, new_bbox)
    cand = np.array(cand).T
    cand = np.vstack((cand, np.ones(cand.shape[1])))
    Ho_cand = np.matmul(H,cand).T
    for pt in Ho_cand:
        pt[0] = pt[0]/pt[2]
        pt[1] = pt[1]/pt[2]
    Ho_cand = Ho_cand[:,:2]
    Ho_cand = Ho_cand.tolist()

    #get rid of points that are not in the boundry
    temp_Ho_cand = []
    temp_s_cand = []
    for i in range(len(Ho_cand)):
        pt = Ho_cand[i]
        if (pt[0]>=100) or (pt[0]<0) or (pt[1]>=80) or (pt[1]<0):
            continue
        else:
            temp_Ho_cand.append(pt)
            temp_s_cand.append(s_cand[i])
    Ho_cand = np.array(temp_Ho_cand)
    s_cand = temp_s_cand
    #divide candidates into clusters
    assignment = []
    assignment_score = []

    #first put in the point that has the highest score
    assignment.append([Ho_cand[0]])
    assignment_score.append([s_cand[0]])
    for i in range(len(Ho_cand)):
        pt = Ho_cand[i]
        dist = []
        for c in assignment:
            dist.append(np.linalg.norm(pt - c[0]))
        if min(dist) > 6:
            assignment.append([pt])
            assignment_score.append([s_cand[i]])

    assignment = np.array(assignment)

    #assign points to clusters
    for i in range(len(Ho_cand)):
        pt = Ho_cand[i]
        if (pt[0] == Ho_cand[0][0]) and (pt[1] == Ho_cand[0][1]):
            continue
        dist = []
        for c in assignment:
            dist.append(np.linalg.norm(pt - c[0]))
        index = np.argsort(dist)[-1]
        np.append(assignment[index], pt)
        assignment_score[index].append(s_cand[i])

    #get centroids for each cluster
    Ho_centroids = []
    for i in range(len(assignment)):
        cl = assignment[i]
        cl = np.array(cl)
        Ho_centroids.append([np.mean(cl.T[0]),np.mean(cl.T[1])])
        assignment_score[i] = sum(assignment_score[i])

    print(len(assignment_score))

    Ho_centroids = np.array(Ho_centroids)
    #get rid of edge points

    xmin = np.amin(Ho_centroids.T[0])
    xmax = np.amax(Ho_centroids.T[0]) 
    ymin = np.amin(Ho_centroids.T[1])
    ymax = np.amax(Ho_centroids.T[1])

    final_cand = []
    final_score = []
    for i in range(len(Ho_centroids)):
        pt = Ho_centroids[i]
        if (abs(pt[0] - xmin) <= 3) or (abs(pt[0] - xmax) <= 3) or (abs(pt[1] - ymin) <= 3) or (abs(pt[1] - ymax) <= 3):
            continue
        else:
            final_cand.append(pt)
            final_score.append(assignment_score[i])
    print("Number of corner found: ")
    print(len(final_cand))

    #get rid of fake corners
    if (len(final_cand)>48):
        ultimate_cand =[]
        for ind in np.argsort(final_score)[:48]:
            ultimate_cand.append(final_cand[ind])
        final_cand = ultimate_cand
        print("real corners count:", len(ultimate_cand))


    #sort the points
    final_cand = np.array(final_cand)
    y_sort_ind = np.argsort(final_cand.T[1])
    final_cand = final_cand.tolist()
    rows = []
    for i in range(6):
        row = []
        for ind in y_sort_ind[i*8:(i+1)*8]:
            row.append(final_cand[ind])
        rows.append(row)

    ordered = []
    for row in rows:
        r = []
        x_sort_ind = np.argsort(np.array(row).T[0])
        for ind in x_sort_ind:
            r.append(row[ind])
        ordered.append(r)

    final_cand = []
    for row in ordered:
        for pt in row:
            final_cand.append(pt)
    


    #get coordinates of the centroids in the original frame
    Ho_centroids = np.array(final_cand)

    centroids = np.vstack((Ho_centroids.T, np.ones(Ho_centroids.shape[0])))
    centroids = np.matmul(np.linalg.inv(H), centroids).T
    for pt in centroids:
        pt[0] = int(pt[0]/pt[2])
        pt[1] = int(pt[1]/pt[2])
    centroids = centroids[:,:2]

#finding saddle points around the centroids
    saddle_points = []
    for pt in centroids:
        img = I[int(pt[1]-ws):int(pt[1]+ws), int(pt[0]-ws):int(pt[0]+ws)]
        saddle = saddle_point(img)
        saddle = [saddle[0][0]+pt[0]-ws, saddle[1][0]+pt[1]-ws]
        saddle_points.append(saddle)

    saddle_points = np.array(saddle_points)
    #------------------
    print(saddle_points.T)
    return saddle_points.T

def dlt_homography(I1pts, I2pts):
    
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

    return H

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

