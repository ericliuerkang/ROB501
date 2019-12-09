import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline.

    Parameters:
    -----------
    Kl   - 3 x 3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3 x 3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---

    # Compute baseline (right camera translation minus left camera translation).
    Or = np.reshape(Twr[:-1,-1],(3,1))
    Ol = np.reshape(Twl[:-1,-1],(3,1))
    b = Or - Ol
    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
    rL = Twl[:3,:3]@inv(Kl)@np.vstack((pl,1))
    rayl = rL/norm(rL)
    rR = Twr[:3,:3]@inv(Kr)@np.vstack((pr,1))
    rayr = rR/norm(rR)
    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    ml = ((b.T@rayl)-(b.T@rayr)*(rayl.T@rayr))/(1-(rayl.T@rayr)**2)
    mr = rayl.T@rayr*ml - b.T@rayr
    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.
    Pl = Ol + rayl*ml
    Pr = Or + rayr*mr
    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.

    # Add code here...
    #l = (norm(rL)*Twl[:3,:3]@inv(Kl) - rayl@rayl.T@Twl[:3,:3]@inv(Kl)/norm(rL))/(norm(rL)**2)
    #r = (norm(rR)*Twr[:3,:3]@inv(Kr) - rayr@rayr.T@Twr[:3,:3]@inv(Kr)/norm(rR))/(norm(rR)**2)
    l = (1/norm(rL))*(np.eye(3)-rayl@rayl.T)@Twl[:3,:3]@inv(Kl)
    r = (1/norm(rR))*(np.eye(3)-rayr@rayr.T)@Twr[:3,:3]@inv(Kr)

    drayl[:,:2] = l[:,:2]
    drayr[:,2:] = r[:,:2]
    #------------------

    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2

    #--- FILL ME IN ---

    # 3D point.
    P = (Pl+Pr)/2
    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    middle = np.zeros((4,4))
    middle[:2,:2] = Sl
    middle[2:,2:] = Sr
    S = JP@middle@JP.T
    #------------------

    return Pl, Pr, P, S