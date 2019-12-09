import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def pose_estimate_nls(K, Twcg, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---

    E = epose_from_hpose(np.linalg.inv(Twcg))
    for iter in range(maxIters):
        Twc = np.linalg.inv(hpose_from_epose(E))

        sum1 = np.reshape(np.zeros(36),(6,6))
        sum2 = np.reshape(np.zeros(6),(6,1))
        for i in range(tp):
            Wpt = np.vstack((np.array([Wpts[0][i]]), np.array([Wpts[1][i]]), np.array([Wpts[2][i]])))
            Wpt_til = np.vstack((np.array([Wpts[0][i]]), np.array([Wpts[1][i]]), np.array([Wpts[2][i]]), np.array([1])))
            Ipt = np.vstack((np.array([Ipts[0][i]]), np.array([Ipts[1][i]])))
            J = find_jacobian(K, Twc, Wpt)

            #find error
            K_til = np.hstack((K,np.array([[0],[0],[0]])))
            K_til = np.vstack((K_til,np.array([0,0,0,1])))
            x_til = K_til@(np.linalg.inv(Twc))@Wpt_til
            x = x_til[0:2]/x_til[2]
            e = x-Ipt

            sum1 += J.T@J
            sum2 += J.T@e
        update = np.linalg.inv(sum1)@sum2

        E = epose_from_hpose(np.linalg.inv(Twc))

        E[0][0] += update[0][0]
        E[1][0] += update[1][0]
        E[2][0] += update[2][0]
        E[3][0] += update[3][0]
        E[4][0] += update[4][0]
        E[5][0] += update[5][0]

    Twc = np.linalg.inv(hpose_from_epose(E))
    #------------------
    
    #return Twc
    return Twc
#----- Functions Go Below -----

def epose_from_hpose(T):
    """Euler pose vector from homogeneous pose matrix."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Homogeneous pose matrix from Euler pose vector."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T
