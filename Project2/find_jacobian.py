import numpy as np

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
    """
    #--- FILL ME IN ---

#finding tx ty tz:
    #get y2
    R = Twc[0:3,0:3].T
    t = np.array([Twc.T[3][:3]]).T
    y2 = np.matmul(R,(Wpt - t))

    #get Dy3Dy2
    Dy3Dy2 = [[1/y2[2][0],          0, -y2[0][0]/(y2[2][0])**2],
              [0,          1/y2[2][0], -y2[1][0]/(y2[2][0])**2],
              [0,                   0,                       0]]
    Dy3Dy2 = np.array(Dy3Dy2)

    #get DxDt
    DxDt = (-1)*np.matmul(K, np.matmul(Dy3Dy2, R))

#finding r p q
    #finding Dy2Dtheta
    rpy = rpy_from_dcm(R.T)
    C1 = dcm_from_rpy(np.array([rpy[0][0],0,0]))
    C2 = dcm_from_rpy(np.array([0,rpy[1][0],0]))
    C3 = dcm_from_rpy(np.array([0,0,rpy[2][0]]))

    #print(C1,'\n','\n',C2,'\n','\n',C3)

    cr13 = [[0, -1, 0],
            [1,  0, 0],
            [0,  0, 0]]
    cr12 = [[0,  0, 1],
            [0,  0, 0],
            [-1, 0, 0]]
    cr11 = [[0, 0,  0],
            [0, 0, -1],
            [0,  1, 0]]

    h3 = np.matmul(cr13, np.matmul(C3, np.matmul(C2, C1))).T
    Dy2Dt3 = np.matmul(h3, (Wpt-t))
    h2 = np.matmul(C3, np.matmul(cr12, np.matmul(C2, C1))).T
    Dy2Dt2 = np.matmul(h2, (Wpt-t))
    h1 = np.matmul(C3, np.matmul(C2, np.matmul(cr11, C1))).T
    Dy2Dt1 = np.matmul(h1, (Wpt-t))

    Dy2Dtheta = np.hstack((Dy2Dt1, Dy2Dt2, Dy2Dt3))

    DxDtheta = np.matmul(K, np.matmul(Dy3Dy2, Dy2Dtheta))

#form the Jacobian
    J = np.hstack((DxDt, DxDtheta))[:2]

    #------------------

    return J

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])
