'''
Script containing useful functions used by many scripts associated with the SysID library

Created by: Jasper van Beers
'''
# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d


# ================================================================================================================================ #
# Functions
# ================================================================================================================================ #
def Continuous2DiscreteAB(A, B, dt, approximation_terms = 5):
    '''Function to convert continuous model to discrete time
    
    :param A: State space matrix
    :param B: Input matrix
    :param dt: Sampling time, in seconds
    :param approximation_terms: Number of terms, as int, in Taylor series expansion used to compute discrete variant. Default = 5. 

    :return: Tuple of discrete (A, B)
    '''
    # Due to the nature of np.linalg, A needs to be a square matrix for operations to work
    # Here, we artificially make A a square matrix - if it is not - to perform necessary calculations
    # Note that this augmentation does not affect the result
    checkA = True
    counter = 0
    while checkA:
        if A.shape[0] == A.shape[1]:
            checkA = False
        else:
            A = np.vstack((A, np.zeros(A.shape[1])))
            B = np.vstack((B, np.zeros(B.shape[1])))
            counter += 1

    # Define discrete counterparts
    n = A.shape[0]
    Ak = np.eye(n)
    Bk = B*dt

    # Compute Taylor up to approximation_terms
    for i in range(approximation_terms):
        Ak += 1/np.math.factorial(i + 1) * A ** (i + 1) * dt ** (i + 1)
        Bk += 1/np.math.factorial(i + 2) * A ** (i + 1) * B * dt ** (i + 2)

    # Revert A back to its original shape
    if counter > 0:
        Ak = Ak[:-counter]

    return np.matrix(Ak), np.matrix(Bk)



def getArgs(func, hasSelf = False):
    '''Function to get expected positional and keyword arguments of a python function
    
    :param func: Function to probe for arguments
    :param hasSelf: Boolean to indicate if the function in question is from an object (i.e. has self as the first positional argument)

    :return: (Positional arguments, Keyword arguments)
    '''
    nArgs = func.__code__.co_argcount
    if hasSelf:
        allArgs = func.__code__.co_varnames[1:nArgs]
    else:
        allArgs = func.__code__.co_varnames[0:nArgs]
    if func.__defaults__:
        nkwargs = len(func.__defaults__)
        posArgs = allArgs[0:(nArgs-nkwargs)]
        kwArgs = allArgs[-nkwargs:]
    else:
        posArgs = allArgs[0:(nArgs)]
        kwArgs = None
    return posArgs, kwArgs



def derivative(x, t):
    '''Function to numerically obtain the derivative of x with respect to t
    
    :param x: Signal to differentiate
    :param t: Signal with which x should be differentiated by

    :return: dx/dt
    '''
    xdot = np.zeros(x.shape)
    for i in range(len(t)-2):
        xdot[i + 1, :] = 0.5 * (x[i + 2, :] - x[i, :])/(0.5 * (t[i+2] - t[i]))
        if np.isinf(xdot[i + 1, :]).any():
            # Correct for inf, if any
            idx_undef = np.where(np.isinf(xdot[i + 1, :]))[0]
            xdot[i + 1, idx_undef] = xdot[i-1, idx_undef]

    # Missing first and last point, set them equal to nearest point
    xdot[0, :] = xdot[1, :]
    xdot[-1, :] = xdot[-2, :]
    # Interpolate NaNs, needs to be done per index in xdot
    for i in range(xdot.shape[1]):
        nanLocs = np.isnan(xdot[:, i])
        nanIdxs = np.where(nanLocs)[0]
        if len(nanIdxs) > 0:
            # Create a function which linearly interpolates the signal, based on known data, which takes
            # index as input -> i.e. y = f(index) -< can be thought of as a proxy for time
            func = interp1d(np.where(~nanLocs)[0], xdot[~nanLocs, i], kind = 'slinear')
            # Interpolate NaN indexes 
            xdot[nanIdxs, i] = func(nanIdxs)

    return xdot



def quatMul(Q1, Q2):
    '''Function to multiply two quaternion arrays
    
    :param Q1: First quaternion, as array with shape [N, 4] where N is the number of samples
    :param Q2: Second quaternion, as array with shape [N, 4] where N is the number of samples

    :return: Product of Q1 and Q2
    '''
    Q_out = np.array([[Q1[:, 0]*Q2[:, 0] - Q1[:, 1]*Q2[:, 1] - Q1[:, 2]*Q2[:, 2] - Q1[:, 3]*Q2[:, 3]],
                    [Q1[:, 0]*Q2[:, 1] + Q1[:, 1]*Q2[:, 0] + Q1[:, 2]*Q2[:, 3] - Q1[:, 3]*Q2[:, 2]],
                    [Q1[:, 0]*Q2[:, 2] - Q1[:, 1]*Q2[:, 3] + Q1[:, 2]*Q2[:, 0] + Q1[:, 3]*Q2[:, 1]],
                    [Q1[:, 0]*Q2[:, 3] + Q1[:, 1]*Q2[:, 2] - Q1[:, 2]*Q2[:, 1] + Q1[:, 3]*Q2[:, 0]]])
    return Q_out.T.reshape(-1, 4)



def QuatRot(q, x, rot='B2E'):
    '''Function to rotate a vector using its quaternion representation. 

    :param q: Quaternion signal, as array with shape [N, 4], where N is the number of samples
    :param x: Signal to rotate, as array with shape [N, 3]
    :param rot: String indicating the order of rotation; options are 'B2E' or 'E2B'. Default = 'B2E', indicating that the rotations are from the body frame to the earth frame. Conversely, 'E2B' denotes rotations from earth frame to body frame. 
    
    :return: Rotated x
    '''
    if rot == 'B2E':
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif rot == 'E2B':
        q0, q1, q2, q3 = q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]
    else:
        raise ValueError('specified rot is unknown. Use "B2E" or "E2B" for body to earth or earth to body rotations respectively')

    # Define rotation matrices for each axis 
    R_1 = np.array([(q0*q0 + q1*q1 - q2*q2 -q3*q3), (2*(q1*q2 - q0*q3)), (2*(q0*q2 + q1*q3))])
    R_2 = np.array([(2*(q1*q2 + q0*q3)), (q0*q0 - q1*q1 + q2*q2 - q3*q3), (2*(q2*q3 - q0*q1))])
    R_3 = np.array([(2*(q1*q3 - q0*q2)), (2*(q0*q1 + q2*q3)), (q0*q0 - q1*q1 - q2*q2 + q3*q3)])

    # Manipulate the indices of the rotation matrices above to get a vector of form
    # N x [3 x 3] such that each element corresponds to the rotation matrix for that
    # specific sample and can therefore be multiplied directly with the acceleration array
    R_1 = R_1.T
    R_2 = R_2.T
    R_3 = R_3.T
    R_stack = np.zeros((3*len(R_1), 3))
    R_stack[0:(3*len(R_1)):3] = R_1
    R_stack[1:(3*len(R_1)):3] = R_2
    R_stack[2:(3*len(R_1)):3] = R_3
    R = R_stack.reshape((len(R_1), 3, 3))
    
    x_rot = np.matmul(R, x.reshape((len(x), -1, 1)))

    return x_rot.reshape(x.shape)



def Eul2Quat(theta):
    '''Function to convert euler angles to their quaternion equivalents
    
    :param theta: Array of the euler angles with shape (N, 3) or (3, N) where N is the number of samples

    :return: Quaternion representation of euler angles
    '''

    # Reshape theta
    theta = theta.reshape(-1, 3)

    quat = np.zeros((len(theta), 4))

    cr = np.cos(theta[:, 0]*0.5)
    sr = np.sin(theta[:, 0]*0.5)
    cp = np.cos(theta[:, 1]*0.5)
    sp = np.sin(theta[:, 1]*0.5)
    cy = np.cos(theta[:, 2]*0.5)
    sy = np.sin(theta[:, 2]*0.5)

    quat[:, 0] = cr*cp*cy + sr*sp*sy
    quat[:, 1] = sr*cp*cy - cr*sp*sy
    quat[:, 2] = cr*sp*cy + sr*cp*sy
    quat[:, 3] = cr*cp*sy - sr*sp*cy

    return quat



def Quat2Eul(quat):
    '''Function to convert quaternion representation of orientation to euler angles
    
    :param quat: Quaternion representation, as array with shape (N, 4), where N is the number of samples
    
    :return: Corresponding euler angles
    '''
    quat = quat.reshape(-1, 4)
    eul = np.zeros((len(quat), 3))
    eul[:, 0] = np.arctan2(2*(quat[:, 0]*quat[:, 1] + quat[:, 2]*quat[:, 3]), 1 - 2*(quat[:, 1]**2+quat[:, 2]**2))
    # Need to round to avoid issues where arcsin(x), with x = 1, is undefined due to floating point errors
    eul[:, 1] = np.arcsin(np.around(2*(quat[:, 0]*quat[:, 2] - quat[:, 3]*quat[:, 1]), 15))
    eul[:, 2] = np.arctan2(2*(quat[:, 0]*quat[:, 3] + quat[:, 2]*quat[:, 1]), 1 - 2*(quat[:, 2]**2+quat[:, 3]**2))
    return eul



def _GetQuat(theta_vec):
    '''Function to convert euler angles to their quaternion equivalents. Note: Accomplishes the same as Eul2Quat but is slower. 
    
    :param theta: Array of the euler angles with shape (N, 3) or (3, N) where N is the number of samples

    :return: Quaternion representation of euler angles
    '''

    # Preallocate quaternion vector
    quat_vec = np.zeros((4, len(theta_vec[0])))

    # For each time step
    for t in range(len(theta_vec[0])):
        # Compute the angular magnitude
        mag = np.sqrt( (theta_vec[0, t]**2 + theta_vec[1, t]**2 + theta_vec[2, t]**2) )

        # Avoid division by 0 for 0 magnitude, which would occur if all angles are 0. 
        if theta_vec[:, t].all() == 0 and mag == 0:
            mag = 1

        # Normalize the angle vector
        Nth_vec = theta_vec[:, t]/mag

        thetaOver2 = mag/2
        sinTO2 = np.sin(thetaOver2)
        cosTO2 = np.cos(thetaOver2)

        # Convert to quaternions 
        quat_vec[0][t] = cosTO2
        quat_vec[1][t] = sinTO2 * Nth_vec[0]
        quat_vec[2][t] = sinTO2 * Nth_vec[1]
        quat_vec[3][t] = sinTO2 * Nth_vec[2]

    return quat_vec



def wrapPi(angles):
    '''Function to constrain angles to [-pi, pi]
    
    :param angles: Array of angles to map to [-pi, pi]

    :return: Angles transformed to [-pi, pi]
    '''
    return (angles + np.pi) % (2*np.pi) - np.pi



def unwrapPi(angles):
    '''Function to unwrap angles from [-pi, pi]
    
    :param angles: Wrapped angles, confided to [-pi, pi]

    :return: Unwrapped angles
    '''
    return np.unwrap(angles)



def EulRotX(ang):
    '''Function to obtain matrix rotation about x-axis through an arbitrary angle 
    
    :param ang: Angle of rotation, in radians

    :return: Matrix, with shape (3, 3), corresponding to this rotation. 
    '''
    R = np.array([[1, 0, 0],
                  [0, np.cos(ang), -1*np.sin(ang)],
                  [0, np.sin(ang), np.cos(ang)]])
    return np.matrix(R)



def EulRotY(ang):
    '''Function to obtain matrix rotation about y-axis through an arbitrary angle 
    
    :param ang: Angle of rotation, in radians

    :return: Matrix, with shape (3, 3), corresponding to this rotation. 
    '''
    R = np.array([[np.cos(ang), 0, np.sin(ang)],
                  [0, 1, 0],
                  [-1*np.sin(ang), 0, np.cos(ang)]])
    return np.matrix(R)



def EulRotZ(ang):
    '''Function to obtain matrix rotation about z-axis through an arbitrary angle 
    
    :param ang: Angle of rotation, in radians

    :return: Matrix, with shape (3, 3), corresponding to this rotation. 
    '''
    R = np.array([[np.cos(ang), -1*np.sin(ang), 0],
                  [np.sin(ang), np.cos(ang), 0],
                  [0, 0, 1]])
    return np.matrix(R)



def EulRotX_arr(ang_arr):
    '''Function to obtain matrix rotation about x-axis through an arbitrary sequence of angles 
    
    :param ang: Array of angle rotations with shape (N, 1) where N is the number of samples

    :return: Matrix, with shape (N, 3, 3), corresponding to this rotation where N is the number of samples. 
    '''
    zeros = np.zeros(ang_arr.shape[0])
    ones = np.ones(ang_arr.shape[0])
    R = np.array([[ones, zeros, zeros],
                  [zeros, np.cos(ang_arr), -1*np.sin(ang_arr)],
                  [zeros, np.sin(ang_arr), np.cos(ang_arr)]])
    return np.transpose(R.T, (0, 2, 1))



def EulRotY_arr(ang_arr):
    '''Function to obtain matrix rotation about y-axis through an arbitrary sequence of angles 
    
    :param ang: Array of angle rotations with shape (N, 1) where N is the number of samples

    :return: Matrix, with shape (N, 3, 3), corresponding to this rotation where N is the number of samples. 
    '''    
    zeros = np.zeros(ang_arr.shape[0])
    ones = np.ones(ang_arr.shape[0])
    R = np.array([[np.cos(ang_arr), zeros, np.sin(ang_arr)],
                  [zeros, ones, zeros],
                  [-1*np.sin(ang_arr), zeros, np.cos(ang_arr)]])
    return np.transpose(R.T, (0, 2, 1))



def EulRotZ_arr(ang_arr):
    '''Function to obtain matrix rotation about y-axis through an arbitrary sequence of angles 
    
    :param ang: Array of angle rotations with shape (N, 1) where N is the number of samples

    :return: Matrix, with shape (N, 3, 3), corresponding to this rotation where N is the number of samples. 
    '''    
    zeros = np.zeros(ang_arr.shape[0])
    ones = np.ones(ang_arr.shape[0])
    R = np.array([[np.cos(ang_arr), -1*np.sin(ang_arr), zeros],
                  [np.sin(ang_arr), np.cos(ang_arr), zeros],
                  [zeros, zeros, ones]])
    return np.transpose(R.T, (0, 2, 1))



def qualityPI(y_true, y_pred, y_var, conf = 0.95):
    '''Function to determine the quality of the prediction intervals, based on their coverage probability (PICP) and the (normalized) mean width of the PIs (MPIW). 

    :param y_true: True target values, as 1-D array
    :param y_pred: Predicted (or estimated) target values, as 1-D array 
    :param y_var: Associated variance with predicted targets, as 1-D array
    :param conf: Confidence level used to construct the prediction intervals, Default = 0.95 (95% prediction intervals)

    :return: (PICP, MPIW)
    '''
    # Define lower and upper bounds of the prediction interval based on the confidence level
    PI_lower, PI_upper = buildIntervalBounds(conf, y_pred, y_var, N = 1)

    N = len(y_true)

    ''' Probability Coverage '''
    # Count number of samples where y_true is within the prediction interval
    n_coverage = np.where((y_true >= PI_lower) & (y_true <= PI_upper))[0]
    # Compute the probability coverage 
    PI_coverage = (len(n_coverage)/N)*100

    ''' Mean PI interval '''
    MPIW = np.nanmean((PI_upper - PI_lower), axis = 0)
    N_MPIW = (MPIW/(np.nanmax(y_true) - np.nanmin(y_true)))*100

    return PI_coverage, N_MPIW



def buildIntervalBounds(confidenceLevel, y_pred, y_var, N = 1):
    '''Function to determine lower and upper limits of a prediction interval based on a given confidence level

    :param confidenceLevel: Confidence level used to construct the prediction intervals
    :param y_pred: Predicted (or estimated) target values, as 1-D array 
    :param y_var: Associated variance with predicted targets, as 1-D array
    :param N: Number of samples for target predictions. Typically N = 1 when making point predictions, so Default = 1
    
    :return: (PI_lower, PI_upper); tuple of 1-D arrays corresponding to the lower and upper interval bounds respectively 
    '''
    if confidenceLevel >= 1:
        print('[ WARNING ] User specified a confidence interval >= 1. Defaulting to 0.99.')
        confidenceLevel = 0.99
    z_conf = norm.ppf((1+confidenceLevel)/2)
    PI_lower = y_pred.reshape(-1) - z_conf*np.sqrt(y_var.reshape(-1))/np.sqrt(N)
    PI_upper = y_pred.reshape(-1) + z_conf*np.sqrt(y_var.reshape(-1))/np.sqrt(N)
    return PI_lower, PI_upper



# End