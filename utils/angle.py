import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """
    Computes the 2D rotation matrix corresponding to a given input angle.

    Parameters
    ----------
    angle : float
        The rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        A 2x2 NumPy array representing the rotation matrix in 2D.

    Examples
    --------
    >>> rot_mat_2d(0.0)
    array([[1., 0.],
           [0., 1.]])
    """
    rotation_matrix = Rot.from_euler('z', angle).as_matrix()
    return rotation_matrix[:2, :2]



def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Normalize angles using modulo operation.

    By default, angles are wrapped to the interval [-π, π).
    Optional settings allow wrapping to [0, 2π) and specifying degrees.

    Parameters
    ----------
    x : float or array_like
        Single angle or array of angles. The input will be flattened internally.
    zero_2_2pi : bool, optional
        If True, normalize angles to the interval [0, 2π) or [0°, 360°).
    degree : bool, optional
        If True, assumes input angles are in degrees and returns results in degrees.

    Returns
    -------
    float or ndarray
        The normalized angle(s), either as a float or a NumPy array.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.2831853071795862

    >>> angle_mod([-4.0])
    array([2.28318531])

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])
    """
    is_scalar = isinstance(x, float)

    angles = np.ravel(np.asarray(x))  # Ensure 1D array

    if degree:
        angles = np.deg2rad(angles)

    if zero_2_2pi:
        wrapped = angles % (2 * np.pi)
    else:
        wrapped = (angles + np.pi) % (2 * np.pi) - np.pi

    if degree:
        wrapped = np.rad2deg(wrapped)

    return wrapped.item() if is_scalar else wrapped

