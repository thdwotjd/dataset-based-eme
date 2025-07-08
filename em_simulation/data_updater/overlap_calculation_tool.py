import numpy as np
from copy import deepcopy
import ray

#region normalize field and overlap calculation
def normalize_field(Efield, Hfield, x, y, prop_axis):
    """
    Parameters:
        - Efields: ndarray of shape ((number of points), (number of modes), 3, len(x), len(y))
        - Hfields: ndarray of shape ((number of points), (number of modes), 3, len(x), len(y))
        - x: x_mexh data (1d array)
        - y: y_mesh data (1d array)
        - prop_axis: propagation axis in FDE file (0: x-aixs, 1: y-axis, 2: z-axis) \n
    Returns:
        - normalized_Efields: ndarray of shape ((number of modes), 3, len(x), len(y))
        - normalized_Hfields: ndarray of shape ((number of modes), 3, len(x), len(y))
    """    
    x = compute_differences(x)
    y = compute_differences(y)

    weight_mask = np.outer(x, y)
    weight_mask_expanded = weight_mask[np.newaxis,np.newaxis,:,:]

    # integrand
    integrand = np.cross(Efield, Hfield, axis=2)
    integrand = integrand[:,:,prop_axis]
    
    # integral to get normalization constant
    normalization_constants = (weight_mask_expanded*integrand/2).sum(axis=(2,3))
    normalization_constants = np.sqrt(normalization_constants)
    normalization_constants_expanded = normalization_constants[:,:,np.newaxis, np.newaxis, np.newaxis]

    # normalize fields
    normalized_Efields = Efield/normalization_constants_expanded
    normalized_Hfields = Hfield/normalization_constants_expanded

    return normalized_Efields, normalized_Hfields

@ray.remote
def calc_overlap(Efield, Hfield, x, y, prop_axis):
    """
    Parameters:
        - Efields: normalized fields of ndarray with shape ((number of modes), 3, len(x), len(y))
        - Hfields: normalized fields of ndarray with shape ((number of modes), 3, len(x), len(y))
        - x: x_mexh data (1d array)
        - y: y_mesh data (1d array)
        - prop_axis: propagation axis in FDE file (0: x-aixs, 1: y-axis, 2: z-axis) \n
    Returns:
        - overlap: ndarray of shape ((number of modes), (number of modes))
            overlap[i,j] is overlap between Efields[i] and Hfields[j]
    """
    overlap = cross_product_2d_matrices(Efield, Hfield, is1conjugate=0, is2conjugate=0)
    overlap = integral_2d_matrices(overlap, x, y)
    overlap = overlap[:,:,prop_axis]/2
    return overlap

#endregion normalize field and overlap calculation

def correct_gain(neff):
    """
    neff: imaginary value
    It imaginary part of neff has negative sign, it force imaginary part to zero. 
    """
    neff_imag = neff.imag
    if neff_imag < 0: neff_imag = 0
    return neff.real + neff_imag*1j

def correct_gain_modified(neff):
    """
    neff: np array of imaginary values
    It imaginary part of neff has negative sign, it force imaginary part to zero. 
    """
    neff_imag = neff.imag
    neff_imag = np.where(neff_imag < 0, 0, neff_imag)
    return neff.real + neff_imag*1j

#region basic vector functions

def cross_product_2d_matrices(vec1s, vec2s, is1conjugate = 0, is2conjugate = 0):
    """
    Parameters:
        - vec1s: ndarray of shape ((number of modes 1), 3, len(x), len(y))
        - vec2s: ndarray of shape ((number of modes 2), 3, len(x), len(y)) \n
    
    Returns:
        - cross_product: ndarray of shape ((number of modes 1), (number of modes 2), 3, len(x), len(y))\n
        cross_products[i,j] is cross product between vec1s[i] and vec2s[j]

    """
    if is1conjugate: vec1s = np.conjugate(vec1s)
    if is2conjugate: vec2s = np.conjugate(vec2s)

    # Add new axes for broadcasting
    vec1s_expanded = vec1s[:, np.newaxis, :, :, :]
    vec2s_expanded = vec2s[np.newaxis, :, :, :, :]
    
    # Compute cross products using broadcasting
    cross_product = np.cross(vec1s_expanded, vec2s_expanded, axis=2)
    
    return cross_product

def integral_2d_matrices(func, x, y):
    """
    Parameters:
        - func: ndarray of shape ((number of modes 1), (number of modes 2), 3, len(x), len(y))
        - x, y: x and y coordinates for integration (1D array) \n
    
    Returns:
        - result: ndarray of shape((number of modes 1), (number of modes 2), 3)
    """    
    x = compute_differences(x)
    y = compute_differences(y)

    weight_mask = np.outer(x, y)

    # The shape of weight_mask should be the same as func
    weight_mask_expanded = weight_mask[np.newaxis, np.newaxis,np.newaxis,:,:]
    
    result = (weight_mask_expanded * func).sum(axis=(3,4))
    return result

####
def compute_differences(arr):
    # Initialize an empty array with the same length as the input array
    output = np.empty_like(arr)
    # Calculate the difference between adjacent elements and store in the output array
    output[:-1] = arr[1:] - arr[:-1]
    # Set the last component as the prior component
    output[-1] = output[-2]
    
    return output

#endregion basic vector functions

