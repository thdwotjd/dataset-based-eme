import numpy as np
from copy import deepcopy
import sys
import os
# em_simulator_path = os.path.abspath("..")
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
em_simulator_path = os.path.dirname(current_directory)


import ray


if not ray.is_initialized():
    ray.init(ignore_reinit_error=True,
             runtime_env={
            "working_dir" : em_simulator_path,
            "excludes" : [".git/", "sample_datasets"]
            }
    )


#region convert between smatrix and tmatrix
def _convert_3Dmatrix(matrix):
    """
    matrix: 3-dimension ndarray where matrix[i] is i-th section Tmatrix or Smatrix
    convert Tmatrix-> Smatrix or Smatrix->Tmatrix
    """
    # tolerance = 0.1 # if specfic mode energy transfer to the next section is less than tolerance, ignore the mode 
    tolerance = 0.9
    mode_count = int(matrix.shape[1]/2)
    m11, m12, m21, m22 = _extract_blockmatrix_3D(matrix)
    n11 = m11 - m12 @ _inverse_3D_matrix(m22, tolerance=tolerance) @ m21
    n12 = (-1) * m12 @ _inverse_3D_matrix(m22, tolerance=tolerance)
    n21 = (-1) * _inverse_3D_matrix(m22, tolerance=tolerance) @ m21
    n22 = _inverse_3D_matrix(m22, tolerance=tolerance)

    result = np.zeros_like(matrix)
    result[:,:mode_count, :mode_count] = n11
    result[:,:mode_count, mode_count:] = n12
    result[:, mode_count:, :mode_count] = n21
    result[:, mode_count:, mode_count:] = n22

    return result

def _convert_3Dmatrix_ray(matrix):
    """
    matrix: 3-dimension ndarray where matrix[i] is i-th section Tmatrix or Smatrix
    convert Tmatrix-> Smatrix or Smatrix->Tmatrix
    """
    # tolerance = 0.1 # if specfic mode energy transfer to the next section is less than tolerance, ignore the mode 
    tolerance = 0.9
    mode_count = int(matrix.shape[1]/2)
    m11, m12, m21, m22 = _extract_blockmatrix_3D(matrix)
    inverse_m22 = _inverse_3D_matrix_ray(m22, tolerance=tolerance)
    n11 = m11 - m12 @ inverse_m22 @ m21
    n12 = (-1) * m12 @ inverse_m22
    n21 = (-1) * inverse_m22 @ m21
    n22 = inverse_m22

    result = np.zeros_like(matrix)
    result[:,:mode_count, :mode_count] = n11
    result[:,:mode_count, mode_count:] = n12
    result[:, mode_count:, :mode_count] = n21
    result[:, mode_count:, mode_count:] = n22

    return result

def _convert_2Dmatrix(matrix):
    """
    matrix: 2-dimension ndarray where matrix[i] is i-th section Tmatrix or Smatrix
    convert Tmatrix-> Smatrix or Smatrix->Tmatrix
    """
    # tolerance = 0.1
    tolerance = 0.9
    m11, m12, m21, m22 = _extract_blockmatrix_2D(matrix)
    n11 = m11 - m12 @ _inverse_2D_matrix(m22, tolerance=tolerance) @ m21
    n12 = (-1) * m12 @ _inverse_2D_matrix(m22, tolerance=tolerance)
    n21 = (-1) * _inverse_2D_matrix(m22, tolerance=tolerance) @ m21
    n22 = _inverse_2D_matrix(m22, tolerance=tolerance)

    mode_count = len(n11)

    result = np.zeros_like(matrix)
    result[:mode_count, :mode_count] = n11
    result[:mode_count, mode_count:] = n12
    result[mode_count:, :mode_count] = n21
    result[mode_count:, mode_count:] = n22


    return result
#endregion convert between smatrix and tmatrix

#region inverse non-zero submatrix

def _inverse_3D_matrix(matrix, tolerance = 1e-9, verbose=True):
    """
    Parameter:
        - matrix: ndarray of square matrix with shape (m,n,n)
    Returns:
        - matrix whose nonzero submatrix is inversed
    """
    # matrix shape validity test
    matrix = deepcopy(matrix)
    if not len(matrix.shape) == 3:
        print("matrix should be shape of (m,n,n)")
        return
    if not matrix.shape[1] == matrix.shape[2]:
        print("matrix should be shape of (m,n,n)")
        return

    result = np.zeros_like(matrix)
    # sectionally inverse the matrix
    for i in range(len(matrix)):
        result_temp = _inverse_2D_matrix(matrix[i], tolerance)
        result[i] = result_temp

    if np.isnan(result).any():  # check matrix 
        result = np.zeros_like(matrix)
        for i in range(len(matrix)):
            result_temp = _inverse_2D_matrix_avoid_nan(matrix[i], tolerance, verbose=False)
            result[i] = result_temp
        if np.isnan(result).any() and verbose: print("There is NaN in an inversed matrix")
        
    return result


def _inverse_2D_matrix(matrix, tolerance = 1e-9):
    """
    Parameter:
        - matrix: ndarray of square matrix with shape (n,n)
    Returns:
        - matrix whose nonzero submatrix is inversed
    """
    matrix = deepcopy(matrix)
    nonzero_indices = _find_nonzero_index(matrix, tolerance)
    nonzero_submatrix = matrix[np.ix_(nonzero_indices, nonzero_indices)]
    submatrix_inverse = np.linalg.inv(nonzero_submatrix)

    new_matrix = np.zeros_like(matrix)
    new_matrix[np.ix_(nonzero_indices, nonzero_indices)] = submatrix_inverse

    return new_matrix

def _inverse_2D_matrix_avoid_nan(matrix, tolerance = 1e-9, max_trial_num=5, verbose=True):
    """
    Parameter:
        - matrix: ndarray of square matrix with shape (n,n)
    Returns:
        - matrix whose nonzero submatrix is inversed
    """
    matrix = deepcopy(matrix)
    nonzero_indices = _find_nonzero_index(matrix, tolerance)
    nonzero_submatrix = matrix[np.ix_(nonzero_indices, nonzero_indices)]
    submatrix_inverse = np.linalg.inv(nonzero_submatrix)
    trial_num = 0
    while np.isnan(submatrix_inverse).any() and trial_num < max_trial_num:
        submatrix_inverse = np.linalg.inv(nonzero_submatrix)
        trial_num += 1

    if np.isnan(submatrix_inverse).any() and verbose:
        print("There is NaN in an inversed matrix")
    new_matrix = np.zeros_like(matrix)
    new_matrix[np.ix_(nonzero_indices, nonzero_indices)] = submatrix_inverse

    return new_matrix

def _find_nonzero_index(matrix, tolerance = 1e-9):
    """
    Parameter:
        - matrix: ndarray of square matrix with shape (n,n)
    Returns:
        - list of indices of nonzero dimension
    """
    if not len(matrix.shape) == 2:
        print("matrix should be shape of (n,n)")
        return
    if not matrix.shape[0] == matrix.shape[1]:
        print("matrix should be shape of (n,n)")
        return
    
    abs_matrix = np.abs(matrix)

    dim = matrix.shape[0]
    nonzero_indices = []
    for i in range(dim):
        # first, check diagonal term
        # if np.abs(matrix[i,i]) > tolerance:
        if abs_matrix[i,i] > tolerance:
            nonzero_indices.append(i)
        # if it does not passed the diagnoal test, check the column and row values.
        else:
            # sum_row = np.sum(np.abs(matrix[i, :]))
            # sum_col = np.sum(np.abs(matrix[:, i]))
            sum_row = np.sum(abs_matrix[i, :])
            sum_col = np.sum(abs_matrix[:, i])
            if sum_row > tolerance and sum_col > tolerance: nonzero_indices.append(i)
            
    return nonzero_indices

#endregion inverse non-zero submatrix

#region extract block submatrix

def _extract_blockmatrix_3D(matrix):
    """
    matrix: ndarray with shape (section_count, 2*mode_count, 2*mode_count)
    when matrix M is defined as
    M[i] = (M[i]_11, M[i]_12)
            (M[i]_21, M[i]_22)
            where i is the section number
            and M is shape of (2*mode_count, 2*mode_count)
    this method extracts M_11, M_12, M_21 and M_22
    where M_ij[k] = M[k]_ij
    M_ij have all same dimension with each others
    """
    new_matrix = deepcopy(matrix)
    mode_count = int(new_matrix.shape[1]/2)
    m11 = new_matrix[:,:mode_count, :mode_count]
    m12 = new_matrix[:,:mode_count, mode_count:]
    m21 = new_matrix[:, mode_count:, :mode_count]
    m22 = new_matrix[:, mode_count:, mode_count:]
    return m11, m12, m21, m22

def _extract_blockmatrix_2D(matrix):
    """
    matrix: ndarray with shape (2*mode_count, 2*mode_count)
    when matrix M is defined as
    M = (M_11, M_12)
            (M_21, M_22)
            and M is shape of (2*mode_count, 2*mode_count)
    this method extracts M_11, M_12, M_21 and M_22
    M_ij have all same dimension with each others
    """
    new_matrix = deepcopy(matrix)
    mode_count = int(len(new_matrix)/2)
    m11 = new_matrix[:mode_count, :mode_count]
    m12 = new_matrix[:mode_count, mode_count:]
    m21 = new_matrix[mode_count:, :mode_count]
    m22 = new_matrix[mode_count:, mode_count:]

    return m11, m12, m21, m22
#endregion extract block submatrix

def _redheffer_star_product(matrix1, matrix2):
    """
    It calculates the redheffer star product of matrix1 and matrix2
    It calculate the combined scattering matrix if matrix1 and matrix2 are scattering matrices.
    matrix1: 2d scattering matrix of system1
    matrix2: 2d scattering matrix of system2
    the resultant matrix is of combined system
    in -> system1 -> system2 -> out  ==   in -> system -> out
    """
    mode_count = int(len(matrix1)/2)
    a11, a12, a21, a22 = _extract_blockmatrix_2D(matrix1)
    b11, b12, b21, b22 = _extract_blockmatrix_2D(matrix2)
    
    I = np.eye(mode_count, dtype=complex)
    common_matrix1 = b11 @ np.linalg.inv(I - a12 @ b21)
    common_matrix2 = a22 @ np.linalg.inv(I - b21 @ a12)

    c11 = common_matrix1 @ a11
    c12 = common_matrix1 @ a12 @ b22 + b12
    c21 = a21 + common_matrix2 @ b21 @ a11
    c22 = common_matrix2 @ b22

    result = np.empty_like(matrix1, dtype = complex)

    result[:mode_count, :mode_count] = c11
    result[:mode_count, mode_count:] = c12
    result[mode_count:, :mode_count] = c21
    result[mode_count:, mode_count:] = c22

    return result

#region energy related functions
def _make_passive_2D(smatrix):
    """
    It makes the scattering matrix passive by forcing L2-norm less than or equal to 1.
    Parameters
        - smatrix: scattering matrix with dimension (m, m)
    Returns
        - smatrix: ndarray L2-norm not larger than 1
    """
    smatrix = deepcopy(smatrix)
    norm = np.linalg.norm(smatrix, axis=(0,1), ord=2)
    if norm < 1: norm = 1
    smatrix = smatrix/norm

    return smatrix

def _find_nearest_unitary_2D(smatrix):
    """
    It finds nearest unitary matrix to the given smatrix. 
    If smatrix does not includes all mode inforamtion, it returns the matrix with nonzero-submatrix is replaced by the nearest unitary
    
    Parameters:
        -smatrix: scattering matrix with dimension (n, m, m) where n corresponds to the section number
    Returns
        - smatrix: ndarray with each section ndarray becomes unitary nearest to the original matrix
    """
    smatrix = deepcopy(smatrix)
    nonzero_indices = _find_nonzero_index(smatrix)
    nonzero_submatrix = smatrix[np.ix_(nonzero_indices, nonzero_indices)]

    U, Sigma, Vt = np.linalg.svd(nonzero_submatrix)
    submatrix_unitary = np.dot(U, Vt)

    smatrix[np.ix_(nonzero_indices, nonzero_indices)] = submatrix_unitary

    return smatrix

def _make_passive_3D(smatrix):
    """
    It makes the scattering matrix passive by forcing L2-norm less than or equal to 1.
    Parameters
        - smatrix: scattering matrix with dimension (n, m, m) where n corresponds to the section number
    Returns
        - smatrix: ndarray with each section ndarray has L2-norm not larger than 1
    """
    smatrix = deepcopy(smatrix)
    norms = np.linalg.norm(smatrix, axis=(1,2), ord=2)
    norms_masked = np.where(norms < 1, 1, norms)
    norms_masked_reshaped = norms_masked[:,np.newaxis, np.newaxis]
    smatrix = smatrix/norms_masked_reshaped

    return smatrix

def _find_nearest_unitary_3D(smatrix):
    """
    It finds nearest unitary matrix to the given smatrix. 
    If smatrix does not includes all mode inforamtion, it returns the matrix with nonzero-submatrix is replaced by the nearest unitary
    
    Parameters:
        -smatrix: scattering matrix with dimension (n, m, m) where n corresponds to the section number
    Returns
        - smatrix: ndarray with each section ndarray becomes unitary nearest to the original matrix
    """
    smatrix = deepcopy(smatrix)
    for i in range(len(smatrix)):
        nonzero_indices = _find_nonzero_index(smatrix[i])
        nonzero_submatrix = smatrix[i][np.ix_(nonzero_indices, nonzero_indices)]

        U, Sigma, Vt = np.linalg.svd(nonzero_submatrix)
        submatrix_unitary = np.dot(U, Vt)

        smatrix[i][np.ix_(nonzero_indices, nonzero_indices)] = submatrix_unitary
    
    return smatrix


#endregion energy related functions


#region ray wrapper function
def ray_wrapper(func, **kwargs):
    @ray.remote
    def wrapped_func(*args, **kwargs2):
        return func(*args, **kwargs, **kwargs2)
    return wrapped_func


@ray.remote
def _find_nonzero_index_ray(matrix, tolerance = 1e-9):
    return _find_nonzero_index(matrix, tolerance)

@ray.remote
def _find_nonzero_indices_ray(smatrix, smatrix_index_list, tolerance = 1e-9):
    nonzero_indices_list = [_find_nonzero_index(smatrix[smatrix_index]) for smatrix_index in smatrix_index_list]
    return nonzero_indices_list

@ray.remote
def np_linalg_svd_ray(matrix):
    return np.linalg.svd(matrix)

@ray.remote
def find_unitary_submatrix_ray(smatrix, index_list):
    # it retruns the list (python list) of submatrix (numpy 2D array)
    unitary_submatrix_list = []
    for index in index_list:
        nonzero_indices = _find_nonzero_index(smatrix[index])
        nonzero_submatrix = smatrix[index][np.ix_(nonzero_indices, nonzero_indices)]

        U, _, Vt = np.linalg.svd(nonzero_submatrix)
        submatrix_unitary = np.dot(U, Vt)
        unitary_submatrix_list.append(submatrix_unitary)
    return unitary_submatrix_list

@ray.remote
def find_unitary_2Dsmatrix_ray(smatrix, index_list):
    # smatrix: 3D array
    # it retruns the list (python list) of submatrix (numpy 2D array)
    unitary_2Dsmatrix_list = []
    for index in index_list:
        unitary_2Dsmatrix_list.append(_find_nearest_unitary_2D(smatrix[index]))

    return unitary_2Dsmatrix_list

@ray.remote
def _inverse_2D_matrices_ray(matrix, index_list, tolerance=1e-9):
    # matrix: 3D array
    inverse_matrix_list = []
    for index in index_list:
        inverse_matrix_list.append(_inverse_2D_matrix(matrix[index], tolerance=tolerance))

    return inverse_matrix_list

@ray.remote
def _inverse_2D_matrices_avoid_nan_ray(matrix, index_list, tolerance = 1e-9, max_trial_num=5):
    # matrix: 3D array
    inverse_matrix_list = []
    for index in index_list:
        inverse_matrix_list.append(_inverse_2D_matrix_avoid_nan(matrix[index], tolerance, max_trial_num, verbose=False))

    return inverse_matrix_list


def _find_nearest_unitary_3D_ray1(smatrix):
    """
    Finds the nearest unitary matrix to the given smatrix using Ray for parallel processing.
    If smatrix does not include all mode information, it replaces the nonzero submatrix with the nearest unitary matrix.

    Parameters:
        smatrix (ndarray): Scattering matrix with dimension (n, m, m), where n corresponds to the section number.

    Returns:
        ndarray: Modified smatrix with each section replaced by the unitary matrix nearest to the original matrix.
    """
    smatrix = deepcopy(smatrix)
    smatrix_len = len(smatrix)

    # Reset the results list for SVD computations
    nonzero_indices_all = ray.get([_find_nonzero_index_ray.remote(smatrix[i]) for i in range(smatrix_len)])
    svd_results = ray.get([np_linalg_svd_ray.remote(smatrix[i][np.ix_(nonzero_indices, nonzero_indices)]) for i, nonzero_indices in enumerate(nonzero_indices_all)])
    # results = []
    for i, svd_result in enumerate(svd_results):
        U, Sigma, Vt = svd_result
        nonzero_indices = nonzero_indices_all[i]
        submatrix_unitary = np.dot(U, Vt)
        smatrix[i][np.ix_(nonzero_indices, nonzero_indices)] = submatrix_unitary

    return smatrix

def _find_nearest_unitary_3D_ray2(smatrix, task_block_size = 30):
    smatrix2D_id_list = []  # list of list of numpy array 
    # [[np.array(), np.array(), ... , np.array()], ... , [np.array(), np.array(), ... , np.array()]]
    smatrix = deepcopy(smatrix)
    smatrix_len = len(smatrix)
    smatrix_ref = ray.put(smatrix)

    index_list = np.linspace(0, smatrix_len-1, smatrix_len, dtype=int)
    start_index = 0
    stop_index = task_block_size

    while stop_index < smatrix_len:
        smatrix2D_id_list.append(find_unitary_2Dsmatrix_ray.remote(smatrix_ref, index_list[start_index:stop_index]))
        start_index += task_block_size
        stop_index += task_block_size
    stop_index = None
    smatrix2D_id_list.append(find_unitary_2Dsmatrix_ray.remote(smatrix_ref, index_list[start_index:stop_index]))

    smatrix2D_list = ray.get(smatrix2D_id_list)
    # [[np.array(), np.array(), ... , np.array()], ... , [np.array(), np.array(), ... , np.array()]]
    smatrix2D_list_flattend = []
    for i in range(len(smatrix2D_list)):
        smatrix2D_list_flattend += smatrix2D_list[i]
    # [np.array(), np.array(), ... , np.array(), ... , np.array(), np.array(), ... , np.array()]
    unitary_smatrix = np.array(smatrix2D_list_flattend)

    return unitary_smatrix
    

def _inverse_3D_matrix_ray(matrix, tolerance = 1e-9, task_block_size = 30, verbose=True):
    """
    Parameter:
        - matrix: ndarray of square matrix with shape (m,n,n)
    Returns:
        - matrix whose nonzero submatrix is inversed
    """
    # matrix shape validity test
    matrix = deepcopy(matrix)
    if not len(matrix.shape) == 3:
        print("matrix should be shape of (m,n,n)")
        return
    if not matrix.shape[1] == matrix.shape[2]:
        print("matrix should be shape of (m,n,n)")
        return
    
    matrix_len = len(matrix)
    matrix_ref = ray.put(matrix)

    inversed_matrix_id_list = []

    index_list = np.linspace(0, matrix_len-1, matrix_len, dtype=int)
    start_index = 0
    stop_index = task_block_size

    while stop_index < matrix_len:
        inversed_matrix_id_list.append(_inverse_2D_matrices_ray.remote(matrix_ref, index_list[start_index:stop_index], tolerance=tolerance))
        start_index += task_block_size
        stop_index += task_block_size
    stop_index = None
    inversed_matrix_id_list.append(_inverse_2D_matrices_ray.remote(matrix_ref, index_list[start_index:stop_index], tolerance=tolerance))

    inversed_matrix_list = ray.get(inversed_matrix_id_list)
    # [[np.array(), np.array(), ... , np.array()], ... , [np.array(), np.array(), ... , np.array()]]
    inversed_matrix_list_flattend = []
    for i in range(len(inversed_matrix_list)):
        inversed_matrix_list_flattend += inversed_matrix_list[i]
    # [np.array(), np.array(), ... , np.array(), ... , np.array(), np.array(), ... , np.array()]
    inverse_matrix = np.array(inversed_matrix_list_flattend)

    if np.isnan(inverse_matrix).any(): # check matrix 
        inversed_matrix_id_list = []

        index_list = np.linspace(0, matrix_len-1, matrix_len, dtype=int)
        start_index = 0
        stop_index = task_block_size

        while stop_index < matrix_len:
            inversed_matrix_id_list.append(_inverse_2D_matrices_avoid_nan_ray.remote(matrix_ref, index_list[start_index:stop_index], tolerance=tolerance))
            start_index += task_block_size
            stop_index += task_block_size
        stop_index = None
        inversed_matrix_id_list.append(_inverse_2D_matrices_avoid_nan_ray.remote(matrix_ref, index_list[start_index:stop_index], tolerance=tolerance))

        inversed_matrix_list = ray.get(inversed_matrix_id_list)
        # [[np.array(), np.array(), ... , np.array()], ... , [np.array(), np.array(), ... , np.array()]]
        inversed_matrix_list_flattend = []
        for i in range(len(inversed_matrix_list)):
            inversed_matrix_list_flattend += inversed_matrix_list[i]
        # [np.array(), np.array(), ... , np.array(), ... , np.array(), np.array(), ... , np.array()]
        inverse_matrix = np.array(inversed_matrix_list_flattend)
        if np.isnan(inverse_matrix).any() and verbose: print("There is NaN in an inversed matrix")
    
    return inverse_matrix


#endregion ray wrapper function
