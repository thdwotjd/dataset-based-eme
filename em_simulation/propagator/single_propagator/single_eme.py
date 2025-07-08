import numpy as np
from copy import deepcopy

from ... import matrix_calculation_tool as mct
from ...geometry.geometry import Geometry
from ..propagator import Propagator


class SingleEME(Propagator):
    """
    The simulation algorithm basically follows the thesis "P. Bienstman, “Rigorous and efficient modelling of wavelenght scale photonic components / Peter Bienstman.,” 2001."
    To see the detail of the mathematical backgorund and terms, see chater2 of the thesis. 
    """
    def __init__(self, geometry:Geometry, force_passive = False, force_unitary = False, is_test_mode = False):
        super().__init__(geometry, force_passive=force_passive, force_unitary=force_unitary)
        # path = self.output_data["path"]
        # delta_zs = self.output_data["delta_zs"]
        # eme_path, delta_zs = self.interp_multi_adj_pts(path, delta_zs)
        # self.overlap_forward_ab, self.overlap_forward_ba =\   # overlap, neff, TE_pol are all ndarray not dict
        #       self.overlap_dict2ndarray(eme_path, self.output_data["overlap_ab"], self.output_data["overlap_ba"] )
        # self.beta_forward = self.dict2ndarray

        # additional_param_dict = self.output_data["additional_param_dict"]

        self.overlap_forward_ab = self.output_data["overlap_ab"][:,:self.mode_count, :self.mode_count]
        self.overlap_forward_ba = self.output_data["overlap_ba"][:,:self.mode_count, :self.mode_count]
        self.beta_forward = self.output_data["beta"][:,:self.mode_count]
        if is_test_mode:
            self.beta_forward = self.output_data["beta"][:,:self.mode_count].real

        self._interface_Tmatrix = None  # ndarray with shape (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count)

        # status
        self._is_interface_Tmatrix_calcualted = False


    #region main functions
    def calc_Smatrix(self):
        if not self._is_tmatrix_calculated:
            self.calc_Tmatrix()
        # smatrix = mct._convert_3Dmatrix(self.tmatrix)
        smatrix = mct._convert_3Dmatrix_ray(self.tmatrix)

        if self._force_unitary:
            # smatrix = mct._find_nearest_unitary_3D(smatrix)
            smatrix = mct._find_nearest_unitary_3D_ray2(smatrix)
        if self._force_passive:
            smatrix = mct._make_passive_3D(smatrix)
        self.smatrix = smatrix
        self._is_smatrix_calculated = True

    def calc_Tmatrix(self):
        interfaces = self._calc_interface_Tmatrix()
        phase_propagations = self._calc_phase_propagation_Tmatrix()
        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])

        total_matrices = np.zeros((2*self.section_count - 2, 2*self.mode_count, 2*self.mode_count), dtype = np.complex64)
        lengths_per_matrix = np.zeros(2*self.section_count-2, dtype = float)
        for i in range(self.section_count - 1):
            total_matrices[2*i] = phase_propagations[i]
            total_matrices[2*i + 1] = interfaces[i]
            lengths_per_matrix[2*i] = delta_zs[i]

        self.tmatrix = deepcopy(total_matrices)
        self._lengths_per_matrix = lengths_per_matrix
        self._is_tmatrix_calculated = True
        return total_matrices
    
    def change_strucutre_length(self, new_length):
        self.tmatrix = self._find_Tmatrix_new_length(new_length)
        self.smatrix = self._find_Smatrix_new_length(new_length)

        lengths_per_matrix = np.zeros(2*self.section_count-2, dtype = float)

        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])
        initial_length = np.sum(delta_zs)
        length_ratio = new_length/initial_length
        for i in range(self.section_count - 1):
            lengths_per_matrix[2*i] = delta_zs[i]

        self._lengths_per_matrix = lengths_per_matrix*length_ratio


        print("Total Length is changed to ", str(new_length * 1e6), "um")
    
    #endregion main functions



    #region functions used in calc_Tmatrix
    def _calc_interface_Tmatrix(self):
        T12 = self._calc_transmission_matrix(self.overlap_forward_ab, self.overlap_forward_ba)
        T21 = self._calc_transmission_matrix(self.overlap_forward_ba, self.overlap_forward_ab)
        R12 = self._calc_reflection_matrix(self.overlap_forward_ab, self.overlap_forward_ba, T12)
        R21 = self._calc_reflection_matrix(self.overlap_forward_ba, self.overlap_forward_ab, T21)

        inverse_T21 = mct._inverse_3D_matrix_ray(T21)
        m11 = T12 - R21 @ inverse_T21 @ R12
        m12 = R21 @ inverse_T21
        m21 = (-1) * inverse_T21 @ R12
        m22 = inverse_T21

        interface_Tmatrix = np.zeros(shape = (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count), dtype = complex)
        interface_Tmatrix[:,:self.mode_count, :self.mode_count] = m11
        interface_Tmatrix[:,:self.mode_count, self.mode_count:] = m12
        interface_Tmatrix[:,self.mode_count:, :self.mode_count] = m21
        interface_Tmatrix[:,self.mode_count:, self.mode_count:] = m22

        self._interface_Tmatrix = interface_Tmatrix
        self._is_interface_Tmatrix_calcualted = True

        return interface_Tmatrix
    
    def _calc_phase_propagation_Tmatrix(self):
        diagonal_mask = np.eye(self.mode_count, dtype = np.complex64)
        i, j, _ = np.meshgrid(np.arange(0, self.section_count-1),\
                              np.arange(0, self.mode_count),\
                              np.arange(0, self.mode_count),\
                              indexing = 'ij')

        forward_matrix = np.exp(1j*self.beta_forward[i, j]*self.output_data["EME_delta_zs"][i]) * diagonal_mask
        backward_matrix = np.exp((-1j)*self.beta_forward[i, j]*self.output_data["EME_delta_zs"][i]) * diagonal_mask

        result = np.zeros(shape = (self.section_count-1, 2*self.mode_count, 2*self.mode_count), dtype = complex)
        result[:,:self.mode_count, :self.mode_count] = forward_matrix
        result[:,self.mode_count:, self.mode_count:] = backward_matrix

        return result
    
    def _calc_transmission_matrix(self, overlap_ab, overlap_ba):
        # if the result is wrong, try reorder overlap (check old version simulator)
        matrix_temp = overlap_ab + np.transpose(overlap_ba, (0,2,1))
        overlap_tolerance = 0.5 # this value should be adjusted if it does not works.
        result = 2 * mct._inverse_3D_matrix_ray(matrix_temp, tolerance = overlap_tolerance)

        return result
    
    def _calc_reflection_matrix(self, overlap_ab, overlap_ba, transmission_matrix):
        result = 0.5 * (np.transpose(overlap_ab, (0,2,1)) - overlap_ba) @ transmission_matrix
        return result
    
    #endregion functions used in calc_Tmatrix

    #region functions for change length 
    def _find_Smatrix_new_length(self, new_length):
        tmatrix = self._find_Tmatrix_new_length(new_length)
        smatrix = mct._convert_3Dmatrix(tmatrix)

        if self._force_unitary:
            smatrix = mct._find_nearest_unitary_3D(smatrix)
        if self._force_passive:
            smatrix = mct._make_passive_3D(smatrix)

        return smatrix

    def _find_Tmatrix_new_length(self, new_length):
        if not self._is_interface_Tmatrix_calcualted:
            self._calc_interface_Tmatrix()
        
        interfaces = deepcopy(self._interface_Tmatrix)
        phase_propagations = self._calc_phase_propagation_Tmatrix_new_length(new_length)

        total_matrices = np.zeros((2*self.section_count - 2, 2*self.mode_count, 2*self.mode_count), dtype = np.complex64)
        for i in range(self.section_count - 1):
            total_matrices[2*i] = phase_propagations[i]
            total_matrices[2*i + 1] = interfaces[i]
        
        return total_matrices

    def _calc_phase_propagation_Tmatrix_new_length(self, new_length):
        initial_length = np.sum(deepcopy(self.output_data["EME_delta_zs"]))
        length_ratio = new_length/initial_length

        diagonal_mask = np.eye(self.mode_count, dtype = complex)
        i, j, _ = np.meshgrid(np.arange(0, self.section_count-1),\
                              np.arange(0, self.mode_count),\
                              np.arange(0, self.mode_count),\
                              indexing = 'ij')

        forward_matrix = np.exp(1j*self.output_data["beta"][i, j]*length_ratio*self.output_data["EME_delta_zs"][i]) * diagonal_mask
        backward_matrix = np.exp((-1j)*self.output_data["beta"][i, j]*length_ratio*self.output_data["EME_delta_zs"][i]) * diagonal_mask

        result = np.zeros(shape = (self.section_count-1, 2*self.mode_count, 2*self.mode_count), dtype = complex)
        result[:,:self.mode_count, :self.mode_count] = forward_matrix
        result[:,self.mode_count:, self.mode_count:] = backward_matrix

        return result
    
    #endregion functions for change length