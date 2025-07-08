import abc
import numpy as np
from copy import deepcopy

from ..geometry.geometry import Geometry
from ..geometry.direct_geometry import DirectGeometry



class Propagator(metaclass=abc.ABCMeta):
    def __init__(self, geometry:Geometry, force_passive = False, force_unitary = False):
        """
        geometry: instance of Geometry class
        """
        if not (issubclass(type(geometry), Geometry) or issubclass(type(geometry), DirectGeometry)):
            print("geometry should be instance of the subclass of the Geometry")
            return 0
        
        if geometry.output_data == None:
            geometry.calc_output_data()
        
        self.output_data = geometry.output_data
        self.section_count, mode_count = self.output_data["neff"].shape
        self.mode_count = int(mode_count/2)
        self.tmatrix = None
        self.smatrix = None
        self._lengths_per_matrix = None # calculated in calc_Tmatrix()
        self._force_passive = force_passive
        self._force_unitary = force_unitary

        # status
        self._is_tmatrix_calculated = False
        self._is_smatrix_calculated = False

    @abc.abstractmethod
    def calc_Tmatrix(self):
        pass
    
    @abc.abstractmethod
    def calc_Smatrix(self):
        pass

    def change_strucutre_length(self, new_length):
        """
        This is function is only valid for straight structure (Not valid for curve structure)
        """
        pass

    def _find_Smatrix_new_length(self, new_length):
        """
        return new length Smatrix (n,m,m)
        """
        pass

    #region extract block submatrix

    def _extract_blockmatrix_3D(self, matrix):
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
        m11 = new_matrix[:,:self.mode_count, :self.mode_count]
        m12 = new_matrix[:,:self.mode_count, self.mode_count:]
        m21 = new_matrix[:, self.mode_count:, :self.mode_count]
        m22 = new_matrix[:, self.mode_count:, self.mode_count:]
        return m11, m12, m21, m22
    
    def _extract_blockmatrix_2D(self, matrix):
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