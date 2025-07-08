from ..geometry.geometry import Geometry
from ..propagator.propagator import Propagator
from ..propagator.multi_propagator.multi_eme import MultiEME
from ..propagator.single_propagator.single_eme import SingleEME

class EME(Propagator):
    def __init__(self, geometry:Geometry, force_passive = False, force_unitary = False, is_test_mode = False):
        """
        calculation_method:
            0: use broadcasting. Use this method when RAM is enough (more than 32GB)
            1: use for-loop when calculating coupling coefficient. Use this method when RAM is not enough
        """
        self._is_composite_geometry = geometry._is_composite_geometry
        if self._is_composite_geometry:
            self.propagator = MultiEME(geometry, force_passive, force_unitary, is_test_mode)
            self._is_multipropagator = 1
        else:
            self.propagator = SingleEME(geometry, force_passive, force_unitary, is_test_mode)
            self._is_multipropagator = 0
    
    def calc_Tmatrix(self):
        self.propagator.calc_Tmatrix()

    def calc_Smatrix(self):
        self.propagator.calc_Smatrix()

    def change_strucutre_length(self, new_length):
        self.propagator.change_strucutre_length(new_length)

