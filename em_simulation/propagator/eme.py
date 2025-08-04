from ..geometry.geometry import Geometry
from ..propagator.propagator import Propagator
from ..propagator.multi_propagator.multi_eme import MultiEME
from ..propagator.single_propagator.single_eme import SingleEME

class EME(Propagator):
    def __init__(self, geometry:Geometry, force_passive:bool = False, force_unitary:bool = False):
        """
        geometry: geometry instance
        force_passive: bool
        calculation_method:
            0: use broadcasting. Use this method when RAM is enough (more than 32GB)
            1: use for-loop when calculating coupling coefficient. Use this method when RAM is not enough
        """
        self._is_composite_geometry = geometry._is_composite_geometry
        if self._is_composite_geometry:
            self.propagator = MultiEME(geometry, force_passive, force_unitary)
            self._is_multipropagator = 1
        else:
            self.propagator = SingleEME(geometry, force_passive, force_unitary)
            self._is_multipropagator = 0
    
    def calc_Tmatrix(self):
        self.propagator.calc_Tmatrix()

    def calc_Smatrix(self):
        self.propagator.calc_Smatrix()

    def change_strucutre_length(self, new_length):
        self.propagator.change_strucutre_length(new_length)

