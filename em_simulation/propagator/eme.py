from ..geometry.geometry import Geometry
from ..propagator.propagator import Propagator
from ..propagator.multi_propagator.multi_eme import MultiEME
from ..propagator.single_propagator.single_eme import SingleEME

class EME(Propagator):
    def __init__(self, geometry:Geometry, force_passive:bool = False, force_unitary:bool = False):
        """Instantiate an eigenmode expansion propagator.

        :param geometry: Geometry description from which modal data is drawn.
        :type geometry: Geometry
        :param force_passive: Enforce passivity by zeroing gain in the transfer
            matrices.
        :type force_passive: bool
        :param force_unitary: Enforce unitarity when building scattering
            matrices.
        :type force_unitary: bool
        """
        self._is_composite_geometry = geometry._is_composite_geometry
        if self._is_composite_geometry:
            self.propagator = MultiEME(geometry, force_passive, force_unitary)
            self._is_multipropagator = 1
        else:
            self.propagator = SingleEME(geometry, force_passive, force_unitary)
            self._is_multipropagator = 0
    
    def calc_Tmatrix(self):
        """Compute and cache the overall transfer matrix."""
        self.propagator.calc_Tmatrix()

    def calc_Smatrix(self):
        """Compute and cache the overall scattering matrix."""
        self.propagator.calc_Smatrix()

    def change_strucutre_length(self, new_length):
        self.propagator.change_strucutre_length(new_length)

