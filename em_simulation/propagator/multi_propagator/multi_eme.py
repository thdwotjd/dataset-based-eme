from ...geometry.composite_geometry import CompositeGeometry
from .multi_propagator import MultiPropagator

class MultiEME(MultiPropagator):
    def __init__(self, composite_geometry:CompositeGeometry, force_passive = False,force_unitary = False, is_test_mode = False):
        super().__init__(composite_geometry, force_passive=force_passive, force_unitar=force_unitary)
        
        # initialize EMEs
        geometries = self._geometries # multipropagator variable
        propagators = []
        from ..single_propagator.single_eme import SingleEME
        for geometry in geometries:
            eme = SingleEME(geometry, force_passive, force_unitary, is_test_mode)
            propagators.append(eme)
        # multieme parameters
        self.propagators = propagators
 
    
    def calc_Tmatrix(self):
        self.tmatrix = []
        for eme in self.propagators:
            eme.calc_Tmatrix()
            self.tmatrix.append(eme.tmatrix)
        self._is_tmatrix_calculated = 1
    
    def calc_Smatrix(self):
        smatrix = []
        for eme in self.propagators:
            eme.calc_Smatrix()
            smatrix.append(eme.smatrix)
        self.smatrix = smatrix
        self.calc_geometry_intersection_smatrix()
        self._merge_result_data()
        self._is_smatrix_calculated = 1
    