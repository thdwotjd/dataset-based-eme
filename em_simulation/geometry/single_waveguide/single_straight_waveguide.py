import numpy as np

from ...data_updater.data_updater import DataUpdater
from ..geometry import Geometry

class SingleStraightWaveguide(Geometry):
    def __init__(self, dataset:DataUpdater, top_width, length, prop_angle= 0, limit_mode_number = 0):
        super().__init__(dataset, limit_mode_number= limit_mode_number)
        self._structure_geometry = "Single Waveguide"
        self._total_length = length
        self._propagation_angle = prop_angle
        self._top_width = top_width
        pass

    def calc_simulation_parameters(self):
        # find the parameter order
        top_width_index = self.parameter_names.index("top_width")
        curvature_index = self.parameter_names.index("curvature")
        if 'rotation_angle' in self.parameter_names:
            rotation_index = self.parameter_names.index("rotation_angle")

        #generate list of tuples
        if 'rotation_angle' in self.parameter_names:
            values = np.zeros(shape = (3, 2), dtype = self._top_width.dtype)
            values[top_width_index] = np.array([self._top_width, self._top_width])
            values[curvature_index] = np.array([0,0])
            values[rotation_index] = -np.array([self._propagation_angle, self._propagation_angle])
        else:
            values = np.zeros(shape = (2, 2), dtype = self._top_width.dtype)
            values[top_width_index] = np.array([self._top_width, self._top_width])
            values[curvature_index] = np.array([0,0])
        
        tuples = list(zip(*values))
        delta_zs = [self._total_length]

        return tuples, delta_zs