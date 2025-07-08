import numpy as np
from scipy.interpolate import interp1d
# from scipy.integrate import trapz, cumtrapz
from scipy.integrate import trapezoid as trapz
from scipy.integrate import cumulative_trapezoid as cumtrapz

from ....data_updater.data_updater import DataUpdater
from ..single_waveguide import SingleWaveguide


class SingleCustomBend(SingleWaveguide):
    def __init__(self, dataset: DataUpdater, prop_len_list, width_list, curvature_list, input_angle = 0, resolution = 3000, limit_mode_number = 0, verbose = True):
        if len(prop_len_list) != len(width_list) or len(width_list) != len(curvature_list):
            print("prop_len_list, width_list, and curvature_list should have same length")
        SingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number, resolution=resolution, verbose=verbose)
        ds = np.diff(prop_len_list, prepend=0)
        prop_angle = input_angle + np.array(self._find_angle_list(prop_len_list, curvature_list))
        self._input_angle = input_angle
        self._prop_length, self._curvature, self._prop_angle, self._ds = prop_len_list, curvature_list, prop_angle, ds
        self._top_widths = width_list
        self._total_length = self.calc_total_length()

    def _parametrized_function(self):
        curvature_function = interp1d(self._prop_length, self._curvature, kind='linear')
        prop_angle_function = interp1d(self._prop_length, self._prop_angle, kind='linear')
        top_width_function = interp1d(self._prop_length, self._top_widths, kind='linear')

        return top_width_function, curvature_function, prop_angle_function
    
    def _find_angle_list(self, prop_len_list, curv_list):
        # angle_list in radian
        angle_list = cumtrapz(curv_list, prop_len_list, initial=0)
        return angle_list
    
    def _calc_xy(self):
        num_points = len(self._prop_length)
        prop_angle = self._find_angle_list(self._prop_length, self._curvature)  # this angle ignores the input angle
        delta_zs = np.diff(self._prop_length)
        x, y = np.zeros(num_points), np.zeros(num_points)

        for i in range(num_points-1):
            x[i+1] = x[i] + delta_zs[i] * np.cos(prop_angle[i])
            y[i+1] = y[i] + delta_zs[i] * np.sin(prop_angle[i])

        x, y = self._rotate_xy(x, y, self._input_angle)

        return x, y
    
    def calc_total_length(self):
        return self._prop_length[-1]

    @staticmethod
    def _rotate_xy(x, y, input_angle):
        """
        x, y: list or np array with same length
        input_angle: angle in radian
        """
        x, y = np.array(x), np.array(y)# make sure that the lists are np array to use broadcasting
        new_x = np.cos(input_angle) * x - np.sin(input_angle) * y
        new_y = np.sin(input_angle) * x + np.cos(input_angle) * y
        return new_x, new_y

    
    #region plot
    def plot_2D_structure(self):
        return SingleWaveguide.plot_2D_structure(self)
    
    def plot_structure_parameters(self):
        return SingleWaveguide.plot_structure_parameters(self)