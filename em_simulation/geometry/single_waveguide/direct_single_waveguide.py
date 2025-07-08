import abc

import numpy as np
from matplotlib import pyplot as plt

from ..direct_geometry import DirectGeometry


class DirectSingleWaveguide(DirectGeometry, metaclass=abc.ABCMeta):
    """
    Structure geometry for curves
    Parameters of dataset:
        Topwidth
        Curvature: 0~ infinity
        Rotation angle: -np.pi/2 ~ np.pi/2
    note: rotation angle is opposite of the propagation angle because the rotation angle means material rotation.
    """
    def __init__(self, dataset, limit_mode_number = 0, use_existing_data= False):
        super().__init__(dataset, limit_mode_number= limit_mode_number, use_existing_data= use_existing_data)
        self._width_function = None
        self._curvature_function = None
        self._prop_angle_function = None
        self._total_length = 0

    def calc_simulation_parameters(self):
        top_width_function, curvature_function, prop_angle_function = self._parametrized_function()
        self._width_function = top_width_function
        self._curvature_function = curvature_function
        self._prop_angle_function = prop_angle_function

        resolution = self.resolution    # resolution is defined in each structure shape
        z_values = np.linspace(0, self._total_length, resolution)

        # parameter values at each propagation length
        top_widths = list()
        curvatures = list()
        for z in z_values:
            top_widths.append(top_width_function(z))
            curvatures.append(curvature_function(z))
        top_widths = np.array(top_widths)
        curvatures = np.array(curvatures)
        if 'rotation_angle' in self.parameter_names:
            prop_angles = list()
            for z in z_values:
                prop_angles.append(prop_angle_function(z))
            prop_angles = np.array(prop_angles)
        
        # find the parameter order
        top_width_index = self.parameter_names.index("top_width")
        curvature_index = self.parameter_names.index("curvature")
        if 'rotation_angle' in self.parameter_names:
            rotation_index = self.parameter_names.index("rotation_angle")
        
        #generate list of tuples
        if 'rotation_angle' in self.parameter_names:
            values = np.zeros(shape = (3, resolution), dtype = float)
            values[top_width_index] = top_widths
            # Lumerical FDE only calculate positive curvature, symmetric opertaion on rotation angle for negative curvature
            rotation_angles = -prop_angles
            values[curvature_index] = np.where(curvatures>=0, curvatures, -curvatures)
            values[rotation_index] = np.where(curvatures>=0, rotation_angles, -rotation_angles)

        else:
            values = np.zeros(shape = (2, resolution), dtype = float)
            values[top_width_index] = top_widths
            # Lumerical FDE only calculate positive curvature
            values[curvature_index] = np.where(curvatures>=0, curvatures, -curvatures)
        
        delta_zs = z_values[1:] - z_values[:-1]
        tuples = list(zip(*values))

        return tuples, delta_zs
    
    @abc.abstractmethod
    def calc_total_length(self):
        pass

    #region plot
    def plot_structure_parameters(self, resolution = 500):
        simul_params, delta_zs = self.calc_simulation_parameters()
        prop_lengths = np.insert(np.cumsum(delta_zs), 0, [0])
        prop_lengths_um = prop_lengths * 1e6
        prop_lengths_idal = np.linspace(0, prop_lengths[-1], resolution)

        # find the parameter order
        top_width_index = self.parameter_names.index("top_width")
        curvature_index = self.parameter_names.index("curvature")
        if 'rotation_angle' in self.parameter_names:
            rotation_index = self.parameter_names.index("rotation_angle")

        # plot top_width (ideal & simul point)
        top_width_points = []
        for i in range(len(simul_params)):
            top_width_points.append(simul_params[i][top_width_index])
        top_width_points = np.array(top_width_points)*1e6
        plt.scatter(prop_lengths_um, top_width_points, label="simulation points")
        plt.plot(prop_lengths_idal*1e6, self._width_function(prop_lengths_idal)*1e6, label="ideal parameters")
        plt.title("Top Width")
        plt.legend()
        plt.xlabel("Propagation Length (um)")
        plt.ylabel("um")
        plt.show()

        # plot Curvature
        curvature_points = []
        for i in range(len(simul_params)):
            curvature_points.append(simul_params[i][curvature_index])
        plt.scatter(prop_lengths_um, curvature_points, label="simulation points")
        # curvature is alway become positive. (If negative curvature exists, it uses symmetry to make it positive)
        plt.plot(prop_lengths_idal*1e6, np.abs(self._curvature_function(prop_lengths_idal)), label="ideal parameters")
        plt.title("Curavature")
        plt.legend()
        plt.xlabel("Propagation Length (um)")
        plt.ylabel("m^(-1)")
        plt.show()

        # plot propagation angle
        if 'rotation_angle' in self.parameter_names:
            rotation_angle_points = []
            for i in range(len(simul_params)):
                rotation_angle_points.append(simul_params[i][rotation_index])
            prop_angle_points = -np.array(rotation_angle_points)

            # undo symmetric operation
            curvatures = self._curvature_function(prop_lengths)
            prop_angle_points = np.where(curvatures >= 0, prop_angle_points, -prop_angle_points)
            # plot
            plt.scatter(prop_lengths_um, prop_angle_points, label="simulation points")
            plt.plot(prop_lengths_idal*1e6, self._prop_angle_function(prop_lengths_idal), label="ideal parameters")
            plt.title("Propagation Angle")
            plt.legend()
            plt.xlabel("Propagation Length (um)")
            plt.ylabel("Rad")
            plt.show()