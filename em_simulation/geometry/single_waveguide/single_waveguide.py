import abc
import numpy as np
from scipy.optimize import fsolve
from copy import deepcopy

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from ..geometry import Geometry

class SingleWaveguide(Geometry, metaclass=abc.ABCMeta):
    """
    Structure geometry for curves
    Parameters of dataset:
        Topwidth
        Curvature: 0~ infinity
        Rotation angle: -np.pi/2 ~ np.pi/2
    note: rotation angle is opposite of the propagation angle because the rotation angle means material rotation.
    """
    def __init__(self, dataset, limit_mode_number = 0, resolution = 1000, verbose = True):
        super().__init__(dataset, limit_mode_number= limit_mode_number, verbose= verbose)
        self._width_function = None
        self._curvature_function = None
        self._prop_angle_function = None
        self._resolution = resolution
        # self._total_length = 0
        
    
    
    def calc_simulation_parameters(self):
        """
        Calculates the list of simulation parameter points along a geometric trajectory
        and the physical distances between them.

        This method is intended to be used in a class hierarchy. It relies on the
        `_parametrized_function()` method, which must be implemented by a subclass.
        The child class defines how key geometric parameters (e.g., top width,
        curvature, propagation angle) vary along the propagation axis.

        The method evaluates the parameter functions over the total propagation
        length, discretizes them according to a predefined grid (as specified in
        `dataset_info.py`), and generates a list of parameter-space tuples
        corresponding to the dataset grid. These tuples represent simulation points.
        Additionally, the physical spacing (in meters) between adjacent simulation
        points is computed.

        Returns:
            simulation_parameters (List[Tuple[float]]): A list of parameter tuples
                corresponding to the dataset grid. Each tuple defines a point in the
                parameter space (e.g., top_width, curvature, rotation_angle).
            delta_zs (np.ndarray): An array of physical distances (in meters)
                between adjacent simulation parameter points.
        """
        top_width_function, curvature_function, prop_angle_function = self._parametrized_function()
        self._width_function = top_width_function
        self._curvature_function = curvature_function
        self._prop_angle_function = prop_angle_function

        # Set the range of z values
        z_values = np.linspace(0, self._total_length, self._resolution)

        # set the reference values including symmetric points        
        top_width_reference = deepcopy(self.parameter_grid['top_width'])
        curvature_reference = deepcopy(self.parameter_grid['curvature'])
        ## Lumerical FDE only calculate positive curvature, but structure parameter includes negative curvature
        minus_curvature_ref = -curvature_reference
        expanded_curvature_ref = np.unique(np.concatenate([minus_curvature_ref, curvature_reference]))
        if 'rotation_angle' in self.parameter_names:
            rotation_reference = deepcopy(self.parameter_grid['rotation_angle'])

        # Scan each function and save the values
        propagation_curvature, curvature_values = self.scan_and_store(z_values, curvature_function, expanded_curvature_ref)
        propagation_top_width, top_width_values = self.scan_and_store(z_values, top_width_function, top_width_reference)
        if 'rotation_angle' in self.parameter_names:
            propagation_prop_angle, prop_angle_values = self.scan_and_store(z_values, prop_angle_function, -rotation_reference)
        
        # find nearest gird point at each propagation_tot point (new)
        if 'rotation_angle' in self.parameter_names:
            propagation_tot = np.unique(np.concatenate((propagation_curvature, propagation_top_width, propagation_prop_angle)))
        else:
            propagation_tot = np.unique(np.concatenate((propagation_curvature, propagation_top_width)))
        top_width_values_tot = np.zeros(shape = (len(propagation_tot)), dtype = float)
        curvature_values_tot = np.zeros(shape = (len(propagation_tot)), dtype = float)
        if 'rotation_angle' in self.parameter_names:
            prop_angle_values_tot = np.zeros(shape = (len(propagation_tot)), dtype = float)
        for i, length in enumerate(propagation_tot):
            top_width_values_tot[i] = self.find_nearest_value(top_width_function(length), top_width_reference)
            curvature_values_tot[i] = self.find_nearest_value(curvature_function(length), expanded_curvature_ref)
            if 'rotation_angle' in self.parameter_names:
                prop_angle_values_tot[i] = self.find_nearest_value(prop_angle_function(length), -rotation_reference)
        
        # remove duplicate points
        if 'rotation_angle' in self.parameter_names:
            propagation_tot, [prop_angle_values_tot, top_width_values_tot, curvature_values_tot] =\
            self.remove_duplicate_points(propagation_tot, prop_angle_values_tot, top_width_values_tot, curvature_values_tot)
        else:
            propagation_tot, [top_width_values_tot, curvature_values_tot] =\
            self.remove_duplicate_points(propagation_tot, top_width_values_tot, curvature_values_tot)

        # if length between different parameter point is too large, it allows duplicate
        max_interpoint_len = 10e-6
        i = 0
        if 'rotation_angle' in self.parameter_names:
            while i < (len(propagation_tot)-1):
                if propagation_tot[i+1] - propagation_tot[i] > max_interpoint_len:
                    propagation_tot = np.insert(propagation_tot, i+1, propagation_tot[i] + max_interpoint_len * 2/3)
                    prop_angle_values_tot = np.insert(prop_angle_values_tot, i+1, prop_angle_values_tot[i])
                    top_width_values_tot = np.insert(top_width_values_tot, i+1, top_width_values_tot[i])
                    curvature_values_tot = np.insert(curvature_values_tot, i+1, curvature_values_tot[i])
                if i > 3:
                    max_interpoint_len = (max_interpoint_len + np.average(np.diff(propagation_tot[:i+1])))*3/5*1.01
                i += 1
        else:
            while i < (len(propagation_tot)-1):
                if propagation_tot[i+1] - propagation_tot[i] > max_interpoint_len:
                    propagation_tot = np.insert(propagation_tot, i+1, propagation_tot[i]+max_interpoint_len * 2/3)
                    top_width_values_tot = np.insert(top_width_values_tot, i+1, top_width_values_tot[i])
                    curvature_values_tot = np.insert(curvature_values_tot, i+1, curvature_values_tot[i])
                if i > 3:
                    max_interpoint_len = (max_interpoint_len + np.average(np.diff(propagation_tot[:i+1])))*3/5*1.01
                i += 1

        # find the parameter order
        top_width_index = self.parameter_names.index("top_width")
        curvature_index = self.parameter_names.index("curvature")
        if 'rotation_angle' in self.parameter_names:
            rotation_index = self.parameter_names.index("rotation_angle")

        #generate list of tuples
        num_points = len(top_width_values_tot)
        if 'rotation_angle' in self.parameter_names:
            values = np.zeros(shape = (3, num_points), dtype = top_width_values.dtype)
            values[top_width_index] = top_width_values_tot
            # Lumerical FDE only calculate positive curvature, symmetric opertaion on rotation angle for negative curvature
            rotation_angle_values_tot = -prop_angle_values_tot
            values[curvature_index] = np.where(curvature_values_tot>=0, curvature_values_tot, -curvature_values_tot)
            values[rotation_index] = np.where(curvature_values_tot>=0, rotation_angle_values_tot, -rotation_angle_values_tot)

        else:
            values = np.zeros(shape = (2, num_points), dtype = top_width_values.dtype)
            values[top_width_index] = top_width_values_tot
            # Lumerical FDE only calculate positive curvature
            values[curvature_index] = np.where(curvature_values_tot>=0, curvature_values_tot, -curvature_values_tot)
        
        delta_zs = propagation_tot[1:] - propagation_tot[:-1]
        tuples = list(zip(*values))

        if len(tuples) == 1:
            tuples.append(tuples[0])
            delta_zs = np.array([self._total_length])

        return tuples, delta_zs
    
    @abc.abstractmethod
    def calc_total_length(self):
        pass

    @abc.abstractmethod
    def _calc_xy(self):
        pass


    def find_nearest_value(self, value, reference_values):
        return super().find_nearest_value(value, reference_values)
    
    #region plot
    def plot_structure_parameters(self, resolution = 500):
        return_values = dict()
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
        return_values["prop_lengths"] = deepcopy(prop_lengths_idal)
        return_values["widths"] = self._width_function(prop_lengths_idal)
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
        return_values["curvatures"] = deepcopy(np.abs(self._curvature_function(prop_lengths_idal)))
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
            return_values["propagation angles"] = self._prop_angle_function(prop_lengths_idal)
            plt.title("Propagation Angle")
            plt.legend()
            plt.xlabel("Propagation Length (um)")
            plt.ylabel("Rad")
            plt.show()
        
        return return_values
    
    def plot_2D_structure(self):
        x, y = self._calc_xy()
        prop_lengths = self._compute_length(x, y)

        correction_factor = self._total_length/(prop_lengths[-1]*1.001)
        prop_lengths *= correction_factor   # sometimes prop_lengths exceed the total length by numerical error

        widths = self._width_function(prop_lengths)
        bdr_x, bdr_y = self.line_to_bdry_pts(x, y, widths)

        fig, ax = plt.subplots()
        ax.plot(bdr_x, bdr_y)

        ax.set_aspect('equal')

        plt.show()
        return (bdr_x, bdr_y)
    
    #endregion plot

    @staticmethod
    def line_to_bdry_pts(xs, ys, widths):
        # Calculate differences between consecutive points
        delta_xs = np.diff(xs)
        delta_ys = np.diff(ys)
        
        # Insert the first difference at the beginning and append the last difference at the end
        delta_xs = np.insert(delta_xs, 0, delta_xs[0])
        delta_ys = np.insert(delta_ys, 0, delta_ys[0])
        delta_xs = np.append(delta_xs, delta_xs[-1])
        delta_ys = np.append(delta_ys, delta_ys[-1])
        
        # Calculate the norms of the differences
        norms = np.sqrt(delta_xs**2 + delta_ys**2)

        # Normalize the differences
        normed_del_xs = delta_xs / norms
        normed_del_ys = delta_ys / norms

        # Calculate the average of consecutive normalized vectors
        vec_xs = (normed_del_xs[:-1] + normed_del_xs[1:]) / 2
        vec_ys = (normed_del_ys[:-1] + normed_del_ys[1:]) / 2
        vecs = list(zip(vec_xs, vec_ys))

        # Define rotation matrices for 90 degree clockwise and counter-clockwise rotations
        cw_90_rot = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])
        ccw_90_rot = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2)], [np.sin(-np.pi/2), np.cos(-np.pi/2)]])

        new_points_r = []
        new_points_l = []
        points = list(zip(xs, ys))
        if type(widths) == float:
            widths = np.ones(len(vecs)) * widths
        for i in range(len(vecs)):
            point = np.array(points[i])
            vec = np.array(vecs[i])
            new_points_r.append(point + (vec @ cw_90_rot) * widths[i] / 2)
            new_points_l.append(point + (vec @ ccw_90_rot) * widths[i] / 2)

        new_points_l.reverse()  # Correct the typo "reverese" to "reverse"

        bdr_points = [*new_points_r, *new_points_l]
        bdr_xs, bdr_ys = zip(*bdr_points)
        # bdr_xs, bdr_ys = np.array(bdr_xs), np.array(bdr_ys)
        bdr_xs, bdr_ys = np.array(list(bdr_xs) + [bdr_xs[0]]), np.array(list(bdr_ys) + [bdr_ys[0]])
        # remove np.nans
        bdr_xs, bdr_ys = bdr_xs[~np.isnan(bdr_xs)], bdr_ys[~np.isnan(bdr_ys)]
        
        return bdr_xs, bdr_ys
    
    @staticmethod
    def _compute_length(xs, ys):
        # Compute first derivatives
        dx = np.gradient(xs)
        dy = np.gradient(ys)

        # Calculate incremental distances between points
        distances = np.sqrt(dx**2 + dy**2)
        
        # Compute cumulative length (propagation length)
        length = np.cumsum(distances)
        
        return length
    #endregion plot