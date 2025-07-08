import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.special import fresnel
import math


class PartialEulerCurve():
    def __init__(self, input_angle, total_length, bend_angle, p, resolution=1000):
        self._input_angle = input_angle
        self._total_length = total_length
        self._bend_angle = bend_angle
        self._p = p
        self._num_points = resolution
    
    def _calc_parameters(self):
        bend_angle = self._bend_angle
        input_angle = self._input_angle
        L = self._total_length
        p = self._p
        num_points = self._num_points
    
        # Calculate the segment lengths
        L_euler = p * L  # Total length of both Euler spiral sections combined
        L_constant = L - L_euler  # Length of the constant curvature section

        # Define curvature for each segment
        kappa_max = np.radians(bend_angle)/L/(1-p/2) #* np.pi/180
        prop_length = np.linspace(0, L, num_points)  # Discretize the curve length
        
        # Initialize arrays to hold curvature, angles, and coordinates
        curvature = np.zeros(num_points)
        prop_angle = np.zeros(num_points)

        # Assign curvature values
        for i in range(num_points):
            if prop_length[i] <= L_euler / 2:  # First Euler spiral section
                curvature[i] = kappa_max * (prop_length[i] / (L_euler / 2))
            elif prop_length[i] <= L_euler / 2 + L_constant:  # Constant curvature section
                curvature[i] = kappa_max
            else:  # Second Euler spiral section
                curvature[i] = kappa_max * (1 - (prop_length[i] - L_euler / 2 - L_constant) / (L_euler / 2))
        
        # Calculate angles and coordinates
        curvature *= (-1) # Adapt to the Lumerical convention
        ds = np.diff(prop_length, prepend=0)
        prop_angle = np.cumsum(curvature) * ds
        prop_angle += np.radians(input_angle)
        prop_angle *= (-1)

        return prop_length, curvature, prop_angle, ds
    
    def calc_total_length(self):
        return self._total_length
    
    def _calc_xy(self):
        x = np.cumsum(np.cos(self._prop_angle + np.radians(self._input_angle))) * self._ds
        y = np.cumsum(np.sin(self._prop_angle + np.radians(self._input_angle))) * self._ds

        return x,y
    
    #region plot
    def plot_2D_structure(self):
        x,y = self._calc_xy()
        # plt.plot(x,y)
        # plt.axis('scaled')
        ##
        # Create main figure and plot
        fig, main_ax = plt.subplots()
        main_ax.plot(x,y)  # Replace with your actual plotting code

        # Create an inset of width 0.2 and height 0.2, located in the upper right corner (location code 1)
        ax_inset = inset_axes(main_ax, width="20%", height="20%", loc=2)

        # Draw an inset plot or arrow indicating direction
        ax_inset.arrow(0.3, 0.3, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k') # n_o axis
        ax_inset.arrow(0.3, 0.3, 0, 0.4, head_width=0.1, head_length=0.1, fc='k', ec='k') # n_e axis

        # Optional: remove axis and ticks for inset
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])

        # Set aspect of inset and main plot
        ax_inset.set_aspect('equal')
        main_ax.set_aspect('equal')

        # Set the inset axes limits
        ax_inset.set_xlim(0, 1)
        ax_inset.set_ylim(0, 1)

        # Add labels to the inset arrows
        ax_inset.text(0.8, 0.05, '$n_o$', verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='blue')
        ax_inset.text(0.25, 0.6, '$n_e$', verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='blue')

        # remove inset boundary
        for spine in ax_inset.spines.values():
            spine.set_visible(False)

        plt.show()
    
    def plot_structure_parameters(self, resolution = 500):
        pass
    #endregion plot


class Euler():
    def __init__(self, p, input_angle_deg, bend_angle_deg, effective_radius, steps: int = 500):
        if bend_angle_deg > 180:
            raise ValueError("angle_deg must be smaller or equal to 180.")
        if p > 1 or p < 0:
            raise ValueError("p must be larger than 0 and smaller than 1.")
        if steps <= 0:
            raise ValueError("steps must have positive even integer value.")
        if effective_radius <= 0:
            raise ValueError("effective_radius must have positive value.")
        if type(steps) != int:
            raise TypeError("Datatype of steps must be int.")
        
        self.p = p
        self.input_angle = np.deg2rad(input_angle_deg)
        self.bend_angle = np.deg2rad(bend_angle_deg)
        angle = np.abs(self.bend_angle)
        steps -= 3

        self.euler_steps, self.circle_steps = self.divide_steps_euler(np.round(steps / 2), p)
        self.half_euler_length = np.sqrt(p * angle / np.pi)
        t = np.linspace(0, self.half_euler_length, self.euler_steps)

        self.euler_end_radius = 1 / np.sqrt(np.pi * p * angle)
        euler_y, euler_x = fresnel(t)
        euler_endpoint_y, euler_endpoint_x = fresnel(self.half_euler_length)
        delta_x = euler_endpoint_x - self.euler_end_radius * np.sin(p * angle / 2)
        delta_y = euler_endpoint_y - self.euler_end_radius * (1 - np.cos(p * angle / 2))
        self.total_length = 2 * self.half_euler_length + self.euler_end_radius * angle * (1 - p)
        t2 = np.linspace(self.half_euler_length, self.total_length / 2, self.circle_steps)
        circle_x = self.euler_end_radius * np.sin((t2 - self.half_euler_length) /
                                                  self.euler_end_radius + p * angle / 2) + delta_x
        circle_y = self.euler_end_radius * (1 - np.cos((t2 - self.half_euler_length) / self.euler_end_radius + p * angle / 2)) + delta_y
        self.rescale_factor = effective_radius / (circle_y[len(circle_y) - 1] + circle_x[len(circle_x) - 1] / np.tan(angle / 2))
        self.radius_var = self.rescale_factor / (np.pi * t)
        half_x = np.concatenate((self.rescale_factor * euler_x, self.rescale_factor * circle_x), axis=0)
        half_y = np.concatenate((self.rescale_factor * euler_y, self.rescale_factor * circle_y), axis=0)
        a = self.rescale_factor * circle_y[len(circle_y) - 1] - effective_radius
        b = -self.rescale_factor * circle_x[len(circle_x) - 1]
        c = -b * effective_radius
        m = np.sqrt(a**2 + b**2)
        a_p = a / m
        b_p = b / m
        c_p = c / m
        point = np.stack((half_x, half_y), axis=1)
        px = np.zeros(shape=(1, int(steps/2)))
        py = np.zeros(shape=(1, int(steps/2)))
        for i in range(len(point)):
            d = a_p * point[i][0] + b_p * point[i][1] + c_p
            j = int(steps/2) - 1 - i
            px[0][j] = point[i][0] - 2 * a_p * d
            py[0][j] = point[i][1] - 2 * b_p * d
        self.x = np.concatenate((half_x, px), axis=None)
        self.y = np.concatenate((half_y, py), axis=None)
        circle_prop_length_1h = np.linspace(self.half_euler_length, self.total_length/2, self.circle_steps)
        circle_prop_length_2h = np.linspace(circle_prop_length_1h[-1], self.total_length - self.half_euler_length, self.circle_steps)
        self.propagation_length = np.concatenate((t, circle_prop_length_1h, circle_prop_length_2h, circle_prop_length_2h[-1] + t), axis=None)

        # some parameters
        self._total_length = self.get_length()
        self._num_points = steps

    def _calc_parameters(self):
        x = self.getx()
        y = self.gety()
        if self.bend_angle < 0: y = -y
        x_final = np.cos(self.input_angle) * x - np.sin(self.input_angle) * y
        y_final = np.sin(self.input_angle) * x + np.cos(self.input_angle) * y

        # prop angles
        prop_angle = self.prop_angle()
        if self.bend_angle < 0: prop_angle = -prop_angle
        prop_angle += self.input_angle
        prop_angle = np.insert(prop_angle[:-1], -1, prop_angle[-2])
        prop_angle = np.concatenate((prop_angle[:self.euler_steps-1], \
                                     prop_angle[self.euler_steps:self.euler_steps+self.circle_steps-1],\
                                          prop_angle[self.euler_steps+self.circle_steps:-1])) # len: steps

        # curvatures
        total_length = self.get_length()
        # curvature_max = 2*self.bend_angle/(total_length*(1+self.p))
        curvature_max = self.bend_angle/(total_length*(1-self.p/2))
        curvatures1 = np.linspace(0, curvature_max, self.euler_steps-1)
        curvatures3 = np.linspace(curvature_max, 0, self.euler_steps-1)
        curvatures2 = np.ones(2*self.circle_steps-1) * curvature_max
        curvature = np.concatenate((curvatures1, curvatures2, curvatures3)) # steps

        prop_length = np.unique(self.prop_length()) # steps
        ds = np.diff(prop_length, prepend=0)

        return prop_length, curvature, prop_angle, ds
    
    def calc_total_length(self):
        return self.get_length()
    
    def _calc_xy(self):
        x = self.getx()
        y = self.gety()
        return x,y
    
    #region plot
    def plot_2D_structure(self):
        x,y = self._calc_xy()
        # plt.plot(x,y)
        # plt.axis('scaled')
        ##
        # Create main figure and plot
        fig, main_ax = plt.subplots()
        main_ax.plot(x,y)  # Replace with your actual plotting code

        # Create an inset of width 0.2 and height 0.2, located in the upper right corner (location code 1)
        ax_inset = inset_axes(main_ax, width="20%", height="20%", loc=2)

        # Draw an inset plot or arrow indicating direction
        ax_inset.arrow(0.3, 0.3, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k') # n_o axis
        ax_inset.arrow(0.3, 0.3, 0, 0.4, head_width=0.1, head_length=0.1, fc='k', ec='k') # n_e axis

        # Optional: remove axis and ticks for inset
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])

        # Set aspect of inset and main plot
        ax_inset.set_aspect('equal')
        main_ax.set_aspect('equal')

        # Set the inset axes limits
        ax_inset.set_xlim(0, 1)
        ax_inset.set_ylim(0, 1)

        # Add labels to the inset arrows
        ax_inset.text(0.8, 0.05, '$n_o$', verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='blue')
        ax_inset.text(0.25, 0.6, '$n_e$', verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='blue')

        # remove inset boundary
        for spine in ax_inset.spines.values():
            spine.set_visible(False)

        plt.show()
    
    #endregion plot
    

    #region original_functions
    def divide_steps_euler(self, steps, p):
        euler_steps = 0
        circle_steps = 0
        if round(steps * p) + round(steps * (1 - p)) == steps:
            euler_steps = round(steps * p)
            circle_steps = round(steps * (1 - p))
        elif round(steps * p) + round(steps * (1 - p)) == steps - 1:
            if p >= 0.5:
                euler_steps = round(steps * p)
                circle_steps = round(steps * (1 - p)) + 1
            else:
                euler_steps = round(steps * p) + 1
                circle_steps = round(steps * (1 - p))
        elif round(steps * p) + round(steps * (1 - p)) == steps + 1:
            if p >= 0.5:
                euler_steps = round(steps * p) - 1
                circle_steps = round(steps * (1 - p))
            else:
                euler_steps = round(steps * p)
                circle_steps = round(steps * (1 - p)) - 1
        return [euler_steps, circle_steps]

    def get_euler_steps(self):
        return self.euler_steps * 2

    def get_circle_steps(self):
        return self.circle_steps * 2

    def mirror_vertical(self):
        self.x = -self.x

    def mirror_horizontal(self):
        self.y = -self.y

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def get_length(self):
        return self.rescale_factor * self.total_length

    def get_euler_length(self):
        return 2 * self.rescale_factor * self.half_euler_length

    def get_circle_length(self):
        return self.rescale_factor * (self.total_length - 2 * self.half_euler_length)

    def get_euler_endpoint_radius(self):
        return self.euler_end_radius * self.rescale_factor

    def get_radius_points(self):
        return np.concatenate((self.radius_var, self.radius_var[len(self.radius_var) - 1]
                               * np.ones(shape=(1, int(self.circle_steps * 2))), np.flipud(self.radius_var)), axis=None)

    def prop_angle(self):
        prop_angle_val = np.empty(shape=(1, len(self.getx())))
        for i in range(len(self.getx())):
            if i == len(self.getx()) - 1:
                prop_angle_val[0][i] = np.pi
            elif self.getx()[i] == self.getx()[i + 1]:
                prop_angle_val[0][i] = prop_angle_val[0][i - 1]
            else:
                prop_angle_val[0][i] = np.arctan(
                    (self.gety()[i + 1] - self.gety()[i]) / (self.getx()[i + 1] - self.getx()[i]))
                if prop_angle_val[0][i] <= 0:
                    prop_angle_val[0][i] = prop_angle_val[0][i] + np.pi
        return prop_angle_val[0, :]

    def prop_length(self):
        return self.rescale_factor * self.propagation_length

    def compress(self, num_of_data: int):
        if type(num_of_data) != int:
            raise TypeError("Datatype of ""num_of_data"" must be int.")
        rad = self.get_radius_points()
        ang = self.prop_angle()
        length = self.prop_length()
        indices_len = np.linspace(0, len(length) - 1, num_of_data, dtype=int)
        indices_rad = np.linspace(0, len(rad) - 1, num_of_data, dtype=int)
        indices_ang = np.linspace(0, len(ang) - 1, num_of_data, dtype=int)
        len_compressed = np.take(length, indices_len)
        rad_compressed = np.take(rad, indices_rad)
        ang_compressed = np.take(ang, indices_ang)
        datas = np.row_stack((len_compressed, rad_compressed, ang_compressed))
        return datas

    #endregion original_functions

#region newest version
class PartialEulerBend:
    def __init__(self, p, input_angle_deg, bend_angle_deg, effective_radius, steps:int = 1000):
        """
        Create a partial Euler bend.
        
        Parameters:
        -----------
        p : float
            Fraction (between 0 and 1) of the bend that is formed by Euler (transition) segments.
            The first p/2 portion is the ramp-up and the last p/2 is the ramp-down.
        input_angle_deg : float
            Input port angle in degrees.
        bend_angle_deg : float
            Total bending angle in degrees.
        effective_radius : float
            The effective radius (R_eff in meters) of a circular arc that approximates the Euler bend.
            Geometrically, it is defined as the distance from the center of curvature of the approximating 
            circular bend to the bend’s endpoint.
        resolution : int
            Number of points used to sample the bend.
        """
        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1")
        self.p = p
        self.input_angle_deg = input_angle_deg
        self.bend_angle_deg = bend_angle_deg
        self.effective_radius = effective_radius
        self.resolution = steps

        # Convert angles to radians.
        self.input_angle_rad = math.radians(input_angle_deg)
        self.theta_bend = math.radians(bend_angle_deg)

        # Determine L_total such that the Euler bend’s endpoint matches that of
        # a circular bend with effective radius R_eff.
        self._total_length = self.calc_total_length()
        
        # Now determine kappa_max from the integrated angle relation:
        self.kappa_max = self.theta_bend / (self._total_length * (1 - self.p/2))
        
        # Define segment lengths.
        self.L1 = (self.p / 2) * self._total_length  # ramp-up and ramp-down length
        self.L2 = (1 - self.p) * self._total_length    # constant-curvature segment
        
        # Create the propagation coordinate.
        self.s = np.linspace(0, self._total_length, self.resolution)
        
        # Compute parameters and (x,y) coordinates.
        self.prop_length, self.curvature, self.prop_angle = self._calc_parameters()
        self.x, self.y = self._calc_xy()
    
    def _calc_parameters(self):
        """
        Compute the propagation lengths, curvature, and propagation angles along the Euler bend.
        
        Returns:
        --------
        prop_length : np.ndarray
            Array of propagation lengths (meters) from 0 to L_total.
        curvature : np.ndarray
            Array of curvature values (in 1/m) along the bend.
        prop_angle : np.ndarray
            Array of cumulative propagation angles (in radians), starting from input_angle.
        """
        s = self.s
        curvature = self._curvature_profile(s, self._total_length)
        ds = s[1] - s[0]
        cumulative_angle = np.empty_like(s)
        cumulative_angle[0] = 0.0
        # Trapezoidal integration.
        cumulative_angle[1:] = np.cumsum((curvature[:-1] + curvature[1:]) / 2.0 * ds)
        # Add the initial input angle.
        prop_angle = self.input_angle_rad + cumulative_angle
        return s, curvature, prop_angle

    def _calc_xy(self):
        """
        Compute the (x,y) coordinates of the bend.
        
        Returns:
        --------
        x : np.ndarray
            x-coordinates (meters) of the bend.
        y : np.ndarray
            y-coordinates (meters) of the bend.
        """
        theta = self.prop_angle
        ds = self.s[1] - self.s[0]
        x = np.empty_like(self.s)
        y = np.empty_like(self.s)
        x[0] = 0.0
        y[0] = 0.0
        x[1:] = np.cumsum((np.cos(theta[:-1]) + np.cos(theta[1:])) / 2.0 * ds)
        y[1:] = np.cumsum((np.sin(theta[:-1]) + np.sin(theta[1:])) / 2.0 * ds)
        return x, y
    
    # def _solve_total_length(self):
    def calc_total_length(self):
        """
        Solve for L_total such that the endpoint of the Euler bend matches the endpoint
        of an equivalent circular bend with effective radius R_eff.
        
        For a circular arc (with input angle zero and turning left) of bending angle theta_bend,
        the endpoint is:
            x_d = R_eff * sin(theta_bend)
            y_d = R_eff * (1 - cos(theta_bend))
        
        We find L_total such that the Euler bend’s computed x_end equals x_d.
        (Because the x–coordinate is monotonic in this geometry, it suffices as the matching condition.)
        """
        # Desired endpoint for a circular arc.
        x_des = self.effective_radius * math.sin(self.theta_bend)
        # We'll bracket L_total. A circular arc would have L_circ = R_eff*theta_bend.
        L_low = self.effective_radius * self.theta_bend  # lower bound: circular arc (shorter)
        L_high = 2 * L_low  # a guess: Euler bend is longer
        
        def f(L_total):
            x_end, _ = self._compute_endpoint(L_total)
            return x_end - x_des
        
        # Ensure that f(L_low) and f(L_high) have opposite signs.
        f_low = f(L_low)
        f_high = f(L_high)
        if f_low * f_high > 0:
            raise ValueError("Cannot bracket the solution for total length. Adjust initial bounds.")
        
        # Bisection.
        tol = 1e-9
        max_iter = 100
        for _ in range(max_iter):
            L_mid = 0.5*(L_low + L_high)
            f_mid = f(L_mid)
            if abs(f_mid) < tol:
                return L_mid
            if f_low * f_mid < 0:
                L_high = L_mid
                f_high = f_mid
            else:
                L_low = L_mid
                f_low = f_mid
        return 0.5*(L_low + L_high)

    def _curvature_profile(self, s, L_total):
        """
        Given a propagation coordinate array s (0 to L_total), compute the curvature profile
        using the piecewise definition.
        """
        L1 = (self.p / 2) * L_total
        L2 = (1 - self.p) * L_total
        curvature = np.empty_like(s)
        
        # Boolean masks for segments.
        ramp_up = s <= L1
        constant = (s > L1) & (s <= L1 + L2)
        ramp_down = s > L1 + L2
        
        # Ramp-up: curvature increases linearly from 0 to kappa_max.
        curvature[ramp_up] = self.kappa_max * (s[ramp_up] / L1)
        # Constant-curvature segment.
        curvature[constant] = self.kappa_max
        # Ramp-down: curvature decreases linearly from kappa_max to 0.
        curvature[ramp_down] = self.kappa_max * (1 - (s[ramp_down] - (L1 + L2)) / L1)
        
        return curvature

    def _compute_endpoint(self, L_total):
        """
        For a given total length L_total, compute the endpoint (x_end, y_end) of the Euler bend.
        Here kappa_max is taken from the relation:
            theta_bend = kappa_max * L_total*(1 - p/2)
        and the integration is performed with high resolution.
        
        This is done in a coordinate system where the input port is at (0,0) and the tangent is
        along the x–axis.
        """
        # Determine kappa_max for this trial L_total.
        kappa_max_trial = self.theta_bend / (L_total * (1 - self.p/2))
        L1 = (self.p / 2) * L_total
        L2 = (1 - self.p) * L_total
        
        # Use a temporary high-resolution sampling.
        N = 1000
        s_temp = np.linspace(0, L_total, N)
        curvature = np.empty_like(s_temp)
        
        # Define curvature piecewise.
        ramp_up = s_temp <= L1
        constant = (s_temp > L1) & (s_temp <= L1 + L2)
        ramp_down = s_temp > L1 + L2
        
        curvature[ramp_up] = kappa_max_trial * (s_temp[ramp_up] / L1)
        curvature[constant] = kappa_max_trial
        curvature[ramp_down] = kappa_max_trial * (1 - (s_temp[ramp_down] - (L1 + L2)) / L1)
        
        # Integrate to get the propagation angle.
        ds = s_temp[1] - s_temp[0]
        # cumulative integration via trapezoidal rule:
        dtheta = (curvature[:-1] + curvature[1:]) / 2 * ds
        theta_cumulative = np.concatenate(([0], np.cumsum(dtheta)))
        # The local tangent is: theta(s) = input_angle + integrated angle.
        # In our local coordinate system, input_angle = 0.
        
        # Integrate to get x and y.
        x_temp = np.empty_like(s_temp)
        y_temp = np.empty_like(s_temp)
        x_temp[0] = 0.0
        y_temp[0] = 0.0
        dx = (np.cos(theta_cumulative[:-1]) + np.cos(theta_cumulative[1:])) / 2 * ds
        dy = (np.sin(theta_cumulative[:-1]) + np.sin(theta_cumulative[1:])) / 2 * ds
        x_temp[1:] = np.cumsum(dx)
        y_temp[1:] = np.cumsum(dy)
        
        return x_temp[-1], y_temp[-1]

#endregion newest version
