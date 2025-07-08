import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import cumtrapz
from scipy.integrate import cumulative_trapezoid as cumtrapz

from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class BezierCurve():
    def __init__(self, control_points, resolution=1000):
        self._control_points = control_points
        self._resolution = resolution
    
    def _calc_parameters(self):
        control_points = self._control_points
        t_values = np.linspace(0, 1, self._resolution)
        x_values, y_values = [], []
        dx_values, dy_values, ddx_values, ddy_values = [], [], [], []
        for t in t_values:
            x, y = self._bezier_curve_point(control_points, t)
            x_values.append(x)
            y_values.append(y)
            dx, dy, ddx, ddy = self.derivative(control_points, t)
            dx_values.append(dx)
            dy_values.append(dy)
            ddx_values.append(ddx)
            ddy_values.append(ddy)
        
        curvatures = [self.bezier_curvature(dx, dy, ddx, ddy) for dx, dy, ddx, ddy in zip(dx_values, dy_values, ddx_values, ddy_values)][:-1]
        prop_angles = np.arctan2(dy_values, dx_values)[:-1]  # Calculate tangential angles
        distances = np.sqrt(np.diff(x_values)**2 + np.diff(y_values)**2)
        propagation_length = np.insert(cumtrapz(distances), 0, 0)  # Insert 0 at the start for the initial point
        # top_widths = self._top_width * np.ones(self._resolution - 1)
        return propagation_length, curvatures, prop_angles
    
    def calc_total_length(self):
        control_points = self._control_points
        t_values = np.linspace(0, 1, self._resolution)
        x_values, y_values = [], []
        dx_values, dy_values, ddx_values, ddy_values = [], [], [], []
        for t in t_values:
            x, y = self._bezier_curve_point(control_points, t)
            x_values.append(x)
            y_values.append(y)
            dx, dy, ddx, ddy = self.derivative(control_points, t)
            dx_values.append(dx)
            dy_values.append(dy)
            ddx_values.append(ddx)
            ddy_values.append(ddy)
        
        distances = np.sqrt(np.diff(x_values)**2 + np.diff(y_values)**2)
        return cumtrapz(distances)[-1]
    
    def _bezier_curve_point(self, control_points, t):
        """Calculate the position of a point in the Bezier curve."""
        n = len(control_points) - 1
        x = y = 0
        for i, (px, py) in enumerate(control_points):
            binomial_coeff = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
            bernstein_poly = binomial_coeff * (t ** i) * ((1 - t) ** (n - i))
            x += px * bernstein_poly
            y += py * bernstein_poly
        return x, y
    
    def _calc_xy(self):
        control_points = self._control_points
        t_values = np.linspace(0, 1, self._resolution)
        x_values, y_values = [], []
        for t in t_values:
            x, y = self._bezier_curve_point(control_points, t)
            x_values.append(x)
            y_values.append(y)
        
        return x_values, y_values
    
    def derivative(self, control_points, t):
        """Calculate the first and second derivatives of the Bezier curve."""
        n = len(control_points) - 1
        dx = dy = ddx = ddy = 0
        for i, (px, py) in enumerate(control_points):
            if i != n:
                dx += n * (control_points[i+1][0] - px) * np.math.factorial(n-1) / (np.math.factorial(i) * np.math.factorial(n-1-i)) * (t ** i) * ((1 - t) ** (n-1-i))
                dy += n * (control_points[i+1][1] - py) * np.math.factorial(n-1) / (np.math.factorial(i) * np.math.factorial(n-1-i)) * (t ** i) * ((1 - t) ** (n-1-i))
            if i != n and i != n-1:
                ddx += n * (n-1) * ((control_points[i+2][0] - 2*control_points[i+1][0] + px) * (t ** i) * ((1 - t) ** (n-2-i)))
                ddy += n * (n-1) * ((control_points[i+2][1] - 2*control_points[i+1][1] + py) * (t ** i) * ((1 - t) ** (n-2-i)))
        return dx, dy, ddx, ddy

    def bezier_curvature(self, dx, dy, ddx, ddy):
        """Calculate the curvature of the Bezier curve."""
        return (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    
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
    
    
    

    
    