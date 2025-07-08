import numpy as np
from .geometry import Geometry

class CompositeGeometry(Geometry):
    def __init__(self, geometries):
        """
        Parameters:
            - geometries: list of geometries. ex) [geo1, geo2, ... , geoN]
        """
        # check whether the obejects in geometries are subclass of Geometry
        for geometry in geometries:
            if not issubclass(type(geometry), Geometry):
                print("geometry should be instance of the subclass of the Geometry")
                return 0
            
        # unfold geometries
        self._geometries = [] # flattend geometries
        for geometry in geometries:
            if geometry._is_composite_geometry: # if the sub_geometry is composite geometry
                for i in range(len(geometry._geometries)):
                    self._geometries.append(geometry._geometries[i])
            else:
                self._geometries.append(geometry)

        self.output_data = None 
        self._is_composite_geometry = True

    def calc_output_data(self):
        for geometry in self._geometries:
            if geometry.output_data == None:
                geometry.calc_output_data()
        output_data = []
        for geometry in self._geometries:
            output_data.append(geometry.output_data)

        self.output_data = output_data
