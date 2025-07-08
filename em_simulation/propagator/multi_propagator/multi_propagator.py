import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import ray
from scipy.integrate import dblquad
from multiprocessing import Pool
from copy import deepcopy
from IPython.display import clear_output

from ... import matrix_calculation_tool as mct

from ..propagator import Propagator
from ...geometry.composite_geometry import CompositeGeometry


class MultiPropagator(Propagator):
    def __init__(self, composite_geometry:CompositeGeometry, force_passive = False, force_unitary = False):
        # check geometry class
        if not issubclass(type(composite_geometry), CompositeGeometry):
            print("geometry should be instance of the subclass of the CompositeGeometry")
            return 0
        self._force_passive = force_passive
        self._force_unitary = force_unitary
        self._is_tmatrix_calculated = 0
        self._is_smatrix_calculated = 0
        self._geometries = composite_geometry._geometries
        self._geometry_intersection_smatrix = None
        
        # list of single propagator variables
        self.propagators = None # list of propagator instances
        self.smatrix = None # list of single propagator smatrices
        self.tmatrix = None # list of single propagator tmatrices
        self._lengths_per_matrices = None   # list of single propagator lengths_per_matrices

        # merged data
        self._merged_smatrix = None # ndarray
        self._radiation_mode_masks = None # list of ndarray (each ndarray is expanded dimension radiation mode mask)
        self._geometries_section_nums = None # list of ints

        # status
        self._is_smatrix_calculated = False
        self._is_tmatrix_calculated = False
        self._is_lengths_per_matrices_calculated = False
        self._is_smatrix_merged = False

        
    def calc_Tmatrix(self):
        pass

    def calc_Smatrix(self):
        pass

    def change_strucutre_length(self, new_length):
        initial_lengths = []
        for propagator in self.propagators:
            initial_length = np.sum(propagator.output_data["delta_zs"])
            initial_lengths.append(initial_length)
        total_length = np.sum(initial_lengths)
        length_ratio_between_structures = initial_lengths/total_length
        
        for i in range(len(self.propagators)):
            propagator = self.propagators[i]
            propagator.change_strucutre_length(new_length*length_ratio_between_structures[i])
        print("Total Length is changed to ", str(new_length * 1e6), "um")

    
    #region calculating intersection smatrix
    def calc_geometry_intersection_smatrix(self):
        geometry_intersection_smatrix = []
        intersection_overlaps_ab, intersection_overlaps_ba = self.calc_geometry_intersections_overlaps()

        for i in tqdm(range(len(intersection_overlaps_ab))):
            overlap_ab = intersection_overlaps_ab[str(i) + "," + str(i+1)]
            overlap_ba = intersection_overlaps_ba[str(i) + "," + str(i+1)]

            # calculating transmission_matrix
            T_ab = self._calculate_transmission_matrix(overlap_ab, overlap_ba)
            T_ba = self._calculate_transmission_matrix(overlap_ba, overlap_ab)

            # calculating reflection matrix
            R_ab = self._calculate_reflection_matrix(overlap_ab, overlap_ba, T_ab)
            R_ba = self._calculate_reflection_matrix(overlap_ba, overlap_ab, T_ba)

            m11 = T_ab - R_ba @ mct._inverse_2D_matrix(T_ba) @ R_ab
            m12 = R_ba @ mct._inverse_2D_matrix(T_ba)
            m21 = (-1) * mct._inverse_2D_matrix(T_ba) @ R_ab
            m22 = mct._inverse_2D_matrix(T_ba)

            # calculate transfer matrix & scatteringg matrix
            mode_count = len(m11)
            transfer_matrix = np.zeros(shape = (2 * mode_count, 2 * mode_count), dtype = complex)
            transfer_matrix[:mode_count, :mode_count] = m11
            transfer_matrix[:mode_count, mode_count:] = m12
            transfer_matrix[mode_count:, :mode_count] = m21
            transfer_matrix[mode_count:, mode_count:] = m22

            scattering_matrix = mct._convert_2Dmatrix(transfer_matrix)
            if self._force_passive:
                scattering_matrix = mct._make_passive_2D(scattering_matrix)
            if self._force_unitary:
                scattering_matrix = mct._find_nearest_unitary_2D(scattering_matrix)
            
            geometry_intersection_smatrix.append(scattering_matrix)

        self._geometry_intersection_smatrix = geometry_intersection_smatrix
        
        return geometry_intersection_smatrix

    def _calculate_transmission_matrix(self, overlap_ab, overlap_ba):
        """
        overlap_ab, overlap_ba : 2D ndarray
        """
        matrix_temp = overlap_ab + np.transpose(overlap_ba, (1,0))
        overlap_tolerance = 0.1 # this value should be adjusted if it does not works.
        transmission_matrix = 2 * mct._inverse_2D_matrix(matrix_temp, overlap_tolerance)

        return transmission_matrix
    
    def _calculate_reflection_matrix(self, overlap_ab, overlap_ba, transmission_matrix):
        """
        overlap_ab, overlap_ba, transmission_matrix : 2D ndarray
        """
        return 0.5 * (np.transpose(overlap_ba, (1,0)) - overlap_ab) @ transmission_matrix
    #endregion calculating intersection smatrix

    #region calculating overlap integral
    def calc_geometry_intersections_overlaps(self):
        ## field 
        # find fields
        print("FDE running..")
        Efields_a_list = []
        Hfields_a_list = []
        Efields_b_list = []
        Hfields_b_list = []
        x_list = []
        y_list = []
        for i in range(len(self._geometries)):
            geometry = self._geometries[i]
            simul_params = geometry.simulation_parameters
            Efields_a, Hfields_a, _, _ = geometry.find_mode_fields(simul_params[-1])
            Efields_b, Hfields_b, x, y = geometry.find_mode_fields(simul_params[0])
            Efields_a_list.append(Efields_a)
            Hfields_a_list.append(Hfields_a)
            Efields_b_list.append(Efields_b)
            Hfields_b_list.append(Hfields_b)
            x_list.append(x)
            y_list.append(y)
        
        # unify field mesh
        clear_output()
        print("Unifying Field mesh")
        x_res = 600
        y_res = 400
        Efields_a_list = self._change_field_mesh(Efields_a_list, x_list, y_list, x_res, y_res)
        Hfields_a_list = self._change_field_mesh(Hfields_a_list, x_list, y_list, x_res, y_res)
        Efields_b_list = self._change_field_mesh(Efields_b_list, x_list, y_list, x_res, y_res)
        Hfields_b_list = self._change_field_mesh(Hfields_b_list, x_list, y_list, x_res, y_res)

        # normalize the fields
        x_final = np.linspace(np.array(x_list).min(), np.array(x_list).max(), num=x_res)
        y_final = np.linspace(np.array(y_list).min(), np.array(y_list).max(), num=y_res)
        clear_output()
        print("Normalizaing Fields..")
        # Efields_a_id_list = []
        # Hfields_a_id_list = []
        # Efields_b_id_list = []
        # Hfields_b_id_list = []
        # for i in range(len(self._geometries)):
        #     Efields_a_id_list[i], Hfields_a_id_list[i] =\
        #           self.normalize_field.remote(Efields_a_list[i], Hfields_a_list[i], x_final, y_final, self._geometries[i])
        #     Efields_b_id_list[i], Hfields_b_id_list[i] =\
        #           self.normalize_field.remote(Efields_b_list[i], Hfields_b_list[i], x_final, y_final, self._geometries[i])
        
        # for i in range(len(self._geometries)):
        #     Efields_a_list[i], Hfields_a_list[i], Efields_b_list[i], Hfields_b_list[i] =\
        #           ray.get(Efields_a_id_list), ray.get(Hfields_a_id_list), ray.get(Efields_b_id_list), ray.get(Hfields_b_id_list)
        for i in range(len(self._geometries)):
            Efields_a_list[i], Hfields_a_list[i] =\
                  self.normalize_field(Efields_a_list[i], Hfields_a_list[i], x_final, y_final, self._geometries[i])
            Efields_b_list[i], Hfields_b_list[i] =\
                  self.normalize_field(Efields_b_list[i], Hfields_b_list[i], x_final, y_final, self._geometries[i])

        # unify the field direction
        clear_output()
        print("Unifying directions..")
        for i in range(len(self._geometries)):
            Efields_a_list[i] = self._unify_field_direction(Efields_a_list[i], self._geometries[i])
            Hfields_a_list[i] = self._unify_field_direction(Hfields_a_list[i], self._geometries[i])
            Efields_b_list[i] = self._unify_field_direction(Efields_b_list[i], self._geometries[i])
            Hfields_b_list[i] = self._unify_field_direction(Hfields_b_list[i], self._geometries[i])

        # field reorder following each geometry track mode number
        clear_output()
        print("Reordering fields..")
        for i in range(len(self._geometries)):
            geometry = self._geometries[i]
            a_mode_names = geometry._tracking_mode_names[-1]
            b_mode_names = geometry._tracking_mode_names[0]
            Efields_a_list[i] = self._reorder_data(a_mode_names, Efields_a_list[i])
            Hfields_a_list[i] = self._reorder_data(a_mode_names, Hfields_a_list[i])
            Efields_b_list[i] = self._reorder_data(b_mode_names, Efields_b_list[i])
            Hfields_b_list[i] = self._reorder_data(b_mode_names, Hfields_b_list[i])

        ## overlap
        # match field dimension
        clear_output()
        print("Matching mode numbers..")
        for i in range(len(x_list)-1):
            Efields_a_list[i], Hfields_b_list[i+1] = self._match_mode_numbers(Efields_a_list[i], Hfields_b_list[i+1])
            Efields_b_list[i+1], Hfields_a_list[i] = self._match_mode_numbers(Efields_b_list[i+1], Hfields_a_list[i])
        
        

        # overlap
        clear_output()
        print("Calculating Overlaps")
        overlap_ab_dict = dict()
        overlap_ba_dict = dict()
        overlap_ab_id_dict = dict()
        overlap_ba_id_dict = dict()
        x_final_id = ray.put(x_final)
        y_final_id = ray.put(y_final)

        for i in range(len(x_list)-1):
            overlap_ab_id_dict[str(i) + "," + str(i+1)] = self._calculate_overlap_matrix.remote(Efields_a_list[i], Hfields_b_list[i+1], x_final_id, y_final_id, 2) # check
            overlap_ba_id_dict[str(i) + "," + str(i+1)] = self._calculate_overlap_matrix.remote(Efields_b_list[i+1], Hfields_a_list[i], x_final_id, y_final_id, 2)
        
        for i in range(len(x_list)-1):
             overlap_ab_dict[str(i) + "," + str(i+1)] = ray.get(overlap_ab_id_dict[str(i) + "," + str(i+1)])
             overlap_ba_dict[str(i) + "," + str(i+1)] = ray.get(overlap_ba_id_dict[str(i) + "," + str(i+1)])
        # for i in range(len(x_list)-1):
        #     overlap_ab_dict[str(i) + "," + str(i+1)] = self._calculate_overlap_matrix(Efields_a_list[i], Hfields_b_list[i+1], x_final, y_final, 2) # check
        #     overlap_ba_dict[str(i) + "," + str(i+1)] = self._calculate_overlap_matrix(Efields_b_list[i+1], Hfields_a_list[i], x_final, y_final, 2)
        return overlap_ab_dict, overlap_ba_dict
    
    #region old calc_geometry_intersections_overlaps

    # def calc_geometry_intersections_overlaps(self):
    #     # the first section of each geometry is b and end section is a
    #     Efields_a_list = []
    #     Hfields_a_list = []
    #     Efields_b_list = []
    #     Hfields_b_list = []
    #     x_list = []
    #     y_list = []

    #     # extract intersection fields data
    #     for i in range(len(self._geometries)):
    #         geometry = self._geometries[i]
    #         simul_params = geometry.simulation_parameters

    #         Efields_a, Hfields_a, _, _ = geometry.find_mode_fields(simul_params[-1])
    #         Efields_b, Hfields_b, x, y = geometry.find_mode_fields(simul_params[0])

    #         # normalize the field
    #         crosssection_x = geometry.data.get_crosssection_x()
    #         crosssection_y = geometry.data.get_crosssection_y()
    #         crosssection_x = 0 if crosssection_x == 'x' else 1 if crosssection_x == 'y' else 2
    #         crosssection_y = 0 if crosssection_y == 'x' else 1 if crosssection_y == 'y' else 2
    #         prop_axis = list({0,1,2} - {crosssection_x,crosssection_y})[0]
    #         Efields_a, Hfields_a = self.normalize_field(Efields_a, Hfields_a, x, y, prop_axis)
    #         Efields_b, Hfields_b = self.normalize_field(Efields_b, Hfields_b, x, y, prop_axis)


    #         # unify field direction
    #         Efields_a = self._unify_mode_numbers(Efields_a, geometry)
    #         Efields_b = self._unify_mode_numbers(Efields_b, geometry)
    #         Hfields_a = self._unify_mode_numbers(Hfields_a, geometry)
    #         Hfields_b = self._unify_mode_numbers(Hfields_b, geometry)

    #         # match mode numbers
    #         Efields_a, Efields_b = self._match_mode_numbers(Efields_a, Efields_b)
    #         Hfields_a, Hfields_b = self._match_mode_numbers(Hfields_a, Hfields_b)

    #         # add backward mode
    #         Efields_a_backward = np.conjugate(Efields_a)
    #         Efields_b_backward = np.conjugate(Efields_b)
    #         Hfields_a_backward = -np.conjugate(Hfields_a)
    #         Hfields_b_backward = -np.conjugate(Hfields_b)

    #         Efields_a = np.concatenate((Efields_a, Efields_a_backward), axis=0)
    #         Efields_b = np.concatenate((Efields_b, Efields_b_backward), axis=0)
    #         Hfields_a = np.concatenate((Hfields_a, Hfields_a_backward), axis=0)
    #         Hfields_b = np.concatenate((Hfields_b, Hfields_b_backward), axis=0)

    #         Efields_a_list.append(Efields_a)
    #         Efields_b_list.append(Efields_b)
    #         Hfields_a_list.append(Hfields_a)
    #         Hfields_b_list.append(Hfields_b)
    #         x_list.append(x)
    #         y_list.append(y)
        
    #     # interpolate fields
    #     x_res = 600
    #     y_res = 400
    #     x_final = np.linspace(np.array(x_list).min(), np.array(x_list).max(), num=x_res)
    #     y_final = np.linspace(np.array(y_list).min(), np.array(y_list).max(), num=y_res)
    #     x_final, y_final = np.meshgrid(x_final, y_final)

    #     Efield_a_ids = np.zeros((len(self._geometries),2*self.mode_count, 3), dtype= object)
    #     Hfield_a_ids = np.zeros((len(self._geometries),2*self.mode_count, 3), dtype= object)
    #     Efield_b_ids = np.zeros((len(self._geometries),2*self.mode_count, 3), dtype= object)
    #     Hfield_b_ids = np.zeros((len(self._geometries),2*self.mode_count, 3), dtype= object)

        # print("interpolating fields..")
        # for i in range(len(self._geometries)):
        #     for mode_num in tqdm(range(2*self.mode_count)):
        #         for dim in range(3):
        #             points = np.array([(x, y) for x in x_list[i] for y in y_list[i]])
        #             E_a_values = Efields_a[i][mode_num, :, :, dim].flatten()
        #             H_a_values = Hfields_a[i][mode_num, :, :, dim].flatten()
        #             H_b_values = Hfields_b[i][mode_num, :, :, dim].flatten()
        #             E_b_values = Efields_b[i][mode_num, :, :, dim].flatten()
        #             Efield_a_ids[i,mode_num, dim] = self.findgrid.remote(points, E_a_values, x_final, y_final)
        #             Hfield_a_ids[i,mode_num, dim] = self.findgrid.remote(points, H_a_values, x_final, y_final)
        #             Efield_b_ids[i,mode_num, dim] = self.findgrid.remote(points, E_b_values, x_final, y_final)
        #             Hfield_b_ids[i,mode_num, dim] = self.findgrid.remote(points, H_b_values, x_final, y_final)
        
        # Efield_a_new = np.zeros((len(self._geometries),2*self.mode_count, 3, len(x_final), len(y_final)), dtype=np.complex64)
        # Hfield_a_new = np.zeros((len(self._geometries),2*self.mode_count, 3, len(x_final), len(y_final)), dtype=np.complex64)
        # Efield_b_new = np.zeros((len(self._geometries),2*self.mode_count, 3, len(x_final), len(y_final)), dtype=np.complex64)
        # Hfield_b_new = np.zeros((len(self._geometries),2*self.mode_count, 3, len(x_final), len(y_final)), dtype=np.complex64)

        # for i in range(len((self._geometries))):
        #     for mode_num in tqdm(range(self.mode_count)):
        #         for dim in range(3):
        #             Efield_a_new[i,mode_num, dim,:,:] = ray.get(Efield_a_ids[i,mode_num, dim])
        #             Hfield_a_new[i,mode_num, dim,:,:] = ray.get(Hfield_a_ids[i,mode_num, dim])
        #             Efield_b_new[i,mode_num, dim,:,:] = ray.get(Efield_b_ids[i,mode_num, dim])
        #             Hfield_b_new[i,mode_num, dim,:,:] = ray.get(Hfield_b_ids[i,mode_num, dim])


        # # Ea_id = ray.put(Efield_a_new)
        # # Ha_id = ray.put(Hfield_a_new)
        # # Eb_id = ray.put(Efield_b_new)
        # # Hb_id = ray.put(Hfield_b_new)
        # x_final_id = ray.put(x_final)
        # y_final_id = ray.put(y_final)

        # # calculate overlaps at intersection 
        # overlap_ab_dict = dict()
        # overlap_ba_dict = dict()
        # overlap_ab_id_dict = dict()
        # overlap_ba_id_dict = dict()
        
        # print("calculating overlaps..")
        # for i in range(len(x_list)-1):
        #     overlap_ab_id_dict[str(i) + "," + str(i+1)] = self._calculate_overlap_matrix.remote(Efield_a_new[i], Hfield_b_new[i+1], x_final_id, y_final_id) # check
        #     overlap_ba_id_dict[str(i) + "," + str(i+1)] = self._calculate_overlap_matrix.remote(Efield_b_new[i+1], Hfield_a_new[i], x_final_id, y_final_id)
        
        # for i in range(len(x_list)-1):
        #      overlap_ab_dict[str(i) + "," + str(i+1)] = ray.get(overlap_ab_id_dict[str(i) + "," + str(i+1)])
        #      overlap_ba_dict[str(i) + "," + str(i+1)] = ray.get(overlap_ba_id_dict[str(i) + "," + str(i+1)])

        # return overlap_ab_dict, overlap_ba_dict
    #endregion old calc_geometry_intersections_overlaps

    def _reorder_data(self, tracking_mode_names, data):
        """
        Parmeters:
            -tracking_mode_names: ndarray of shape (mode_number) with int type
            -data: ndarray shape of (mode_numbers, ... ) data orderd by effective indices.
        Returns:
            - reorderd data: data orderd by polarization modes. ndarray of shape (max_mode_num , ...)
        """
        # num_modes = int(data.shape[0]/2)
        data = deepcopy(data)
        new_shape = (int(tracking_mode_names.max()+1), ) + data.shape[1:]
        reordered_data = np.zeros(shape = new_shape, dtype = data.dtype)

        mode_numbers = tracking_mode_names.shape[0]
        for j in range(mode_numbers):
            index_num = tracking_mode_names[j]
            reordered_data[index_num] = data[j]

        return reordered_data
    

    def _find_prop_axis(self, geometry):
        # prop axis
        crosssection_x = geometry.data.get_crosssection_x()
        crosssection_y = geometry.data.get_crosssection_y()
        crosssection_x = 0 if crosssection_x == 'x' else 1 if crosssection_x == 'y' else 2
        crosssection_y = 0 if crosssection_y == 'x' else 1 if crosssection_y == 'y' else 2
        prop_axis = list({0,1,2} - {crosssection_x,crosssection_y})[0]
        return prop_axis

    def _change_field_mesh(self, fields_list, x_list, y_list, x_res, y_res):
        """
        Parameters:
            - fields_list : list of ndarrays with shape (mode_num*2, len(x), len(y), 3)
            - x_list: list of 1darray (x)
            - y_list: list of 1darray (y)
            - x_res, y_res : int; new resolution of x and y mesh
        """

        # interpolate fields
        x_final = np.linspace(np.array(x_list).min(), np.array(x_list).max(), num=x_res)
        y_final = np.linspace(np.array(y_list).min(), np.array(y_list).max(), num=y_res)
        x_final, y_final = np.meshgrid(x_final, y_final, indexing='ij')
        x_final_id = ray.put(x_final)
        y_final_id = ray.put(y_final)
        field_ids = dict()
        # field_ids = np.zeros((len(self._geometries),2*self.mode_count, 3), dtype= object)

        for i in range(len(self._geometries)):
            for mode_num in tqdm(range(len(fields_list[i]))):
                for dim in range(3):
                    points = np.array([(x, y) for x in x_list[i] for y in y_list[i]])
                    field_values = fields_list[i][mode_num, :, :, dim].flatten()
                    field_ids[(i,mode_num, dim)] = self.findgrid.remote(points, field_values, x_final_id, y_final_id)
        new_field_list = []
        for i in range(len(self._geometries)):
            new_field = np.zeros((len(fields_list[i]), x_res, y_res, 3), dtype= np.complex64)
            for mode_num in range(len(fields_list[i])):
                for dim in range(3):
                    new_field[mode_num, :, :, dim] = ray.get(field_ids[(i,mode_num, dim)])
            new_field_list.append(deepcopy(new_field))
        return new_field_list
    
    @staticmethod
    @ray.remote
    def findgrid(points, values, x_final, y_final):
        return griddata(points, values, (x_final, y_final), method='linear')
    
    @staticmethod
    @ray.remote
    def _calculate_overlap_matrix(Efield, Hfield, x, y, prop_axis):
        """
        Parameters:
            - Efield: ndarray with shape (mode_numbers, len(x), len(y), 3)
            - Hfield: ndarray with shape (mode_numbers, len(x), len(y), 3)
            - x : 1darray
            - y : 1darray
            - prop_axis : int (0 or 1 or 2)
        """
        def compute_differences(arr):
            # Initialize an empty array with the same length as the input array
            output = np.empty_like(arr)
            # Calculate the difference between adjacent elements and store in the output array
            output[:-1] = arr[1:] - arr[:-1]
            # Set the last component as the prior component
            output[-1] = output[-2]
            
            return output
        Efield_expanded = Efield[:,np.newaxis,:,:,:]
        Hfield_expanded = Hfield[np.newaxis,:,:,:,:]
        integrand = (np.cross(Efield_expanded, Hfield_expanded, axis=4)[:,:,:,:,prop_axis]/2)
        x = compute_differences(x)
        y = compute_differences(y)

        weight_mask = np.outer(x, y)
        weight_mask_expanded = weight_mask[np.newaxis, np.newaxis,:,:]
        
        overlap = (weight_mask_expanded * integrand).sum(axis=(2,3))
        return overlap

    def _unify_field_direction(self, field, geometry):
        """
        It matches the x,y,z direction order by crosssection x , crossection y, propagation axis
        Parameters:
            - field: ndarray with shape (mode_nums, len(x), len(y), 3)
            - geometry: geometry instance
        Return:
            - field: reordered field
        """
        crosssection_x = geometry.data.get_crosssection_x()
        crosssection_y = geometry.data.get_crosssection_y()
        crosssection_x = 0 if crosssection_x == 'x' else 1 if crosssection_x == 'y' else 2
        crosssection_y = 0 if crosssection_y == 'x' else 1 if crosssection_y == 'y' else 2
        prop_axis = list({0,1,2} - {crosssection_x,crosssection_y})[0]

        new_field = np.zeros_like(field)
        new_field[:,:,:,0] = field[:,:,:,crosssection_x]
        new_field[:,:,:,1] = field[:,:,:,crosssection_y]
        new_field[:,:,:,2] = field[:,:,:,prop_axis]

        return new_field

    def _unify_mode_numbers(self, field, geometry):
        """
        It matches the x,y,z order by crosssection x , crossection y, propagation axis
        Parameters:
            - field: ndarray with shape (mode_nums, len(x), len(y), 3)
            - geometry: geometry instance
        Return:
            - field: reordered field
        """
        crosssection_x = geometry.data.get_crosssection_x()
        crosssection_y = geometry.data.get_crosssection_y()
        crosssection_x = 0 if crosssection_x == 'x' else 1 if crosssection_x == 'y' else 2
        crosssection_y = 0 if crosssection_y == 'x' else 1 if crosssection_y == 'y' else 2
        prop_axis = list({0,1,2} - {crosssection_x,crosssection_y})[0]

        new_field = np.zeros_like(field)
        new_field[:,:,:,0] = field[:,:,:,crosssection_x]
        new_field[:,:,:,1] = field[:,:,:,crosssection_y]
        new_field[:,:,:,2] = field[:,:,:,prop_axis]

        return new_field
    
    def _match_mode_numbers(self, field_a, field_b):
        """
        It matches the shape of field_ a and field_b by filing the empty mode data to be 0
        Parameters:
            -field_a : ndarray shape (mode_num_a, len(x_a), len(y_a), 3)
            -field_b : ndarray shape (mode_num_b, len(x_b), len(y_b), 3)
        """
        a_modenum, len_xa, len_ya, _ = field_a.shape
        b_modenum, len_xb, len_yb, _ = field_b.shape

        if a_modenum > b_modenum:
            field_b_new = np.zeros(a_modenum, len_xb, len_yb, 3)
            field_b_new[:b_modenum] = field_b
            field_b = field_b_new
        elif a_modenum < b_modenum:
            field_a_new = np.zeros(b_modenum, len_xa, len_ya, 3)
            field_a_new[:a_modenum] = field_a
            field_a = field_a_new
        
        return field_a, field_b
        
    # @staticmethod
    # @ray.remote
    def normalize_field(self, Efield, Hfield, x, y, geometry):
        """
        Parameters:
            - Efields: ndarray of shape ((number of modes), len(x), len(y), 3)
            - Hfields: ndarray of shape ((number of modes), len(x), len(y), 3)
            - x: x_mexh data (1d array)
            - y: y_mesh data (1d array)
            - geometry: geometry instance
            # - prop_axis: propagation axis in FDE file (0: x-aixs, 1: y-axis, 2: z-axis) \n
        Returns:
            - normalized_Efields: ndarray of shape ((number of modes), len(x), len(y), 3)
            - normalized_Hfields: ndarray of shape ((number of modes), len(x), len(y), 3)
        """ 
        def compute_differences(arr):
            # Initialize an empty array with the same length as the input array
            output = np.empty_like(arr)
            # Calculate the difference between adjacent elements and store in the output array
            output[:-1] = arr[1:] - arr[:-1]
            # Set the last component as the prior component
            output[-1] = output[-2]
    
            return output
        
        x = compute_differences(x)
        y = compute_differences(y)

        weight_mask = np.outer(x, y)
        weight_mask_expanded = weight_mask[np.newaxis,:,:]
        # find prop axis
        crosssection_x = geometry.data.get_crosssection_x()
        crosssection_y = geometry.data.get_crosssection_y()
        crosssection_x = 0 if crosssection_x == 'x' else 1 if crosssection_x == 'y' else 2
        crosssection_y = 0 if crosssection_y == 'x' else 1 if crosssection_y == 'y' else 2
        prop_axis = list({0,1,2} - {crosssection_x,crosssection_y})[0]

        # integrand
        integrand = np.cross(Efield, Hfield, axis=3)
        integrand = integrand[:,:,:,prop_axis]  # shape: ((number of modes), len(x), len(y))
        
        # integral to get normalization constant
        normalization_constants = (weight_mask_expanded*integrand/2).sum(axis=(1,2))
        normalization_constants = np.sqrt(normalization_constants)
        normalization_constants_expanded = normalization_constants[:,np.newaxis, np.newaxis, np.newaxis]

        # normalize fields
        normalized_Efields = Efield/normalization_constants_expanded
        normalized_Hfields = Hfield/normalization_constants_expanded

        return normalized_Efields, normalized_Hfields

    #endregion calculating overlap integral

    #region merge smatirx
    def _merge_result_data(self):
        """
        merge smatrix by matching mode nums
        """
        smatrices, intersection_smatrix = self._match_smatrix_dimension()
        radiation_mode_masks = self._match_radiation_mode_mask()
        
        geometries_section_nums = []
        for i in range(len(smatrices)):
            geometries_section_nums.append(len(smatrices[i]))
        
        _, dim, _ = smatrices[0].shape
        merged_smatrix_len = sum(geometries_section_nums) + len(intersection_smatrix)
        merged_smatrix = np.zeros(shape=(merged_smatrix_len, dim, dim), dtype=complex)

        for i in range(len(intersection_smatrix)):
            start_index = sum(geometries_section_nums[:i])+i
            end_index = sum(geometries_section_nums[:i+1])+i
            merged_smatrix[start_index:end_index, :, :] = smatrices[i]
            merged_smatrix[end_index:end_index+1, :, :] = intersection_smatrix[i]
        merged_smatrix[end_index+1:] = smatrices[-1]

        self._merged_smatrix = merged_smatrix
        self._radiation_mode_masks = radiation_mode_masks
        self._geometries_section_nums = geometries_section_nums

        self._is_smatrix_merged = True

        return merged_smatrix
    
    def _match_smatrix_dimension(self):
        smatrices = deepcopy(self.smatrix)
        intersection_smatrices = deepcopy(self._geometry_intersection_smatrix)
        max_dim = 0

        # find maximum dimension
        for smatrix in smatrices:
            section_num, dim1, dim2 = smatrix.shape
            if dim1 > max_dim: max_dim = dim1
        
        expanded_matrices = []
        expanded_intersectoin_matrices = []
        # expand smatrices into maximum dimension
        for smatrix in smatrices:
            expanded_matrices.append(self._expand_matrix_dimension(smatrix, max_dim))

        # expand intersection smatrices into maximum dimension
        for intersection_smatrix in intersection_smatrices:
            intersection_smatrix = intersection_smatrix[np.newaxis, :, :]   # 2D array -> 3D array
            expanded_intersectoin_matrices.append(self._expand_matrix_dimension(intersection_smatrix, max_dim))

        return expanded_matrices, expanded_intersectoin_matrices
    
    def _expand_matrix_dimension(self, matrix, dim):
        """
        expand smatrix or tmatrix to the dim where dim is equal or larger than dimension of input matrix
        Parameters:
            - matrix: smatrix or tmatrix ndarray of shape (section_num, 2*mode_num, 2*mode_num)
            - dim: int, it should satisfies the following conditions \n
                    1) dim > 2*mode_num
                    2) dim is even number
        """
        section_num, input_dim1, input_dim2 = matrix.shape
        if input_dim1 > dim:
            print("dim should be larger than input matrix dimension")
            return
        if dim%2 != 0:
            print("dim should be even number")
        
        matrix_mode_num = int(input_dim1/2)
        new_mode_num = int(dim/2)

        expanded_matrix = np.zeros(shape = (section_num, dim, dim), dtype=np.complex64)
        expanded_matrix[:,:matrix_mode_num, :matrix_mode_num] = matrix[:, :matrix_mode_num, :matrix_mode_num]
        expanded_matrix[:,:matrix_mode_num, new_mode_num:new_mode_num+matrix_mode_num] = matrix[:, :matrix_mode_num, matrix_mode_num:]
        expanded_matrix[:, new_mode_num:new_mode_num+matrix_mode_num, :matrix_mode_num] = matrix[:, matrix_mode_num:, :matrix_mode_num]
        expanded_matrix[:, new_mode_num:new_mode_num+matrix_mode_num, new_mode_num:new_mode_num+matrix_mode_num] = matrix[:, matrix_mode_num:, matrix_mode_num:]

        return expanded_matrix
    
    def _match_radiation_mode_mask(self):
        radiation_mode_masks = []
        max_dim = 0
        for i in range(len(self.propagators)):
            radiation_mode_masks.append(deepcopy(self.propagators[i].output_data["radiation_mode_mask"]))
            dim = radiation_mode_masks[i].shape[1]
            if dim > max_dim: max_dim = dim

        max_mode_num = int(max_dim/2)
        for i in range(len(radiation_mode_masks)):
            section_num, dim = radiation_mode_masks[i].shape
            mode_num = int(dim/2)
            mask = np.ones(shape=(section_num, max_dim), dtype= bool)
            mask[:, :mode_num] = radiation_mode_masks[i][:,:mode_num]
            mask[:, max_mode_num:max_mode_num + mode_num] = radiation_mode_masks[i][:,mode_num:]

            radiation_mode_masks[i] = mask
        
        return radiation_mode_masks
    #endregion merge smatirx

    def _find_lengths_per_matrices(self):
        lengths_per_matrices = []
        for propagator in self.propagators:
            lengths_per_matrices.append(deepcopy(propagator._lengths_per_matrix))
        self._lengths_per_matrices = lengths_per_matrices
        self._is_lengths_per_matrices_calculated = True

    
    #region test functions
    def _test_reorder(self):
        # find field
        geometry = self._geometries[0]
        Efields, Hfields, x, y = geometry.data._extract_field_test()

        Efields_backward = np.conjugate(Efields)
        Hfields_backward = -np.conjugate(Hfields)
        Efields = np.concatenate((Efields, Efields_backward), axis=0)
        Hfields = np.concatenate((Hfields, Hfields_backward), axis=0)
        # get trackmodenames
        mode_names = geometry._tracking_mode_names[0]
        # reorder
        Efields = self._reorder_data(mode_names, Efields)
        
        return Efields
    
    def _test_overlap(self):
        # find field
        geometry = self._geometries[0]
        Efields, Hfields, x, y = geometry.data._extract_field_test()

        Efields_backward = np.conjugate(Efields)
        Hfields_backward = -np.conjugate(Hfields)
        Efields = np.concatenate((Efields, Efields_backward), axis=0)
        Hfields = np.concatenate((Hfields, Hfields_backward), axis=0)

        return self._calculate_overlap_matrix(Efields, Hfields, x, y, 2)


    #endregion test functions
    