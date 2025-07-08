from ..data_extractor.data_extractor import DataExtractor
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import abc


class DirectGeometry(metaclass=abc.ABCMeta):
    def __init__(self, data_extractor:DataExtractor, limit_mode_number = 0, use_existing_data = False):
        self.data = data_extractor
        self.simulation_parameters = None
        self.output_data = None
        self._limit_mode_number = limit_mode_number
        self._tracking_mode_names = None
        self.parameter_names = self.data.parameter_names
        self.wavelength = self.data.wavelength
        self._is_composite_geometry = False
        self._use_existing_data = use_existing_data

    def calc_output_data(self):
        # calculate simulation parameter
        simul_params, delta_zs = self.calc_simulation_parameters()

        if not self._use_existing_data:
            # extract data using data_extractor
            self.data.set_fde_sweep(simul_params)
            if self.data.is_testmode:
                return
            
            self.data.run_fde_sweep()
        neff, TE_pol, overlap_ab, overlap_ba = self.data.post_process_sweep_data(simul_params)

        num_sections, mode_numbers = neff.shape
        mode_numbers = int(mode_numbers/2)
        if self._limit_mode_number and mode_numbers > self._limit_mode_number:
            neff, TE_pol, overlap_ab, overlap_ba = self._reducue_mode_number(neff, TE_pol, overlap_ab, overlap_ba)

        # reorder data by best overlap modes
        mode_links = self._generate_mode_links(overlap_ab)
        tracking_mode_names = self._generate_tracking_mode_names(mode_links)

        # new_tracking_mode_names = self._generate_new_tracking_mode_names(additional_overlap_dict, tracking_mode_names,)
        self._tracking_mode_names = tracking_mode_names

        neff = self._reorder_data(tracking_mode_names, neff)
        TE_pol = self._reorder_data(tracking_mode_names, TE_pol)
        overlap_ab, overlap_ba = self._reorder_overlap(tracking_mode_names, overlap_ab, overlap_ba)

        # equalize overlap phase
        overlap_ab, overlap_ba = self._equalize_overlap_phase(overlap_ab, overlap_ba)

        # additional data
        radiation_mode_mask = self._get_radiation_mode_mask(neff)
        beta = (2*np.pi)*neff/self.wavelength

        # generate output_data
        output_data = dict()
        output_data["overlap_ab"] = overlap_ab
        output_data["overlap_ba"] = overlap_ba
        output_data["delta_zs"] = delta_zs
        output_data["neff"] = neff
        output_data["beta"] = beta
        output_data["radiation_mode_mask"] = radiation_mode_mask
        output_data["TE_pol"] = TE_pol
        output_data["path"] = simul_params

        # dummy output to match with grid dataset base geometry
        output_data["EME_delta_zs"] = delta_zs
        output_data["EME_path"] = simul_params
        output_data["additional_overlap_dict"] = {"ab": dict(), "ba": dict()}
        output_data["index_mapping"] = dict() 

        self.output_data = output_data
        return output_data
    
    @abc.abstractmethod
    def calc_simulation_parameters(self):
        """
        Returns: 
            -simul_params: list of tuple where each tuple is parameter point.\n
                each parameter point should be matched in the grid point in dataset
            -delta_zs: list of length between parameter points. len(delta_zs) == len(simul_params)-1
        """
        pass
    
    #region get data from dataset
    def _get_wavelength(self):
        return self.data.get_wavelength()
    
    def _get_cladding_index(self):
        return self.data.get_cladding_index()
    #endregion get data from dataset
    

    #region reduce mode number
    def _reducue_mode_number(self, neff, TE_pol, overlap_ab, overlap_ba):
        num_sections, mode_numbers = neff.shape
        mode_numbers = int(mode_numbers/2)

        overlap_ab_new = np.zeros(shape = (num_sections-1, 2*self._limit_mode_number, 2*self._limit_mode_number), dtype = np.complex64)
        overlap_ba_new = np.zeros(shape = (num_sections-1, 2*self._limit_mode_number, 2*self._limit_mode_number), dtype = np.complex64)
        neff_new = np.zeros(shape = (num_sections, 2*self._limit_mode_number), dtype=np.complex64)
        TE_pol_new = np.zeros(shape=(num_sections, 2*self._limit_mode_number), dtype=np.float16)

        overlap_ab_new[:, :self._limit_mode_number, :self._limit_mode_number] = overlap_ab[:,:self._limit_mode_number,:self._limit_mode_number]
        overlap_ab_new[:, :self._limit_mode_number, self._limit_mode_number:] = \
            overlap_ab[:,:self._limit_mode_number,mode_numbers:mode_numbers+self._limit_mode_number]
        overlap_ab_new[:, self._limit_mode_number:, :self._limit_mode_number] = \
            overlap_ab[:,mode_numbers:mode_numbers+self._limit_mode_number,:self._limit_mode_number]
        overlap_ab_new[:, self._limit_mode_number:, self._limit_mode_number:] = \
            overlap_ab[:,mode_numbers:mode_numbers+self._limit_mode_number, mode_numbers:mode_numbers+self._limit_mode_number]
        
        overlap_ba_new[:, :self._limit_mode_number, :self._limit_mode_number] = overlap_ba[:,:self._limit_mode_number,:self._limit_mode_number]
        overlap_ba_new[:, :self._limit_mode_number, self._limit_mode_number:] = \
            overlap_ba[:,:self._limit_mode_number,mode_numbers:mode_numbers+self._limit_mode_number]
        overlap_ba_new[:, self._limit_mode_number:, :self._limit_mode_number] = \
            overlap_ba[:,mode_numbers:mode_numbers+self._limit_mode_number,:self._limit_mode_number]
        overlap_ba_new[:, self._limit_mode_number:, self._limit_mode_number:] = \
            overlap_ba[:,mode_numbers:mode_numbers+self._limit_mode_number, mode_numbers:mode_numbers+self._limit_mode_number]
        neff_new[:,:self._limit_mode_number] = neff[:,:self._limit_mode_number]
        neff_new[:,self._limit_mode_number:] = neff[:, mode_numbers:mode_numbers+self._limit_mode_number]
        TE_pol_new[:,:self._limit_mode_number] = TE_pol[:,:self._limit_mode_number]
        TE_pol_new[:,self._limit_mode_number:] = TE_pol[:, mode_numbers:mode_numbers+self._limit_mode_number]
        return neff_new, TE_pol_new, overlap_ab_new, overlap_ba_new
    #endregion reduce mode number
    

    #region track mode
    def _generate_mode_links(self, overlap_matrices):
        """
        Generate mode links between sections.
        Parameters:
            - overlap_matrices: ndarray of shape (num_section-1, 2*num_modes, 2*num_modes)
        
        Returns:
            - mode_link: ndarray of shape (num_section-1, mode_number) \n
                mode in (i+1)th section, j-th mode is linked to i-th section (mode_link[i,j])-th mode
        """
        num_sections = overlap_matrices.shape[0] + 1
        num_modes = int(overlap_matrices.shape[1]/2)

        overlap_matrices = deepcopy(overlap_matrices[:,:num_modes, :num_modes])
        overlap_matrices = np.abs(overlap_matrices)

        dummy_number = -500
        mode_links = np.ones(shape = (num_sections-1, num_modes), dtype=int) * dummy_number

        link_tolerance = 0.5
        for i in range(num_sections-1):
            for j in range(num_modes):
                max_index = np.argmax(overlap_matrices[i,:,j])
                if overlap_matrices[i,max_index,j] > link_tolerance:
                    mode_links[i,j] = max_index
        
        return mode_links
    
    def _generate_tracking_mode_names(self, mode_links):
        """
        Generate mode names for tracking the physical mode along the propagation
        Parameters:
            - mode_links: ndarray of shape (num_section-1, mode_number)
        
        Returns:
            - tracking_mode_names: ndarray of shape (num_section, mode_number)\n
                new_mode_number is total number of modes appeared in the structure
        """
        num_sections, num_modes = mode_links.shape
        num_sections += 1

        tracking_mode_names = np.ones(shape=(num_sections, num_modes), dtype=int) * (-1)

        # initialize the first section name
        for i in range(num_modes):
            tracking_mode_names[0][i] = i
        
        # update remaining section names
        for i in range(num_sections-1):
            for j in range(num_modes):
                linked_index = mode_links[i,j]
                if linked_index < 0:
                    # new mode
                    tracking_mode_names[i+1,j] = tracking_mode_names.max() + 1
                else:
                    # tracked mode
                    tracking_mode_names[i+1,j] = tracking_mode_names[i, linked_index]
        
        return tracking_mode_names
    
    #endregion track mode
    

    #region reorder
    def _reorder_data(self, tracking_mode_names, data):
        """
        Parmeters:
            -tracking_mode_names: ndarray of shape (num_section, mode_number) with int type
            -data: ndarray shape of (section_count, 2*mode_numbers, ... ) data orderd by effective indices.
        Returns:
            - reorderd data: data orderd by polarization modes. ndarray of shape (section_count, max_mode_num , ...)
        """
        num_modes = int(data.shape[1]/2)
        forward_data = deepcopy(data[:,:num_modes])
        backward_data = deepcopy(data[:,num_modes:])

        new_half_shape = data.shape[:1] + (int(tracking_mode_names.max()+1), ) + data.shape[2:]
        new_total_shape = data.shape[:1] + (2*int(tracking_mode_names.max()+1), ) + data.shape[2:]

        reordered_data = np.zeros(shape = new_total_shape, dtype = data.dtype)
        reordered_forward_data = np.zeros(shape = new_half_shape, dtype = data.dtype)
        reordered_backward_data = np.zeros(shape = new_half_shape, dtype = data.dtype)

        section_count = tracking_mode_names.shape[0]
        mode_numbers = tracking_mode_names.shape[1]
        for i in range(section_count):
            for j in range(mode_numbers):
                index_num = tracking_mode_names[i,j]
                reordered_forward_data[i, index_num] = forward_data[i,j]
                reordered_backward_data[i, index_num] = backward_data[i,j]
        
        reordered_data[:,:int(tracking_mode_names.max()+1)] = reordered_forward_data
        reordered_data[:,int(tracking_mode_names.max()+1):] = reordered_backward_data

        return reordered_data

    def _reorder_overlap(self, tracking_mode_names, overlap_ab, overlap_ba):
        """
        Parmeters:
            -tracking_mode_names: ndarray of shape (num_section, mode_number) with int type
            -overlap_ab: ndarray shape of (section_count-1, 2*mode_numbers, 2*mode_numbers ) data orderd by effective indices.\n
                overlap_ab[i,j,k] is ovverlap(E_j(i-th section), H_k(i+1th secion))
            -overlap_bb: ndarray shape of (section_count-1, 2*mode_numbers, 2*mode_numbers ) data orderd by effective indices.\n
                overlap_ba[i,j,k] is ovverlap(E_j(i+1th section), H_k(i-th secion))
        Returns:
            - reorderd data: data orderd by polarization modes. ndarray of shape (section_count-1, 2*max_mode_num, 2*max_mode_num )
        """
        section_count, mode_num, _ = overlap_ab.shape
        mode_num = int(mode_num/2)
        new_mode_num = tracking_mode_names.max()+1

        # Initialize separated overlap matrices
        # expand overlap if reordered matrix is larger than original matrix
        overlap_ab_11 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_12 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_21 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_22 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        overlap_ba_11 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_12 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_21 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_22 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        
        # separate overlap matrices
        overlap_ab_11[:,:mode_num,:mode_num] = overlap_ab[:,:mode_num,:mode_num]
        overlap_ab_12[:,:mode_num,:mode_num] = overlap_ab[:,:mode_num,mode_num:]
        overlap_ab_21[:,:mode_num,:mode_num] = overlap_ab[:,mode_num:,:mode_num]
        overlap_ab_22[:,:mode_num,:mode_num] = overlap_ab[:,mode_num:,mode_num:]

        overlap_ba_11[:,:mode_num,:mode_num] = overlap_ba[:,:mode_num,:mode_num]
        overlap_ba_12[:,:mode_num,:mode_num] = overlap_ba[:,:mode_num,mode_num:]
        overlap_ba_21[:,:mode_num,:mode_num] = overlap_ba[:,mode_num:,:mode_num]
        overlap_ba_22[:,:mode_num,:mode_num] = overlap_ba[:,mode_num:,mode_num:]

        


        # initialize rows reordered overlap_ab and cols reordered overlap_ba
        overlap_ab_11_row_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_12_row_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_21_row_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_22_row_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        overlap_ba_11_col_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_12_col_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_21_col_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_22_col_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        # reorder row of overlap_ab and col of overlap_ba
        for i in range(section_count):
            for j in range(mode_num):
                index_num = tracking_mode_names[i,j]
                overlap_ab_11_row_reordered[i, index_num] = overlap_ab_11[i,j]
                overlap_ab_12_row_reordered[i, index_num] = overlap_ab_12[i,j]
                overlap_ab_21_row_reordered[i, index_num] = overlap_ab_21[i,j]
                overlap_ab_22_row_reordered[i, index_num] = overlap_ab_22[i,j]

                overlap_ba_11_col_reordered[i, :, index_num] = overlap_ba_11[i,:,j]
                overlap_ba_12_col_reordered[i, :, index_num] = overlap_ba_12[i,:,j]
                overlap_ba_21_col_reordered[i, :, index_num] = overlap_ba_21[i,:,j]
                overlap_ba_22_col_reordered[i, :, index_num] = overlap_ba_22[i,:,j]

        # initialize rows and column reordered overlap_ab and overlap_ba  
        overlap_ab_11_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_12_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_21_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_22_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        overlap_ba_11_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_12_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_21_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_22_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        # reorder col of overlap_ab and row of overlap_ba
        for i in range(section_count):
            for j in range(mode_num):
                index_num = tracking_mode_names[i+1,j]
                overlap_ba_11_tot_reordered[i, index_num] = overlap_ba_11_col_reordered[i,j]
                overlap_ba_12_tot_reordered[i, index_num] = overlap_ba_12_col_reordered[i,j]
                overlap_ba_21_tot_reordered[i, index_num] = overlap_ba_21_col_reordered[i,j]
                overlap_ba_22_tot_reordered[i, index_num] = overlap_ba_22_col_reordered[i,j]

                overlap_ab_11_tot_reordered[i, :, index_num] = overlap_ab_11_row_reordered[i,:,j]
                overlap_ab_12_tot_reordered[i, :, index_num] = overlap_ab_12_row_reordered[i,:,j]
                overlap_ab_21_tot_reordered[i, :, index_num] = overlap_ab_21_row_reordered[i,:,j]
                overlap_ab_22_tot_reordered[i, :, index_num] = overlap_ab_22_row_reordered[i,:,j]
        
        # merge reorderd overlap
        reordered_overlap_ab = np.zeros(shape=(section_count, 2*new_mode_num, 2*new_mode_num), dtype = np.complex64)
        reordered_overlap_ba = np.zeros(shape=(section_count, 2*new_mode_num, 2*new_mode_num), dtype = np.complex64)

        reordered_overlap_ab[:,:new_mode_num, :new_mode_num] = overlap_ab_11_tot_reordered
        reordered_overlap_ab[:,:new_mode_num, new_mode_num:] = overlap_ab_12_tot_reordered
        reordered_overlap_ab[:,new_mode_num:, :new_mode_num] = overlap_ab_21_tot_reordered
        reordered_overlap_ab[:,new_mode_num:, new_mode_num:] = overlap_ab_22_tot_reordered

        reordered_overlap_ba[:,:new_mode_num, :new_mode_num] = overlap_ba_11_tot_reordered
        reordered_overlap_ba[:,:new_mode_num, new_mode_num:] = overlap_ba_12_tot_reordered
        reordered_overlap_ba[:,new_mode_num:, :new_mode_num] = overlap_ba_21_tot_reordered
        reordered_overlap_ba[:,new_mode_num:, new_mode_num:] = overlap_ba_22_tot_reordered

        return reordered_overlap_ab, reordered_overlap_ba
    
    #endregion reorder

    def _equalize_overlap_phase(self, overlap_ab, overlap_ba):
        print("Equalizing phase..")
        section_num, mode_count, _ = overlap_ab.shape
        mode_count = int(mode_count/2)
        backward_mode_mask = np.ones(shape=(mode_count,))
        backward_mode_mask = np.concatenate((backward_mode_mask, -backward_mode_mask))

        i = 0
        for i in range(len(overlap_ab) - 1):
            # Create mask for each diagonal element in the 2D slice overlap[i]
            mask = np.sign(np.diagonal(overlap_ab[i]).real)
            mask *= backward_mode_mask

            # mask needs to be reshaped to broadcast correctly along rows and columns
            mask_row = mask[:, np.newaxis]  # Shape (m, 1) for broadcasting across columns
            mask_col = mask[np.newaxis, :]   # Shape (1, m) is fine for broadcasting across rows

            # Apply the mask across all columns for the i-th layer
            overlap_ab[i] *= mask_col

            # Apply the mask across all rows for the (i+1)-th layer
            overlap_ab[i + 1] *= mask_row

        if i > 0:
            # last overlap matrix row phase
            mask = np.sign(np.diagonal(overlap_ab[i+1]).real)
            mask *= backward_mode_mask
            mask_col = mask[np.newaxis, :]
            overlap_ab[i+1] *= mask_col


        for i in range(len(overlap_ba) - 1):
            # for some reason, overlap_ab and overlap_ba have different sign sometimes
            mask = np.sign(np.diagonal(overlap_ba[i]).real)
            mask *= backward_mode_mask

            mask_row = mask[:, np.newaxis]
            mask_col = mask[np.newaxis, :]

            overlap_ba[i] *= mask_row
            overlap_ba[i + 1] *= mask_col
        
        if i > 0:
            # last overlap matrix col phase
            mask = np.sign(np.diagonal(overlap_ba[i+1]).real)
            mask *= backward_mode_mask
            mask_row = mask[:, np.newaxis]
            overlap_ba[i+1] *= mask_row

        return overlap_ab, overlap_ba
    
    def _get_radiation_mode_mask(self, neffs):
        n_clad = self._get_cladding_index()
        neff_real_mask = np.where(neffs.real < n_clad, True, False)
        neff_imag_mask = np.where(np.abs(neffs.imag)> 0.000284, True, False)    # loss >100dB/cm
        mask = neff_real_mask | neff_imag_mask
        return mask
    

    #region plots
    def plot_neffs(self, show_radiation_mode = False):
        """
        It plots the neffs along the propagation
        """
        if self.output_data == None:
            self.calc_output_data()
        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])
        delta_zs = np.insert(delta_zs, 0, [0])
        propagation_length = delta_zs.cumsum()

        neffs = deepcopy(self.output_data["neff"])
        mode_numbers = int(neffs.shape[1]/2)
        neffs = neffs[:,:mode_numbers]
        threshold = 1
        neffs = np.where(neffs < threshold, np.nan, neffs)

        if not show_radiation_mode:
            radiation_mode_mask = deepcopy(self.output_data["radiation_mode_mask"])[:,:mode_numbers]
            neffs = np.where(radiation_mode_mask, np.nan, neffs)

        for i in range(mode_numbers):
            mode_name = "mode " + str(i)
            plt.plot(propagation_length, neffs[:,i], label = mode_name)

        plt.xlabel("propagation length")
        plt.ylabel("neff")
        plt.legend()
        plt.title("neffs along the propagation")
        plt.show()
    

    def plot_TE_polarization_fraction(self, show_radiation_mode = False):
        """
        It plots the neffs along the propagation
        """
        if self.output_data == None:
            self.calc_output_data()
        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])
        delta_zs = np.insert(delta_zs, 0, [0])
        propagation_length = delta_zs.cumsum()

        TE_pols = deepcopy(self.output_data["TE_pol"])
        neffs = deepcopy(self.output_data["neff"])
        mode_numbers = int(neffs.shape[1]/2)
        TE_pols = TE_pols[:,:mode_numbers]
        neffs = neffs[:,:mode_numbers]
        threshold = 1
        TE_pols = np.where(neffs < threshold, np.nan, TE_pols)

        if not show_radiation_mode:
            radiation_mode_mask = deepcopy(self.output_data["radiation_mode_mask"])[:,:mode_numbers]
            TE_pols = np.where(radiation_mode_mask, np.nan, TE_pols)

        for i in range(mode_numbers):
            mode_name = "mode " + str(i)
            plt.plot(propagation_length, TE_pols[:,i], label = mode_name)

        plt.xlabel("propagation length")
        plt.ylabel("TE polarization fraction")
        plt.legend()
        plt.title("TE polarization fraction along the propagation")
        plt.show()
    
    def plot_structure_parameters(self, resolution = 500):
        pass

    def plot_2D_structure(self):
        pass

        #endregion plots
