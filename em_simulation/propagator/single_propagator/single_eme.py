import numpy as np
from copy import deepcopy
import warnings

from ... import matrix_calculation_tool as mct
from ...geometry.geometry import Geometry
from ..propagator import Propagator
from ..stability import (
    EMEStabilityConfig,
    mode_reliability_weights,
    new_stability_diagnostics,
    stabilized_inverse,
)

class SingleEME(Propagator):
    """
    The simulation algorithm basically follows the thesis "P. Bienstman, “Rigorous and efficient modelling of wavelenght scale photonic components / Peter Bienstman.,” 2001."
    To see the detail of the mathematical backgorund and terms, see chater2 of the thesis. 
    """
    def __init__(
        self,
        geometry:Geometry,
        force_passive=False,
        force_unitary=False,
        stability_config=None,
    ):
        if stability_config is None:
            stability_config = EMEStabilityConfig()
        elif not isinstance(stability_config, EMEStabilityConfig):
            raise TypeError("stability_config must be an EMEStabilityConfig")

        requested_threshold = stability_config.pml_basis_loss_threshold
        output_threshold = (
            geometry.output_data.get("pml_basis_loss_threshold")
            if geometry.output_data is not None
            else None
        )
        if requested_threshold is not None and (
            geometry.output_data is None
            or output_threshold != requested_threshold
        ):
            geometry.calc_output_data(
                pml_basis_loss_threshold=requested_threshold
            )
        elif requested_threshold is None and output_threshold is not None:
            geometry.calc_output_data(pml_basis_loss_threshold=None)

        super().__init__(geometry, force_passive=force_passive, force_unitary=force_unitary)

        self.overlap_forward_ab = np.asarray(
            self.output_data["overlap_ab"][:,:self.mode_count, :self.mode_count],
            dtype=np.complex128,
        )
        self.overlap_forward_ba = np.asarray(
            self.output_data["overlap_ba"][:,:self.mode_count, :self.mode_count],
            dtype=np.complex128,
        )
        self.beta_forward = np.asarray(
            self.output_data["beta"][:,:self.mode_count], dtype=np.complex128
        )
        self.neff_forward = np.asarray(
            self.output_data["neff"][:,:self.mode_count], dtype=np.complex128
        )
        default_presence = self.neff_forward != 0
        self.mode_present = np.asarray(
            self.output_data.get("mode_present", default_presence)[:,:self.mode_count],
            dtype=bool,
        )
        self.stability_config = stability_config
        self.stability_diagnostics = new_stability_diagnostics()
        self.basis_mode_present = self._calc_basis_mode_presence()
        self._mode_weights = mode_reliability_weights(
            self.neff_forward, self.basis_mode_present, self.stability_config
        )

        self._interface_Tmatrix = None  # ndarray with shape (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count)
        self._interface_Smatrix = None

        # status
        self._is_interface_Tmatrix_calcualted = False
        self._is_interface_Smatrix_calculated = False

    #region main functions
    def calc_Smatrix(self):
        self.stability_diagnostics = new_stability_diagnostics()
        interfaces = self._calc_interface_Smatrix()
        phase_propagations = self._calc_phase_propagation_Smatrix()
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])

        smatrix = np.zeros(
            (2*self.section_count - 2, 2*self.mode_count, 2*self.mode_count),
            dtype=np.complex128,
        )
        lengths_per_matrix = np.zeros(2*self.section_count - 2, dtype=float)
        for i in range(self.section_count - 1):
            smatrix[2*i] = phase_propagations[i]
            smatrix[2*i + 1] = interfaces[i]
            lengths_per_matrix[2*i] = delta_zs[i]

        smatrix = self._restore_sink_phase_matrices(smatrix, phase_propagations)
        smatrix = self._mask_basis_channels(smatrix)
        if self.stability_config.pml_mode_sink:
            self.stability_diagnostics["pml_sink_mode_steps"] = int(
                np.count_nonzero(self._pml_mode_mask())
            )
        self.smatrix = smatrix
        self._lengths_per_matrix = lengths_per_matrix
        self._is_smatrix_calculated = True
        return smatrix

    def calc_Tmatrix(self):
        self._require_tmatrix_sink_disabled()
        interfaces = self._calc_interface_Tmatrix()
        phase_propagations = self._calc_phase_propagation_Tmatrix()
        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])

        total_matrices = np.zeros((2*self.section_count - 2, 2*self.mode_count, 2*self.mode_count), dtype = np.complex64)
        lengths_per_matrix = np.zeros(2*self.section_count-2, dtype = float)
        for i in range(self.section_count - 1):
            total_matrices[2*i] = phase_propagations[i]
            total_matrices[2*i + 1] = interfaces[i]
            lengths_per_matrix[2*i] = delta_zs[i]

        self.tmatrix = deepcopy(total_matrices)
        self._lengths_per_matrix = lengths_per_matrix
        self._is_tmatrix_calculated = True
        return total_matrices
    
    def change_strucutre_length(self, new_length):
        if (
            self.stability_config.pml_mode_sink
            or self.stability_config.pml_basis_loss_threshold is not None
        ):
            self.tmatrix = None
            self._is_tmatrix_calculated = False
        else:
            self.tmatrix = self._find_Tmatrix_new_length(new_length)
        self.smatrix = self._find_Smatrix_new_length(new_length)

        lengths_per_matrix = np.zeros(2*self.section_count-2, dtype = float)

        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])
        initial_length = np.sum(delta_zs)
        length_ratio = new_length/initial_length
        for i in range(self.section_count - 1):
            lengths_per_matrix[2*i] = delta_zs[i]

        self._lengths_per_matrix = lengths_per_matrix*length_ratio


        print("Total Length is changed to ", str(new_length * 1e6), "um")
    
    #endregion main functions



    #region functions used in calc_Tmatrix
    def _calc_interface_Tmatrix(self):
        T12 = self._calc_transmission_matrix(self.overlap_forward_ab, self.overlap_forward_ba)
        T21 = self._calc_transmission_matrix(self.overlap_forward_ba, self.overlap_forward_ab)
        R12 = self._calc_reflection_matrix(self.overlap_forward_ab, self.overlap_forward_ba, T12)
        R21 = self._calc_reflection_matrix(self.overlap_forward_ba, self.overlap_forward_ab, T21)

        inverse_T21 = mct._inverse_3D_matrix_ray(T21)
        m11 = T12 - R21 @ inverse_T21 @ R12
        m12 = R21 @ inverse_T21
        m21 = (-1) * inverse_T21 @ R12
        m22 = inverse_T21

        interface_Tmatrix = np.zeros(shape = (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count), dtype = complex)
        interface_Tmatrix[:,:self.mode_count, :self.mode_count] = m11
        interface_Tmatrix[:,:self.mode_count, self.mode_count:] = m12
        interface_Tmatrix[:,self.mode_count:, :self.mode_count] = m21
        interface_Tmatrix[:,self.mode_count:, self.mode_count:] = m22

        self._interface_Tmatrix = interface_Tmatrix
        self._is_interface_Tmatrix_calcualted = True

        return interface_Tmatrix

    def _calc_interface_Smatrix(self):
        """Build and embed active-basis interface scattering matrices."""

        interface_count = self.section_count - 1
        result = np.zeros(
            (interface_count, 2 * self.mode_count, 2 * self.mode_count),
            dtype=np.complex128,
        )
        for interface_index in range(interface_count):
            local_smatrix, left_indices, right_indices = (
                self._calc_local_interface_Smatrix(interface_index)
            )
            local_smatrix = self._condition_local_interface_smatrix(
                local_smatrix
            )
            global_rows = np.concatenate(
                (right_indices, self.mode_count + left_indices)
            )
            global_columns = np.concatenate(
                (left_indices, self.mode_count + right_indices)
            )
            result[interface_index][
                np.ix_(global_rows, global_columns)
            ] = local_smatrix

            active_support = np.zeros(
                (2 * self.mode_count, 2 * self.mode_count), dtype=bool
            )
            active_support[np.ix_(global_rows, global_columns)] = True
            inactive_values = np.abs(
                result[interface_index][~active_support]
            )
            leakage = (
                float(np.max(inactive_values))
                if inactive_values.size
                else 0.0
            )
            self.stability_diagnostics["max_inactive_leakage"] = max(
                self.stability_diagnostics["max_inactive_leakage"], leakage
            )
            if leakage > 0.0:
                self.stability_diagnostics["inactive_leakage_count"] += 1
                result[interface_index][~active_support] = 0.0

        self._interface_Smatrix = result
        self._is_interface_Smatrix_calculated = True
        return result

    def _calc_local_interface_Smatrix(self, interface_index):
        """Build one active-basis interface S-matrix.

        The overlap blocks may be rectangular when the two adjacent sections
        retain different numbers of modes.  Equations 2.37 and 2.38 are
        applied with a rectangular stabilized pseudoinverse.  The returned
        scattering matrix uses this package's port ordering and sign
        convention: rows are [right-forward, left-backward], columns are
        [left-forward, right-backward], and the upper-right reflection block
        carries the package's legacy minus sign.

        This helper returns only the local result and its active indices;
        _calc_interface_Smatrix embeds it into the fixed global basis.
        """

        if not 0 <= interface_index < self.section_count - 1:
            raise IndexError("interface_index is out of range")

        left_indices = np.flatnonzero(
            self.basis_mode_present[interface_index]
        )
        right_indices = np.flatnonzero(
            self.basis_mode_present[interface_index + 1]
        )
        overlap_lr = self.overlap_forward_ab[interface_index][
            np.ix_(left_indices, right_indices)
        ]
        overlap_rl = self.overlap_forward_ba[interface_index][
            np.ix_(right_indices, left_indices)
        ]
        left_weights = self._mode_weights[interface_index, left_indices]
        right_weights = self._mode_weights[interface_index + 1, right_indices]

        transmission_lr = 2.0 * stabilized_inverse(
            overlap_lr + overlap_rl.T,
            right_weights,
            left_weights,
            self.stability_config,
            self.stability_diagnostics,
            interface_index,
            "local_12",
        )
        transmission_rl = 2.0 * stabilized_inverse(
            overlap_rl + overlap_lr.T,
            left_weights,
            right_weights,
            self.stability_config,
            self.stability_diagnostics,
            interface_index,
            "local_21",
        )

        # Eq. 2.38.  These products remain dimensionally valid for n_left !=
        # n_right, unlike the legacy square-only reflection expressions.
        reflection_ll = (
            0.5 * (overlap_rl.T - overlap_lr) @ transmission_lr
        )
        reflection_rr = (
            0.5 * (overlap_lr.T - overlap_rl) @ transmission_rl
        )

        left_count = left_indices.size
        right_count = right_indices.size
        local_smatrix = np.zeros(
            (left_count + right_count, left_count + right_count),
            dtype=np.complex128,
        )
        local_smatrix[:right_count, :left_count] = transmission_lr
        local_smatrix[:right_count, left_count:] = -reflection_rr
        local_smatrix[right_count:, :left_count] = reflection_ll
        local_smatrix[right_count:, left_count:] = transmission_rl

        return local_smatrix, left_indices, right_indices

    def _condition_local_interface_smatrix(self, local_smatrix):
        """Apply optional physical constraints before global embedding."""

        local_smatrix = np.asarray(local_smatrix, dtype=np.complex128)
        if local_smatrix.size == 0:
            return local_smatrix.copy()
        force_unitary = getattr(self, "_force_unitary", False)
        force_passive = getattr(self, "_force_passive", False)
        if not (force_unitary or force_passive):
            return local_smatrix

        u, singular_values, vh = np.linalg.svd(
            local_smatrix, full_matrices=False
        )
        if force_unitary:
            return u @ vh

        passive_singular_values = np.minimum(singular_values, 1.0)
        return (
            u * passive_singular_values[np.newaxis, :]
        ) @ vh
    
    def _calc_phase_propagation_Tmatrix(self):
        self._require_tmatrix_sink_disabled()
        diagonal_mask = np.eye(self.mode_count, dtype = np.complex64)
        i, j, _ = np.meshgrid(np.arange(0, self.section_count-1),\
                              np.arange(0, self.mode_count),\
                              np.arange(0, self.mode_count),\
                              indexing = 'ij')

        forward_matrix = np.exp(1j*self.beta_forward[i, j]*self.output_data["EME_delta_zs"][i]) * diagonal_mask
        backward_matrix = np.exp((-1j)*self.beta_forward[i, j]*self.output_data["EME_delta_zs"][i]) * diagonal_mask

        result = np.zeros(shape = (self.section_count-1, 2*self.mode_count, 2*self.mode_count), dtype = complex)
        result[:,:self.mode_count, :self.mode_count] = forward_matrix
        result[:,self.mode_count:, self.mode_count:] = backward_matrix

        return result

    def _calc_phase_propagation_Smatrix(self, length_ratio=1.0):
        diagonal_mask = np.eye(self.mode_count, dtype=np.complex128)
        propagation = np.exp(
            1j
            * self.beta_forward[:-1]
            * length_ratio
            * np.asarray(self.output_data["EME_delta_zs"])[:, np.newaxis]
        )
        if self.stability_config.pml_basis_loss_threshold is not None:
            propagation[~self.basis_mode_present[:-1]] = 0.0
        if self.stability_config.pml_mode_sink:
            propagation[self._pml_mode_mask()] = 0.0
        propagation_magnitude = np.abs(propagation)
        max_magnitude = (
            float(np.max(propagation_magnitude))
            if propagation_magnitude.size
            else 0.0
        )
        gain_mask = propagation_magnitude > 1.0 + 1e-12
        gain_count = int(np.count_nonzero(gain_mask))
        self.stability_diagnostics["max_propagation_magnitude"] = max(
            self.stability_diagnostics["max_propagation_magnitude"],
            max_magnitude,
        )
        self.stability_diagnostics["propagation_gain_count"] += gain_count
        if gain_count:
            warnings.warn(
                "Propagation contains modal gain: "
                f"max(abs(exp(i*beta*dz)))={max_magnitude:.6g}.",
                RuntimeWarning,
                stacklevel=2,
            )
        propagation = propagation[:, :, np.newaxis] * diagonal_mask
        result = np.zeros(
            (self.section_count - 1, 2*self.mode_count, 2*self.mode_count),
            dtype=np.complex128,
        )
        result[:, :self.mode_count, :self.mode_count] = propagation
        result[:, self.mode_count:, self.mode_count:] = propagation
        return result

    def _pml_mode_mask(self):
        """Return present PML modes for each longitudinal propagation step."""

        return self.basis_mode_present[:-1] & (
            np.abs(self.neff_forward[:-1].imag)
            >= self.stability_config.pml_sink_loss
        )

    def _calc_basis_mode_presence(self):
        """Return tracked modes retained by the optional loss cutoff."""

        threshold = self.stability_config.pml_basis_loss_threshold
        if threshold is None:
            return self.mode_present.copy()
        return self.mode_present & (
            np.abs(self.neff_forward.imag) < threshold
        )

    def _mask_basis_channels(self, smatrix):
        """Keep cutoff channels exactly zero in the fixed global basis."""

        if self.stability_config.pml_basis_loss_threshold is None:
            return smatrix

        masked = np.array(smatrix, dtype=np.complex128, copy=True)
        for section_index in range(self.section_count - 1):
            left_active = self.basis_mode_present[section_index]
            right_active = self.basis_mode_present[section_index + 1]

            phase_active = np.concatenate((left_active, left_active))
            phase_index = 2 * section_index
            masked[phase_index] *= (
                phase_active[:, np.newaxis] & phase_active[np.newaxis, :]
            )

            # Interface rows are [right-forward, left-backward], while columns
            # are [left-forward, right-backward].
            interface_rows = np.concatenate((right_active, left_active))
            interface_columns = np.concatenate((left_active, right_active))
            interface_index = phase_index + 1
            masked[interface_index] *= (
                interface_rows[:, np.newaxis]
                & interface_columns[np.newaxis, :]
            )
        return masked

    def _restore_sink_phase_matrices(self, smatrix, phase_propagations):
        """Keep absorbing phase blocks after passivity/unitarity projections.

        A complete sink and a unitary phase matrix are mutually exclusive.  When
        both options are requested, the sink deliberately has final priority.
        """

        if self.stability_config.pml_mode_sink:
            smatrix = np.array(smatrix, dtype=np.complex128, copy=True)
            smatrix[0::2] = phase_propagations
        return smatrix

    def _require_tmatrix_sink_disabled(self):
        if self.stability_config.pml_mode_sink:
            raise RuntimeError(
                "PML mode sink is supported only by the direct S-matrix path; "
                "a zero-transmission phase block has no transfer-matrix inverse."
            )
        if self.stability_config.pml_basis_loss_threshold is not None:
            raise RuntimeError(
                "PML basis cutoff is supported only by the direct S-matrix "
                "path; a cutoff phase block has no transfer-matrix inverse."
            )
    
    def _calc_transmission_matrix(self, overlap_ab, overlap_ba):
        # if the result is wrong, try reorder overlap (check old version simulator)
        matrix_temp = overlap_ab + np.transpose(overlap_ba, (0,2,1))
        overlap_tolerance = 0.5 # this value should be adjusted if it does not works.
        result = 2 * mct._inverse_3D_matrix_ray(matrix_temp, tolerance = overlap_tolerance)

        return result
    
    def _calc_reflection_matrix(self, overlap_ab, overlap_ba, transmission_matrix):
        # Eq. 2.38: R_I,II = 1/2 (O_II,I^T - O_I,II) T_I,II.
        result = 0.5 * (
            np.transpose(overlap_ba, (0,2,1)) - overlap_ab
        ) @ transmission_matrix
        return result
    
    #endregion functions used in calc_Tmatrix

    #region functions for change length 
    def _find_Smatrix_new_length(self, new_length):
        if not self._is_interface_Smatrix_calculated:
            self._calc_interface_Smatrix()
        initial_length = np.sum(deepcopy(self.output_data["EME_delta_zs"]))
        length_ratio = new_length / initial_length
        phase_propagations = self._calc_phase_propagation_Smatrix(length_ratio)
        smatrix = np.zeros(
            (2*self.section_count - 2, 2*self.mode_count, 2*self.mode_count),
            dtype=np.complex128,
        )
        for i in range(self.section_count - 1):
            smatrix[2*i] = phase_propagations[i]
            smatrix[2*i + 1] = self._interface_Smatrix[i]

        smatrix = self._restore_sink_phase_matrices(smatrix, phase_propagations)
        smatrix = self._mask_basis_channels(smatrix)

        return smatrix

    def _find_Tmatrix_new_length(self, new_length):
        self._require_tmatrix_sink_disabled()
        if not self._is_interface_Tmatrix_calcualted:
            self._calc_interface_Tmatrix()
        
        interfaces = deepcopy(self._interface_Tmatrix)
        phase_propagations = self._calc_phase_propagation_Tmatrix_new_length(new_length)

        total_matrices = np.zeros((2*self.section_count - 2, 2*self.mode_count, 2*self.mode_count), dtype = np.complex64)
        for i in range(self.section_count - 1):
            total_matrices[2*i] = phase_propagations[i]
            total_matrices[2*i + 1] = interfaces[i]
        
        return total_matrices

    def _calc_phase_propagation_Tmatrix_new_length(self, new_length):
        self._require_tmatrix_sink_disabled()
        initial_length = np.sum(deepcopy(self.output_data["EME_delta_zs"]))
        length_ratio = new_length/initial_length

        diagonal_mask = np.eye(self.mode_count, dtype = complex)
        i, j, _ = np.meshgrid(np.arange(0, self.section_count-1),\
                              np.arange(0, self.mode_count),\
                              np.arange(0, self.mode_count),\
                              indexing = 'ij')

        forward_matrix = np.exp(1j*self.output_data["beta"][i, j]*length_ratio*self.output_data["EME_delta_zs"][i]) * diagonal_mask
        backward_matrix = np.exp((-1j)*self.output_data["beta"][i, j]*length_ratio*self.output_data["EME_delta_zs"][i]) * diagonal_mask

        result = np.zeros(shape = (self.section_count-1, 2*self.mode_count, 2*self.mode_count), dtype = complex)
        result[:,:self.mode_count, :self.mode_count] = forward_matrix
        result[:,self.mode_count:, self.mode_count:] = backward_matrix

        return result
    
    #endregion functions for change length
