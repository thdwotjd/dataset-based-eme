from ..geometry.geometry import Geometry
from ..propagator.propagator import Propagator
from ..propagator.multi_propagator.multi_eme import MultiEME
from ..propagator.single_propagator.single_eme import SingleEME

class EME(Propagator):
    def __init__(
        self,
        geometry:Geometry,
        force_passive:bool = False,
        force_unitary:bool = False,
        stability_config=None,
        neff_interpolation:bool = False,
    ):
        """Instantiate an eigenmode expansion propagator.

        :param geometry: Geometry description from which modal data is drawn.
        :type geometry: Geometry
        :param force_passive: Enforce passivity by zeroing gain in the transfer
            matrices.
        :type force_passive: bool
        :param force_unitary: Enforce unitarity when building scattering
            matrices.
        :type force_unitary: bool
        :param stability_config: Numerical-stability settings. If omitted,
            :class:`~em_simulation.propagator.stability.EMEStabilityConfig`
            uses the SiN convergence defaults,
            including a PML sink for direct S-matrix propagation. Pass
            ``EMEStabilityConfig(pml_mode_sink=False)`` for the T-matrix path.
            The sink has final priority over unitary phase projection.
        :type stability_config: EMEStabilityConfig or None
        :param neff_interpolation: Interpolate guided-mode neff along one
            varying dataset parameter for propagation phases. Interface
            overlaps and PML treatment are unchanged. Disabled by default.
        :type neff_interpolation: bool
        """
        self._is_composite_geometry = geometry._is_composite_geometry
        if self._is_composite_geometry:
            self.propagator = MultiEME(
                geometry, force_passive, force_unitary, stability_config,
                neff_interpolation=neff_interpolation,
            )
            self._is_multipropagator = 1
        else:
            self.propagator = SingleEME(
                geometry, force_passive, force_unitary, stability_config,
                neff_interpolation=neff_interpolation,
            )
            self._is_multipropagator = 0
    
    @property
    def neff_interpolation_diagnostics(self):
        """Summarize optional phase interpolation for each geometry.

        A single geometry returns a dictionary with ``enabled``, ``axis``,
        ``guided_mode_steps``, ``interpolated_mode_steps``, and
        ``tracking_conflicts``. A composite geometry returns a list of
        these dictionaries, one per component.
        """
        if self._is_multipropagator:
            return [item.neff_interpolation_diagnostics
                    for item in self.propagator.propagators]
        return self.propagator.neff_interpolation_diagnostics

    def calc_Tmatrix(self):
        """Compute the transfer matrix; the config must disable PML sink."""
        self.propagator.calc_Tmatrix()

    def calc_Smatrix(self):
        """Compute and cache the overall scattering matrix."""
        self.propagator.calc_Smatrix()

    def change_strucutre_length(self, new_length):
        self.propagator.change_strucutre_length(new_length)

