DREME documentation
===================

DREME is a dataset-based eigenmode expansion (EME) simulator. Its
:class:`~em_simulation.propagator.eme.EME` propagator provides numerical
stability controls and optional one-axis effective-index interpolation.

EME stability and phase options
-------------------------------

``EMEStabilityConfig`` uses the SiN convergence settings by default. It
stabilizes interface inversions and scattering-matrix feedback, and can absorb
strongly attenuated PML modes during propagation. Loss thresholds refer to
the dimensionless quantity
:math:`\ell_m = |\operatorname{Im}(n_{\mathrm{eff},m})|`. The defaults were
selected for a SiN dataset at a wavelength of 1.55 micrometers. Check their
sensitivity with a different dataset or wavelength.

.. list-table:: Stability settings at a glance
   :header-rows: 1
   :widths: 30 15 55

   * - Setting
     - Default
     - Role
   * - ``guided_loss``
     - ``2.84e-5``
     - Loss at or below which an interface mode has full reliability weight.
   * - ``pml_regularization_loss``
     - ``2.84e-3``
     - Loss at or above which that weight is zero. Intermediate weights decrease logarithmically.
   * - ``pml_sink_loss``
     - ``8.4e-3``
     - Separate loss threshold for setting a present mode's propagation transmission to zero.
   * - ``pml_mode_sink``
     - ``True``
     - Enables the sink in direct S-matrix propagation. Disable it for ``calc_Tmatrix()``.
   * - ``guided_rcond``
     - ``1e-5``
     - Relative singular-value cutoff for guided-heavy interface directions.
   * - ``pml_rcond``
     - ``1e-3``
     - Stronger cutoff for other interface directions.
   * - ``guided_fraction_threshold``
     - ``0.8``
     - Minimum guided weight of an interface singular direction for ``guided_rcond``.
   * - ``feedback_pml_rcond``
     - ``1e-3``
     - Relative cutoff for PML-heavy directions in scattering feedback solves.
   * - ``feedback_pml_fraction_threshold``
     - ``0.8``
     - Minimum PML weight of a feedback singular direction for that cutoff.
   * - ``absolute_rcond``
     - ``1e-8``
     - Global relative singular-value floor for interface and feedback solves.

These fractions describe *singular directions*, which can mix several modes;
they are not fractions of the mode count. See
:doc:`EME_stability_and_interpolation` for the equations, the distinction
between regularization and the PML sink, and guidance on changing defaults.

Set ``neff_interpolation=True`` on ``EME`` to interpolate guided-mode
effective indices along one physically varying dataset parameter. This
changes propagation phases only; interface overlaps and stability settings
retain their dataset values. The option is disabled by default.

.. code-block:: python

   import em_simulation as sim

   dataset = sim.DataUpdater(
       "sample_datasets/Si_rectangular_single_waveguide", is_testmode=True
   )
   geometry = sim.LinearTaper(dataset, 1e-6, 2e-6, 10e-6)
   config = sim.EMEStabilityConfig()
   interpolated_eme = sim.EME(
       geometry, stability_config=config, neff_interpolation=True
   )
   result = sim.Runner(interpolated_eme).propagate([1, 0, 0, 0, 0])
   print(interpolated_eme.neff_interpolation_diagnostics)

The :doc:`EME_stability_and_interpolation` guide explains interpolation,
fallbacks, supported geometries, and a runnable example.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   DataUpdater
   Geometry
   Propagator
   EME_stability_and_interpolation
   SiN_curvature_interpolation_example
   Runner

If you use this framework or the associated datasets in your research, please cite:

   Song, J. & Sohn, Y.-I.
   *Ultra-fast and accurate multimode waveguide design based on a dataset-based eigenmode expansion method.*
   Opt. Express 33, 46815–46827 (2025).
   (https://doi.org/10.1364/OE.567425)
