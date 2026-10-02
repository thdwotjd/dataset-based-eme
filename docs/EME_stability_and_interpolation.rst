EME stability and one-axis phase interpolation
==============================================

This guide describes the optional settings of
:class:`~em_simulation.propagator.eme.EME`. The default
:class:`~em_simulation.propagator.stability.EMEStabilityConfig` was chosen
from a SiN convergence study at a wavelength of 1.55 micrometers. It is a
numerical policy, not a guarantee that the same thresholds are appropriate
for every dataset.

Where stability controls act
----------------------------

An EME simulation alternates modal propagation within a section and coupling
at an interface. Three different numerical operations need attention:

#. **Interface inversion.** Transmission blocks use inverses of matrices
   assembled from adjacent-section overlap integrals. Small singular values
   can amplify errors in poorly resolved or PML-like modal directions.
#. **Scattering feedback.** Combining adjacent S matrices with the Redheffer
   star product requires solving feedback systems. Near-singular directions
   can similarly amplify errors.
#. **Propagation through PML modes.** A strongly lossy mode can be removed
   from the direct S-matrix propagation block by setting its transmission to
   zero. This is the PML sink; it does not delete the mode from the interface
   overlap calculation.

The first two operations truncate selected singular directions. The sink is
a separate rule applied to propagation. Accordingly,
``pml_regularization_loss`` and ``pml_sink_loss`` need not be equal.

Modal loss and reliability weights
----------------------------------

The settings compare the dimensionless modal loss
:math:`\ell_m = |\operatorname{Im}(n_{\mathrm{eff},m})|`. A present mode
gets an interface reliability weight :math:`w_m`:

.. math::

   w_m =
   \begin{cases}
      1, & \ell_m \leq \ell_{\mathrm{guided}},\\
      1-\dfrac{\log \ell_m-\log \ell_{\mathrm{guided}}}
                 {\log \ell_{\mathrm{reg}}-\log \ell_{\mathrm{guided}}},
         & \ell_{\mathrm{guided}} < \ell_m < \ell_{\mathrm{reg}},\\
      0, & \ell_m \geq \ell_{\mathrm{reg}}.
   \end{cases}

Here :math:`\ell_{\mathrm{guided}}` is ``guided_loss`` and
:math:`\ell_{\mathrm{reg}}` is ``pml_regularization_loss``. An absent
mode also has weight zero. These weights express numerical confidence in
singular directions; they do not by themselves suppress a propagation
phase. They are also distinct from the geometry's radiation-mode mask used
by phase interpolation.

At 1.55 micrometers, an imaginary effective index of ``2.84e-5`` corresponds
to about **10 dB/cm** of power attenuation, and ``2.84e-3`` to about
**10 dB per 100 micrometers**, using
:math:`(20/\ln 10)(2\pi/\lambda)|\operatorname{Im}n_{\mathrm{eff}}|`
in dB per meter. The code uses the dimensionless effective-index values,
so a fixed dB-per-length interpretation changes with wavelength. Check the
dataset's imaginary-index sign convention when interpreting attenuation.

Interface SVD cutoffs
---------------------

For adjacent sections :math:`a` and :math:`b`, the direct S-matrix path
inverts the overlap combinations
:math:`M_{12}=O_{ab}+O_{ba}^{T}` and
:math:`M_{21}=O_{ba}+O_{ab}^{T}`. Each is decomposed as
:math:`M=U\Sigma V^{H}`. A singular direction can mix several modes. Its
guided fraction averages the reliability weights of its input and output
singular vectors:

.. math::

   f_k = \frac{1}{2}
   \left(\sum_j |V_{jk}|^2 w_j^{\mathrm{in}}+
         \sum_j |U_{jk}|^2 w_j^{\mathrm{out}}\right).

If :math:`f_k \geq` ``guided_fraction_threshold``, a singular value is
retained only when
:math:`\sigma_k/\sigma_{\max}\geq` ``guided_rcond``. Other directions
use the larger ``pml_rcond`` cutoff. Every direction must also satisfy
:math:`\sigma_k/\sigma_{\max}\geq` ``absolute_rcond``. Discarded singular
values contribute zero to the pseudoinverse rather than
:math:`1/\sigma_k`.

The defaults are ``guided_fraction_threshold=0.8``,
``guided_rcond=1e-5``, ``pml_rcond=1e-3``, and
``absolute_rcond=1e-8``. A larger cutoff drops more directions and may
reduce numerical amplification, but it can also remove physically relevant
coupling. A guided-heavy direction below ``absolute_rcond`` produces a
runtime warning.

Scattering feedback cutoffs
---------------------------

The Redheffer star product solves systems involving feedback matrices such
as :math:`I-A_{12}B_{21}` and :math:`I-B_{21}A_{12}`. Their singular
vectors receive a PML fraction based on :math:`1-w_m` for modes that are
present. A direction is dropped when either:

* its PML fraction is at least
  ``feedback_pml_fraction_threshold=0.8`` **and** its normalized
  singular value is below ``feedback_pml_rcond=1e-3``; or
* its normalized singular value is below ``absolute_rcond=1e-8``.

Otherwise the code uses a direct solve. If that solve fails or has an
unacceptable residual, it falls back to least squares. The feedback
cutoffs are separate from the interface cutoffs because they operate on
different matrices. A guided-heavy direction removed by the absolute
floor produces a runtime warning.

PML sink
--------

With ``pml_mode_sink=True``, a **present** mode whose
:math:`|\operatorname{Im}n_{\mathrm{eff}}|` is at least
``pml_sink_loss=8.4e-3`` has its forward and backward transmission set
to zero in that section's direct S-matrix phase block. Other modes use
:math:`\exp(i\beta_m\Delta z)`. The sink acts after the interface
matrices are built; it does not alter modal overlap values.

The sink is restored after optional passivity or unitarity projections, so
``force_unitary=True`` does not undo a zero-transmission sink phase.
A zero-transmission phase block has no transfer-matrix inverse. Therefore,
disable the sink before calling ``calc_Tmatrix()``:

.. code-block:: python

   import em_simulation as sim

   transfer_config = sim.EMEStabilityConfig(pml_mode_sink=False)
   transfer_eme = sim.EME(geometry, stability_config=transfer_config)
   transfer_eme.calc_Tmatrix()

Choosing settings
-----------------

Keep the required ordering
``0 < guided_loss < pml_regularization_loss <= pml_sink_loss`` and
``0 < guided_rcond <= pml_rcond < 1``. The fraction thresholds must lie
between zero and one; the other relative cutoffs must lie strictly between
zero and one. The constructor rejects invalid combinations.

Change one setting at a time and compare modal transmission, reflection,
and convergence as the dataset grid is refined. In particular, varying
``pml_sink_loss`` changes which phase transmissions are forced to zero;
varying ``pml_regularization_loss`` changes which interface and feedback
singular directions are considered reliable. These are different physical
and numerical interventions. See the
`example notebook <https://github.com/thdwotjd/dataset-based-eme/blob/main/examples/EME_stability_and_neff_interpolation.ipynb>`_
for a configuration and phase-interpolation comparison.

Optional one-axis effective-index interpolation
------------------------------------------------

``neff_interpolation=False`` is the default. With ``True``, the code
examines the continuous physical profile along the propagation coordinate
and requires at most one varying dataset parameter. For each sampled
position :math:`z` between neighboring dataset coordinates :math:`p_0`
and :math:`p_1`, it uses

.. math::

   t(z)=\frac{p(z)-p_0}{p_1-p_0},\qquad
   n_{\mathrm{eff},m}(z)
      \approx [1-t(z)]n_{0,m}+t(z)n_{1,m}.

The modal pair is matched through the stored overlap-based mode tracking,
not merely by raw mode number. Five physical samples per EME propagation
step are combined with Simpson quadrature to obtain an effective
:math:`\beta_m=(2\pi/\lambda)\langle n_{\mathrm{eff},m}\rangle`.
Thus the option changes propagation phases, including the phase accumulated
within a section. It does **not** interpolate mode fields or interface
overlaps, recompute an FDE solution, or modify stability weights and PML
sink decisions.

Only present guided modes according to the geometry's existing
radiation-mode mask are candidates. If a mode has no valid, unambiguous
guided pair at any sample in a step, that mode-step retains its original
dataset phase. The option reports this through
``EME.neff_interpolation_diagnostics``:

* ``enabled``: whether interpolation was requested;
* ``axis``: the varying dataset parameter, or ``None`` for a constant profile;
* ``guided_mode_steps``: present, non-radiation mode-steps considered;
* ``interpolated_mode_steps``: mode-steps whose phase was interpolated;
* ``tracking_conflicts``: conflicting tracked effective indices at dataset
  grid points.

For composite geometries, the property returns one diagnostics dictionary
per component. A geometry with two or more varying dataset parameters
raises ``ValueError`` in this version. A physical profile outside the
dataset grid also raises ``ValueError``; those cases do not silently
extrapolate.

For a linear width taper, run the following from the repository root.
Launch amplitudes must match the dataset's modal input convention:

.. code-block:: python

   import em_simulation as sim

   dataset = sim.DataUpdater(
       "sample_datasets/Si_rectangular_single_waveguide",
       is_testmode=True,
   )
   geometry = sim.LinearTaper(dataset, 1e-6, 2e-6, 10e-6)
   interpolated_eme = sim.EME(geometry, neff_interpolation=True)
   result = sim.Runner(interpolated_eme).propagate([1, 0, 0, 0, 0])
   print(interpolated_eme.neff_interpolation_diagnostics)

See the `example notebook <https://github.com/thdwotjd/dataset-based-eme/blob/main/examples/EME_stability_and_neff_interpolation.ipynb>`_
for a comparison against the original phase calculation with the same
stability settings.

For a curvature-only SiN validation with both Partial Euler and Full Euler
bends, see :doc:`SiN_curvature_interpolation_example`. It compares two
sample dataset steps against archived FDTD output powers.
