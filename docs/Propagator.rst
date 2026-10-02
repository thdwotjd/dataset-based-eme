Propagator
==========

.. module:: em_simulation.propagator.eme
.. currentmodule:: em_simulation.propagator.eme

.. autoclass:: EME
   :no-members:

Stability configuration
-----------------------

.. autoclass:: em_simulation.propagator.stability.EMEStabilityConfig
   :no-members:

The loss thresholds, SVD cutoffs, feedback rules, and PML sink are explained
in :doc:`EME_stability_and_interpolation`. That guide also gives a runnable
example and the supported scope of phase interpolation.

Interpolation diagnostics
-------------------------

.. autoproperty:: EME.neff_interpolation_diagnostics

Methods
-------

.. automethod:: EME.calc_Tmatrix

.. automethod:: EME.calc_Smatrix
