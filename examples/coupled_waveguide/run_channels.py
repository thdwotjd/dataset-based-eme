"""Run complex channel transmission using a saved 220 nm SOI dataset, without MODE.

From the checkout, run:
    python -m examples.coupled_waveguide.run_channels PATH_TO_CHANNEL_DATASET
Generate that dataset with examples.coupled_waveguide.channel_data_fde.
"""

import argparse
from pathlib import Path

import numpy as np

from em_simulation import ChannelRunner, DataUpdater, EME, Runner, StraightGapCoupler, profile_from_samples
from .channel_data_fde import example


def run(dataset_path):
    dataset = DataUpdater(dataset_path, is_testmode=True)
    gaps = dataset.coupled_settings.gap_grid
    model, _ = example([gap * 1e9 for gap in gaps])
    length = 10e-6
    for name, endpoints in (("uniform", (gaps[-1], gaps[-1])), ("varying", (gaps[0], gaps[-1]))):
        geometry = StraightGapCoupler(dataset, model, profile_from_samples([0, length], endpoints), verbose=False)
        channels = ChannelRunner(Runner(EME(geometry, neff_interpolation=False)),
                                 guide_labels={"wg0": "bus", "wg1": "ring"})
        inputs = {item["display_name"]: item["key"] for item in channels.available_channels("input", verbose=True)}
        bus, ring = inputs["bus_TE0"], inputs["ring_TE0"]
        print(f"{name}: {endpoints[0] * 1e9:g} -> {endpoints[1] * 1e9:g} nm")
        responses = []
        for label, excitation in (("bus TE0", {bus: 1}), ("ring TE0", {ring: 1}),
                                  ("coherent", {bus: 1, ring: 1j})):
            result = channels.run(excitation)
            responses.append(list(result.output_amplitudes.values()))
            mapping = result.diagnostics.channel_mapping
            values = ", ".join(f"{result.output_labels[key]}={amp:.4g}"
                               for key, amp in result.output_amplitudes.items())
            evaluation = mapping.evaluation
            power = evaluation.power
            print(f"  {label}: {values}")
            print(f"    P0/P1/P2/P3 (W): {power.P0:.5g}, {power.P1:.5g}, {power.P2:.5g}, {power.P3:.5g}")
            for side, residuals in (("input", evaluation.input_residuals), ("output", evaluation.output_residuals)):
                print(f"    {side} relative E/H residual: {residuals.E.relative_norm}, {residuals.H.relative_norm}")
        np.testing.assert_allclose(responses[2], np.asarray(responses[0]) + 1j * np.asarray(responses[1]),
                                   rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    run(parser.parse_args().dataset)
