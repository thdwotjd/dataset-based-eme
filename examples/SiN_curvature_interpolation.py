"""Run the 5 um SiN Euler-bend comparison without starting FDE or FDTD.

Run --step 1000 and --step 5000 in separate Python processes, then --plot-only.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


STEPS = (1000, 5000)
SHAPES = ("partial_euler", "full_euler")
RADII = tuple(range(40, 61, 2))
GROUPS = ("TE0", "TE1", "TE2+")
OUT = ROOT / "examples" / "results" / "sin_curvature_interpolation"
REF = ROOT / "examples" / "data" / "SiN_w5_curvature_fdtd_reference.csv"


def cumtrap(y, x):
    return np.r_[0., np.cumsum((y[:-1] + y[1:]) * np.diff(x) / 2)]


def euler_profile(shape, radius_um):
    """90-degree centerline; same construction used for the FDTD cases."""
    t = np.linspace(0., 1., 4001)
    ramp = 0.25 if shape == "partial_euler" else 0.5
    q = np.minimum(np.minimum(t / ramp, (1 - t) / ramp), 1).clip(0, 1)
    theta = (math.pi / 2) * cumtrap(q, t) / cumtrap(q, t)[-1]
    x = cumtrap(np.cos(theta), t)
    y = cumtrap(np.sin(theta), t)
    scale = radius_um * 1e-6 / x[-1]
    stretch = x[-1] / y[-1]
    metric = np.sqrt(np.cos(theta)**2 + stretch**2 * np.sin(theta)**2)
    s = scale * cumtrap(metric, t)
    k = stretch * math.pi / 2 * q / (cumtrap(q, t)[-1] * scale * metric**3)
    if k.max() >= 50000 or not np.isclose(np.trapezoid(k, s), math.pi / 2, atol=1e-6):
        raise ValueError("Bend profile falls outside the dataset or is not 90 degrees")
    return s, k


def te_modes(geometry, dataset, section):
    n = geometry.output_data["neff"].shape[1] // 2
    neff = np.asarray(geometry.output_data["neff"][section, :n])
    pol = np.asarray(geometry.output_data["TE_pol"][section, :n])
    loss = 8.686 * (2 * np.pi / geometry.wavelength) * np.abs(neff.imag) / 100
    guided = ((neff.real > dataset.data_info.get_cladding_index() + 1e-4)
              & (loss <= 100) & (pol >= 0.5))
    tracks = np.flatnonzero(guided)
    tracks = tracks[np.argsort(-neff.real[tracks])]
    return {f"TE{i}": int(track) for i, track in enumerate(tracks)}


def group_powers(power, modes):
    return {"TE0": float(power[modes["TE0"]]),
            "TE1": float(power[modes["TE1"]]),
            "TE2+": float(sum(power[track] for name, track in modes.items()
                              if int(name[2:]) >= 2))}


def run(step, radii=RADII):
    import em_simulation as sim
    dataset = sim.DataUpdater(str(ROOT / "sample_datasets" / f"w_5um_curv_{step:05d}pm"),
                              is_testmode=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"SiN_w5_curvature_step_{step:05d}.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = old.to_dict("records")
    done = set()
    if not old.empty:
        for (shape, radius), group in old.groupby(["shape", "effective_radius_um"]):
            if len(group) == 12 and group.output_power.notna().all():
                done.add((shape, int(radius)))
    for shape in SHAPES:
        for radius in radii:
            if (shape, radius) in done:
                continue
            print(f"{step}/m | {shape} | Reff={radius} um", flush=True)
            s, k = euler_profile(shape, radius)
            geometry = sim.SingleCustomBend(dataset, prop_len_list=s,
                width_list=np.full(len(s), 5e-6), curvature_list=k,
                input_angle=0, resolution=3000, limit_mode_number=0, verbose=False)
            geometry.calc_output_data()
            n = geometry.output_data["neff"].shape[1] // 2
            inputs = te_modes(geometry, dataset, 0)
            outputs = te_modes(geometry, dataset, -1)
            if any(mode not in inputs or mode not in outputs for mode in ("TE0", "TE1")):
                raise RuntimeError("Missing guided input or output TE mode")
            prefix = np.flatnonzero(~geometry.output_data["radiation_mode_mask"][0, :n])
            if not np.array_equal(prefix, np.arange(len(prefix))):
                raise RuntimeError("Runner input modes must be a contiguous guided prefix")
            for interpolate in (False, True):
                eme = sim.EME(geometry, stability_config=sim.EMEStabilityConfig(),
                              neff_interpolation=interpolate)
                runner = sim.Runner(eme)
                for input_mode in ("TE0", "TE1"):
                    amplitude = np.zeros(len(prefix), dtype=complex)
                    amplitude[inputs[input_mode]] = 1
                    power = np.abs(np.asarray(runner.propagate(amplitude))[-1, :n])**2
                    diagnostic = eme.neff_interpolation_diagnostics
                    for group, value in group_powers(power, outputs).items():
                        rows.append(dict(curvature_step_per_m=step, shape=shape,
                            effective_radius_um=radius, input_mode=input_mode,
                            output_group=group, neff_interpolation=interpolate,
                            output_power=value,
                            guided_mode_steps=diagnostic.get("guided_mode_steps", 0),
                            interpolated_mode_steps=diagnostic.get("interpolated_mode_steps", 0),
                            tracking_conflicts=diagnostic.get("tracking_conflicts", 0)))
            pd.DataFrame(rows).sort_values(["shape", "effective_radius_um", "input_mode",
                "output_group", "neff_interpolation"]).to_csv(path, index=False)
    return path


def plot():
    data = pd.concat([pd.read_csv(OUT / f"SiN_w5_curvature_step_{s:05d}.csv")
                      for s in STEPS], ignore_index=True)
    ref = pd.read_csv(REF)
    colors = {"TE0": "#1f77b4", "TE1": "#ff7f0e", "TE2+": "#2ca02c"}
    for shape in SHAPES:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        for row, step in enumerate(STEPS):
            for col, input_mode in enumerate(("TE0", "TE1")):
                ax = axes[row, col]
                for group in GROUPS:
                    curve = data[(data["shape"] == shape)
                                 & (data["curvature_step_per_m"] == step)
                                 & (data["input_mode"] == input_mode)
                                 & (data["output_group"] == group)]
                    for interpolate, style, label in ((False, ":", "existing"),
                                                      (True, "-", "interpolated")):
                        part = curve[curve["neff_interpolation"] == interpolate]
                        ax.plot(part["effective_radius_um"], part["output_power"],
                                color=colors[group], linestyle=style,
                                label=f"{group} {label}")
                    points = ref[(ref["shape"] == shape)
                                 & (ref["input_mode"] == input_mode)
                                 & (ref["output_group"] == group)]
                    ax.scatter(points["effective_radius_um"], points["fdtd_power"],
                               color=colors[group], marker="X", s=70, edgecolor="black",
                               linewidth=0.6, zorder=5, label=f"{group} FDTD")
                ax.set_title(f"Step {step:,}/m | input {input_mode}")
                ax.set_ylim(0, 1)
                ax.set_xlim(39, 61)
                ax.set_xticks([40, 45, 50, 55, 60])
                ax.grid(alpha=0.25)
                if row == 1:
                    ax.set_xlabel("Effective radius (um)")
                if col == 0:
                    ax.set_ylabel("Guided TE output power")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))
        fig.suptitle(f"5 um SiN | {shape.replace('_', ' ').title()} | phase interpolation")
        fig.tight_layout(rect=(0, 0, 0.82, 0.94))
        fig.savefig(OUT / f"{shape}_w5_curvature_interpolation.png", dpi=170)
        plt.close(fig)
    comparison = data.merge(ref, on=["shape", "effective_radius_um",
                                     "input_mode", "output_group"])
    comparison["absolute_error_pp"] = 100 * (comparison["output_power"]
                                              - comparison["fdtd_power"]).abs()
    summary = comparison.groupby(["shape", "curvature_step_per_m",
        "neff_interpolation"])["absolute_error_pp"].agg(["mean", "max", "count"])
    summary.to_csv(OUT / "fdtd_error_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=STEPS)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--radii", type=int, nargs="+", default=RADII)
    args = parser.parse_args()
    if args.plot_only:
        plot()
    elif args.step is not None:
        run(args.step, tuple(args.radii))
    else:
        parser.error("Specify --step 1000, --step 5000, or --plot-only")
