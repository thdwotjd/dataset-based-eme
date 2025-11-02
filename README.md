# Dataset-Based Eigenmode Expansion (EME) Framework for Integrated Photonics

A fast, modular, and dataset-based **Eigenmode Expansion (EME)** simulation framework for integrated photonics.  
This repository provides a reproducible workflow for analyzing **multimode photonic devices** using pre-computed datasets (effective indices and overlap integrals) generated from commercial FDE solvers such as **Lumerical MODE**.

> ⚠️ This project is under active development — APIs and directory structures may change as features mature.  
> Feedback and contributions are welcome!

---

## Overview

The dataset-based EME method accelerates optical simulations by separating **modal field calculation**, **overlap calculation**, and **propagation/coupling analysis**.  
Once the dataset is generated (e.g., with Lumerical MODE), the EME solver computes transmission, reflection, and modal evolution with high accuracy and minimal computational cost.

```markdown
Dataset generation (mode field & overlap calculation using FDE)
↓
Dataset-based EME Solver
↓
Transmission, Reflection, Mode Coupling Analysis
```

---
## Documentation

Full project documentation is available on Read the Docs:

<https://dataset-based-emedreme.readthedocs.io>
---

## Repository Structure
```markdown
dataset-based-eme/
├── config/                     # Configuration files (Lumerical API path, dataset naming conventions)
├── em_simulation/              # Core EME solver modules
├── examples/                   # Example scripts (start here!)
├── sample_datasets/            # Example datasets
│   └── Si_rectangular_single_waveguide/
│       ├── dataset_info.py
│       ├── neff.pkl
│       ├── TE_pol.pkl
│       ├── overlap.pkl
│       └── wg_crosssection.lms
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```
---

## Python Environment

- Python **3.9–3.11**
- Tested on **macOS** and **Windows**

---

## Core Dependencies

The following Python libraries are required to run the simulation and examples.

```bash
# Clone repository
git clone https://github.com/thdwotjd/dataset-based-eme.git
cd dataset-based-eme

# (Optional) Create virtual environment using conda
conda create -n venv python=3.11

# Install dependencies
pip install -r requirements.txt
```

---
##  Lumerical API Configuration

To enable Python–Lumerical communication, ensure your Lumerical installation’s Python API path is included in your environment variables.
Adjust the parameters in config/config.yaml by modifying ansys_path and ansys_api_path according to your system configuration.

---
## Quick Start

1. Clone and Install
```bash
git clone https://github.com/thdwotjd/dataset-based-eme.git
cd dataset-based-eme
pip install -r requirements.txt
```

2. Run the Example

Open and run the Jupyter notebook:
```bash
examples/Si_linear_taper_simul.ipynb
```

It includes:
- Dataset loading
- Transfer matrix calculation
- Plot generation and analysis

---
## Concept: Dataset-Based EME

Traditional EME repeatedly calls mode solvers, which is computationally expensive.
Here, all mode field distributions and overlap integrals are pre-computed and stored as datasets.

This separation enables:
- ⚡ Fast sweeping over geometry
- 🎯 Multimode coupling analysis between arbitrary waveguide sections
- 🔁 Rapid optimization and inverse design workflows

---
## Contributing

We welcome community feedback and improvements:
1. Fork the repository
2.	Create a feature branch
3.	Submit a pull request with a clear description

If you encounter bugs or have feature requests, please open an Issue on GitHub.

---
## Citation

If you use this framework or the associated datasets in your research, please cite the following paper:
> Song, J. & Sohn, Y.-I.
> Ultra-fast and accurate multimode waveguide design based on a dataset-based eigenmode expansion method.
> Opt. Express 33, 46815–46827 (2025).
> https://doi.org/10.1364/OE.567425

**Bibtex**
```bibtex
@article{10.1364/oe.567425,
  author  = {Song, Jaesung and Sohn, Young-Ik},
  title   = {Ultra-fast and accurate multimode waveguide design based on a dataset-based eigenmode expansion method},
  journal = {Optics Express},
  volume  = {33},
  number  = {22},
  pages   = {46815--46827},
  year    = {2025},
  doi     = {10.1364/OE.567425}
}
```
---
## License


This project is licensed under the MIT License 

---
## Contact

```markdown
Maintainer:
Jaesung Song (KAIST EE)
📧 thdwotjd98@kaist.ac.kr￼
```
