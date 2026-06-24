# TorchFEA —— A PyTorch-based Finite Element Analysis Framework

**Developed by Zenan Song**

A research-oriented FEA framework supporting nonlinear materials, contact analysis, and dynamics (implicit/explicit). Built with a modular design: the **Assembly** layer manages geometry, materials, loads, and constraints, while the **Solver** layer provides static implicit, dynamic implicit (Newmark-β), and dynamic explicit (central difference) solvers.

## Features

- Clean Assembly–Solver separation with well-defined interfaces, easy to extend
- Sparse matrix assembly with Pardiso / Conjugate Gradient linear solvers
- Contact (self-contact / body-body contact), pressure, body force, and other load components
- Built on PyTorch — GPU acceleration and automatic differentiation for sensitivity analysis & optimization

## Quick Start

```bash
pip install torchfea
```

1. Install dependencies. Python ≥ 3.12, PyTorch (float64) recommended.
2. Run a minimal example (static + surface pressure):

```bash
python tests/pressure_test/test_benchmark.py
```

The script reads an INP file (e.g., `tests/pressure_test/C3D4Less.inp`), applies surface pressure and fixes bottom nodes, runs the static implicit solver, and saves the displacement vector.

3. More usage examples:

- Usage guide (INP → solve → export → visualization): `docs/usage.md`
- Architecture & data flow: `docs/structure.md`

## Directory Overview

- `src/torchfea/` — Core code (assembly, elements, loads, constraints, solvers, etc.)
- `tests/` — Test cases & benchmarks (elements, pressure, contact, dynamics, gradients/optimization)
- `docs/` — Architecture and usage documentation

## License

Research use preferred. For production or commercial use, please evaluate and ensure numerical and engineering robustness first.
