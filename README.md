# Mixed-Traffic Boundary Stabilization Experiments

Python code for **Boundary Stabilization of Mixed Traffic With Connected
Automated Vehicles**. It implements state-feedback (SF), output-feedback (OF),
and PI control experiments for a nominal linearized ARZ traffic model.

This repository contains source code and documentation only. Simulations
regenerate their numerical results locally. TGSIM calibration takes a local
trajectory file supplied separately.

## Layout

```text
traffic/                         Model, kernels, controllers, and numerical helpers
experiments/
  nominal_experiment.py          OL/SF/OF at six penetration levels
  pi_controller_experiment.py    PI gain search and fixed-gain comparison
  traffic_performance_experiment.py
  sensitivity_experiment.py      CAV gap and headway sensitivity
  actuator_experiment.py         Magnitude and magnitude-plus-rate constraints
  noise_experiment.py            Density/speed sensor noise
  tgsim_calibration.py           Local raw trajectories -> cells -> fitted diagram
  tgsim_refit.py                 Refit locally generated screened cells
  tgsim_control_experiment.py    OL/SF/OF with the calibrated diagram
  numerical_validation.py        Coordinate, kernel, boundary, and metric checks
  numerical_convergence_experiment.py
scripts/
  run_all.py                    Run experiments in dependency order
  plot_results.py               Plot numerical results after running experiments
requirements.txt               Reference dependency versions
```

## Environment

The reference environment is Python 3.9.13, NumPy 1.21.5, SciPy 1.9.1,
pandas 1.4.4, and Matplotlib 3.5.2. The computations run on CPU.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Run all commands below from
the repository root.

## Run simulations without external data

```bash
python scripts/run_all.py
```

This runs nominal, PI, traffic-performance, sensitivity, actuator, noise, and
analytic-check workflows. It creates `runs/full_experiment/`, with a separate
subdirectory per experiment. No raw data are needed for these simulations.
No figures are generated automatically.


## Run experiments individually

Individual entry points write to `runs/<experiment_name>/`. PI and actuator
comparisons first need the nominal run. The traffic-performance table needs
both nominal and PI runs.

```bash
python experiments/nominal_experiment.py
python experiments/pi_controller_experiment.py
python experiments/traffic_performance_experiment.py
python experiments/sensitivity_experiment.py
python experiments/actuator_experiment.py
python experiments/noise_experiment.py
python experiments/numerical_validation.py
python experiments/numerical_convergence_experiment.py
```

With custom output paths, pass
`--nominal-dir` and `--pi-dir` to the dependent scripts. PI and actuator
comparisons require nominal results with matching simulation settings.
The PI runner normally repeats the 18-candidate gain search and a tighter
integration-tolerance check; `--skip-tuning` uses the recorded gains directly,
and `--skip-validation` omits the extra tolerance check.

## TGSIM experiments with separately stored data

The input is the FHWA TGSIM I-294 L1 trajectory CSV. See
[data instructions](docs/TGSIM_DATA.md) for its official source and checksum.
The script accepts an uncompressed `.csv` or losslessly compressed `.csv.gz`.

Run calibration and then calibrated control:

```bash
python experiments/tgsim_calibration.py --raw-data /path/to/i294_l1_trajectories.csv.gz
python experiments/tgsim_control_experiment.py
```

Calibration writes `runs/tgsim_calibration/triangular_fd_parameters.csv`; the
control script reads that locally generated file by default. To repeat fitting
and bootstrap from the generated screened cells:

```bash
python experiments/tgsim_refit.py
```

To run the full simulation sequence together with TGSIM calibration and control:

```bash
python scripts/run_all.py --raw-data /path/to/i294_l1_trajectories.csv.gz
```

This places both calibration and simulation results under
`runs/full_experiment/`. A different data export can change the calibration
results; use the documented input checksum when matching the reference case.

## Optional plotting

After the combined simulation sequence:

```bash
python scripts/plot_results.py --results-dir runs/full_experiment
```

Figures are saved under `runs/figures/`. If the selected results directory
also contains TGSIM calibration and control outputs, their figures are included.
Otherwise, only the simulation figures are produced. When using individual
experiment entry points, set `--results-dir runs` instead.

## Numerical implementation

The numerical solvers are the same as in the complete code-and-data package.
Only default input paths and orchestration were adapted for a source-only
repository. Runs use adaptive RK45.

Independent mathematical and metric consistency checks can be run using:

```bash
python experiments/numerical_validation.py
```

## Contact

Feel free to email me for further discussion or collaboration at yihuai.zhang@cityu.edu.hk