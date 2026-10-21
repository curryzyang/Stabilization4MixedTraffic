from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'runs/full_experiment')
    parser.add_argument('--raw-data', type=Path,
                        help='Optional local TGSIM CSV or CSV.gz; enables calibration and calibrated control')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running experiments')
    args = parser.parse_args()
    out = args.output_root.resolve()
    raw = args.raw_data.resolve() if args.raw_data is not None else None
    if raw is not None and not raw.is_file():
        parser.error(f'TGSIM input does not exist: {raw}')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new output root: {out}')
    nominal = out/'nominal_experiment'
    pi = out/'pi_controller_experiment'
    jobs = [
        ('nominal_experiment', []),
        ('pi_controller_experiment', ['--nominal-dir', str(nominal)]),
        ('traffic_performance_experiment', ['--nominal-dir',str(nominal),'--pi-dir',str(pi)]),
        ('sensitivity_experiment', []),
        ('actuator_experiment', ['--nominal-dir',str(nominal)]),
        ('noise_experiment', []),
        ('numerical_validation', []),
    ]
    if raw is not None:
        jobs.extend([
            ('tgsim_calibration', ['--raw-data', str(raw)]),
            ('tgsim_control_experiment', ['--fd-params',
                str(out/'tgsim_calibration/triangular_fd_parameters.csv')]),
        ])
    else:
        print('No --raw-data supplied: running simulation experiments without TGSIM calibration.', flush=True)
    for name, extra in jobs:
        command = [sys.executable,str(ROOT/'experiments'/f'{name}.py'),
                   '--output-dir',str(out/name),*extra]
        print('Running:', ' '.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
