from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic import experiment_utils as suite
from traffic.fundamental_diagram import load_triangular_fd
from traffic.simulation import SimulationConfig, run_output_feedback

BASE = Path(__file__).resolve().parents[1]
DEFAULT_FD = BASE / 'runs/tgsim_calibration/triangular_fd_parameters.csv'
DEFAULT_OUT = BASE / 'runs/tgsim_control_experiment'


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_clean(value), indent=2, allow_nan=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fd-params', type=Path, default=DEFAULT_FD)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.fd_params.is_file():
        ap.error('Run tgsim_calibration.py first or supply --fd-params /path/to/triangular_fd_parameters.csv.')
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing results: {out}')
    out.mkdir(parents=True, exist_ok=True)
    suite.OUT = out
    cfg = SimulationConfig()
    fd = load_triangular_fd(args.fd_params)
    start = time.time()
    results = suite.run_case(fd, cfg)
    suite.save_runs(0, results, scenario='empirical')
    pair = results['OF']
    np.savez_compressed(out / 'trajectories/empirical/OF_full_state.npz',
                        raw_state=pair['raw_state'], rho_hat=pair['rho_hat'],
                        v_hat=pair['v_hat'], design_control=pair['design_control_of'])
    ol = results['OL']
    np.savez_compressed(out / 'trajectories/empirical/OL_full_state.npz',
                        raw_state=ol['raw_state'], bar_w=ol['bar_w'],
                        delta_v=ol['delta_v'], design_control=ol['design_control'])
    sys = pair['system']
    vstar = sys.vs
    mu = sys.linearization_term - vstar
    tf = cfg.road_length_m / vstar + cfg.road_length_m / mu
    facts = {
        'calibration_file': str(args.fd_params.resolve()),
        'fd': asdict(fd), 'configuration': asdict(cfg),
        'initial_relative_perturbation': .1,
        'initial_density': 'rho_star*(1+0.1*sin(3*pi*x/L))',
        'initial_velocity': 'v_star*(1-0.1*sin(3*pi*x/L))',
        'observer_initial_perturbation': 0,
        'rho_star_veh_km': cfg.rs_veh_per_km,
        'v_star_kmh': vstar * 3.6, 'v_star_ms': vstar,
        'equilibrium_flow_veh_h': cfg.rs_veh_per_km * vstar * 3.6,
        'S_ms': sys.linearization_term,
        'Vprime_ms_per_veh_m': -sys.linearization_term / sys.rs,
        'mu_ms': mu, 'boundary_reflection_r_b': sys.q,
        'input_map_d': sys.input_scale,
        'tf_s': tf, 'of_continuum_extinction_bound_s': 2 * tf,
        'settling_tolerance_sqrt_m': suite.SETTLING_TOLERANCE,
        'physical_norm_definition': 'sqrt integral ((rho/rho_star-1)^2+(v/v_star-1)^2) dx',
        'input_units': 'U_b and U in m/s',
        'state_storage_units': 'rho in veh/km/lane; v in km/h',
        'scope': 'calibrated linearized-model simulation; no field closed-loop validation',
    }
    # Store the actual newly computed coefficients used by all three cases.
    pd.DataFrame({'x_m': sys.xspan, 'K1_L_x': sys.kvu1,
                  'K2_L_x': sys.kvv1, 'observer_p1': sys.p1,
                  'observer_p2': sys.p2,
                  'source_c': -sys.c_2}).to_csv(out / 'design_coefficients.csv', index=False)
    rows = []
    checks = []
    for mode, r in results.items():
        rho, velocity, norm, ub = suite.fields(r, mode)
        row = suite.row(fd, cfg, r, mode)
        row.pop('max_control_kmh')
        row.update(max_U_b_ms=float(abs(ub).max()), max_U_ms=float(abs(ub).max()/sys.input_scale))
        if mode=='OL':
            row['theoretical_bound_s']=None  # Closed-loop extinction does not apply to OL.
        row.update(final_norm_sqrt_m=float(norm[-1]), peak_norm_sqrt_m=float(norm.max()))
        rows.append(row)
        reconstructed_norm = np.sqrt(np.trapz(
            (rho / cfg.rs_veh_per_km - 1) ** 2 + (velocity / (3.6*vstar) - 1) ** 2,
            x=r['x'], axis=1))
        boundary_error = (velocity[:, -1]/3.6-vstar
                          + vstar/sys.rs*(rho[:, -1]/1000-sys.rs)-ub)
        checks.append({'mode': mode,
            'norm_max_absolute_error': float(abs(norm-reconstructed_norm).max()),
            'physical_outlet_identity_max_error_ms': float(abs(boundary_error).max()),
            'all_states_finite': bool(np.isfinite(rho).all() and np.isfinite(velocity).all()),
            'positive_physical_states': bool(rho.min()>0 and velocity.min()>0)})
    table = pd.DataFrame(rows)
    olrow = table.loc[table['mode'].eq('OL')].iloc[0]
    for key in ['J_fuel', 'fuel_L_per_veh_km', 'J_discomfort', 'J_TTT', 'J_delay']:
        table[key+'_matched_OL_change_pct'] = 100*(table[key]/olrow[key]-1)
    table.to_csv(out/'empirical_metrics.csv', index=False)
    facts['results'] = table.to_dict(orient='records')
    # One independent integration-tolerance check verifies the new dynamic scale.
    tight = run_output_feedback(fd, replace(cfg, rtol=1e-9, atol=1e-11), initial_perturbation=.1)
    validation=[]
    for mode, suffix in [('SF','sf'),('OF','of')]:
        nr, tr = pair['norm_'+suffix], tight['norm_'+suffix]
        validation.append({'mode':mode,
            'default_settling_s':suite.settling(pair['t'],nr),
            'tight_RK_settling_s':suite.settling(tight['t'],tr),
            'max_norm_difference_sqrt_m':float(abs(nr-tr).max()),
            'max_density_difference_veh_km':float(abs(pair['rho_'+suffix]-tight['rho_'+suffix]).max()),
            'max_speed_difference_kmh':float(abs(pair['v_'+suffix]-tight['v_'+suffix]).max())})
    pd.DataFrame(validation).to_csv(out/'tight_RK_verification.csv',index=False)
    source_paths=[Path(__file__),BASE/'traffic/experiment_utils.py',BASE/'traffic/simulation.py',
        BASE/'traffic/fundamental_diagram.py',BASE/'traffic/kernel_solver.py',BASE/'traffic/performance_indices.py',args.fd_params]
    facts['source_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    facts['wall_time_s']=time.time()-start
    write_json(out/'facts.json',facts)
    write_json(out/'numerical_checks.json',{'checks':checks,'tight_RK':validation})
    assert all(c['norm_max_absolute_error']<1e-10 and c['physical_outlet_identity_max_error_ms']<1e-10
               and c['all_states_finite'] and c['positive_physical_states'] for c in checks)
    assert all(c['max_speed_difference_kmh']<1e-3 and c['max_density_difference_veh_km']<1e-3 for c in validation)
    print(json.dumps(json_clean(facts),indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
