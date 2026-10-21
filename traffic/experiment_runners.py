"""Named, independent entry points around the existing numerical solvers."""
from pathlib import Path
from dataclasses import replace, asdict
import argparse
import time
import numpy as np
import pandas as pd
from . import experiment_utils as u
from .fundamental_diagram import MixedTrafficFD
from .simulation import SimulationConfig, build_system
from .boundary_control import run, InputLimiter
from .sensor_noise import Sensor, run_sensor_noise


def nominal_reference(args, cfg):
    directory = args.nominal_dir
    settings = __import__('json').loads((directory / 'configuration.json').read_text())
    settings = settings.get('simulation', settings)
    for key in ['n', 'sim_horizon_s', 'dt_s', 'tau_s', 'rs_veh_per_km']:
        if settings[key] != getattr(cfg, key):
            raise ValueError(f'Nominal reference differs in {key}; supply a matched --nominal-dir')
    return pd.read_csv(directory / 'nominal_metrics.csv')


def nominal_experiment(args, cfg, out):
    rows = []
    for p in args.penetrations:
        fd = MixedTrafficFD(p)
        results = u.run_case(fd, cfg)
        u.save_runs(p, results)
        rows.extend(u.row(fd, cfg, result, mode) for mode, result in results.items())
        print(f'Nominal experiment: P={p:g}', flush=True)
    u.percentages(pd.DataFrame(rows)).to_csv(out / 'nominal_metrics.csv', index=False)
    u.write_configuration(out, cfg, penetrations=args.penetrations)


def sensitivity_experiment(args, cfg, out):
    for kind, values, parameter in [('gap', [1., 2., 3.], 's_av_m'),
                                     ('headway', [.6, .9, 1.1], 'h_av_s')]:
        all_rows, summaries = [], []
        for value in values:
            fd = MixedTrafficFD(1., **{parameter: value})
            results = u.run_case(fd, cfg)
            u.save_runs(1, results, scenario=f'{kind}_{value:g}/P_1')
            group = pd.DataFrame([u.row(fd, cfg, result, mode) for mode, result in results.items()])
            all_rows.extend(group.to_dict('records'))
            ol = group[group['mode'].eq('OL')].iloc[0]
            summary = dict(parameter=value,
                Qmax_veh_h_lane=120*1000/(120*fd.hmix_s/3.6+fd.leff_m),
                v_star_kmh=ol.equilibrium_kmh)
            for mode in ['SF', 'OF']:
                row = group[group['mode'].eq(mode)].iloc[0]
                summary[f'{mode}_settling_s'] = row.settling_s
                summary[f'{mode}_delay_reduction_pct'] = 100*(1-row.J_delay/ol.J_delay)
            summaries.append(summary)
            print(f'{kind.capitalize()} sensitivity: {value:g}', flush=True)
        pd.DataFrame(all_rows).to_csv(out / f'{kind}_metrics.csv', index=False)
        pd.DataFrame(summaries).to_csv(out / f'{kind}_sensitivity.csv', index=False)
    u.write_configuration(out, cfg, penetration=1., gap_values_m=[1,2,3], headway_values_s=[.6,.9,1.1],
                          protocol='recompute equilibrium and controller/observer kernels for every scenario')


def pi_controller_experiment(args, cfg, out):
    nominal = nominal_reference(args, cfg)
    kp, ki = .1, -.002
    if not args.skip_tuning:
        tuning = []
        candidates = [(.5, .01)] + [(p, i) for p in [.1,.3,.5]
            for i in [-.005,-.002,-.0005,.001,.005,.01] if (p,i)!=(.5,.01)]
        for candidate_p, candidate_i in candidates:
            start = time.time()
            r = run(MixedTrafficFD(.6), replace(cfg, n=60), mode='PI',
                    kp=candidate_p, ki_per_s=candidate_i)
            positive = r['rho'].min()>0 and r['v'].min()>0
            horizon = cfg.sim_horizon_s
            score = (np.trapz(r['norm']**2,r['t'])/(horizon*r['norm'][0]**2)
                + .01*np.trapz((r['design_input']/r['system'].vs)**2,r['t'])/horizon) if positive else np.inf
            tuning.append(dict(kp=candidate_p, ki_per_s=candidate_i, score=score, physical_positive=positive,
                final_norm=r['norm'][-1], peak_norm=r['norm'].max(), rho_min_veh_km=r['rho'].min(),
                v_min_kmh=r['v'].min(), runtime_s=time.time()-start))
            u.archive(r, f'pi_tuning_kp{candidate_p:g}_ki{candidate_i:g}')
            print(f'PI gain selection: kp={candidate_p:g}, ki={candidate_i:g}', flush=True)
        tuning = pd.DataFrame(tuning)
        tuning.to_csv(out / 'pi_tuning.csv', index=False)
        eligible = tuning[tuning.physical_positive]
        best = eligible.loc[eligible.score.idxmin()]
        kp, ki = float(best.kp), float(best.ki_per_s)
        summary = []
        for gp, gi, role in [(.5,.01,'candidate validation'), (kp,ki,'selected coarse-grid gains')]:
            gr = tuning[(tuning.kp==gp)&(tuning.ki_per_s==gi)].iloc[0]
            summary.append(dict(role=role,P=.6,n=60,kp=gp,ki_per_s=gi,score=gr.score,
                physical_positive=gr.physical_positive,rho_min_veh_km=gr.rho_min_veh_km,
                v_min_kmh=gr.v_min_kmh,final_norm_sqrt_m=gr.final_norm))
        pd.DataFrame(summary).to_csv(out / 'pi_gain_selection_summary.csv', index=False)
    rows, runs = [], {}
    for p in args.penetrations:
        r = run(MixedTrafficFD(p), cfg, mode='PI', kp=kp, ki_per_s=ki)
        runs[p] = r
        u.archive(r, f'PI_P{p:g}')
        rows.append(u.summarize(r, 'nominal_PI'))
        print(f'PI controller experiment: P={p:g}', flush=True)
    table = pd.DataFrame(rows)
    for i, row in table.iterrows():
        ol = nominal[(nominal.P==row.P)&nominal['mode'].eq('OL')].iloc[0]
        for metric in u.METRICS:
            table.loc[i, metric+'_pct'] = 100*(row[metric]/ol[metric]-1)
    table.to_csv(out / 'pi_metrics.csv', index=False)
    if not args.skip_validation and .6 in runs:
        tight = run(MixedTrafficFD(.6), replace(cfg, rtol=1e-10, atol=1e-12),
                    mode='PI', kp=kp, ki_per_s=ki)
        u.archive(tight, 'validation_PI_P0.6_tight_RK')
        baseline = runs[.6]
        bm = table[table.P.eq(.6)].iloc[0]
        tm = u.summarize(tight, 'tight_RK_selected_PI')
        refinement = dict(kp=kp,ki_per_s=ki,P=.6,n=cfg.n,nominal_rtol=cfg.rtol,tight_rtol=1e-10,
            max_density_difference_veh_km=np.max(abs(baseline['rho']-tight['rho'])),
            max_speed_difference_kmh=np.max(abs(baseline['v']-tight['v'])),
            max_norm_difference_sqrt_m=np.max(abs(baseline['norm']-tight['norm'])),
            max_U_difference_ms=np.max(abs(baseline['design_input']-tight['design_input'])),
            nominal_final_norm=bm.final_norm,tight_final_norm=tm['final_norm'],
            max_relative_metric_difference=max(abs(tm[k]-bm[k])/abs(bm[k]) for k in u.METRICS))
        pd.DataFrame([refinement]).to_csv(out / 'pi_selected_RK_verification.csv', index=False)
        assert refinement['max_relative_metric_difference'] < 1e-6
    u.write_configuration(out, cfg, penetrations=args.penetrations,
        pi=dict(kp=kp,ki_per_s=ki,eta0=0,tuning_P=.6,tuning_n=60,
                law='U=kp*delta_v(0)+ki*eta; dot eta=delta_v(0); Ub=dU'),
        tuning_repeated=not args.skip_tuning)


def actuator_experiment(args, cfg, out):
    nominal = nominal_reference(args, cfg)
    ol = nominal[nominal.P.eq(.6)&nominal['mode'].eq('OL')].iloc[0]
    cfg = replace(cfg, rtol=1e-10, atol=1e-12)
    rows = []
    for mode in ['SF', 'OF']:
        for kind in ['magnitude', 'magnitude_rate']:
            result = run(MixedTrafficFD(.6), cfg, mode=mode, limiter=InputLimiter(kind))
            u.archive(result, f'actuator_{mode}_{kind}')
            summary = u.summarize(result, kind)
            summary['delay_reduction_pct'] = 100*(1-summary['J_delay']/ol.J_delay)
            rows.append(summary)
            assert summary['max_amplitude_violation_ms'] < 1e-9
            assert summary['boundary_map_max_error'] < 1e-10
            print(f'Actuator experiment: {mode}, {kind}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'actuator_metrics.csv', index=False)
    u.write_configuration(out, cfg, penetration=.6, limiter=asdict(InputLimiter('magnitude_rate')))


def noise_experiment(args, cfg, out):
    rows = []
    for sigma in args.noise_levels:
        for seed in args.seeds[:1] if sigma==0 else args.seeds:
            sensor = Sensor(relative_sigma=sigma, seed=seed)
            r = run_sensor_noise(MixedTrafficFD(.6), cfg, sensor)
            name = f'noise_sigma{sigma:g}_seed{seed}'
            u.archive(r, name)
            rows.append(u.summarize_noise(r, 'noise', name))
            print(f'Noise experiment: sigma={sigma:g}, seed={seed}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'noise_metrics.csv', index=False)
    u.write_configuration(out, cfg, penetration=.6, noise_levels=args.noise_levels,
        seeds=args.seeds, noise_knot_interval_s=1., controller='OF', actuator='ideal')


def traffic_performance_experiment(args, cfg, out):
    nominal = pd.read_csv(args.nominal_dir / 'nominal_metrics.csv')
    pi = pd.read_csv(args.pi_dir / 'pi_metrics.csv')
    merged = pd.concat([nominal,pi],ignore_index=True).sort_values(['P','mode'])
    rows = []
    for p in sorted(merged.P.unique()):
        group = merged[merged.P.eq(p)]
        for mode in ['OL','SF','OF','PI']:
            r = group[group['mode'].eq(mode)].iloc[0]
            rows.append(dict(P=p,mode=mode,value_type='absolute' if mode=='OL' else 'percent',
                **{k:r[k] if mode=='OL' else r[k+'_pct'] for k in u.METRICS}))
    merged.to_csv(out / 'controller_metrics.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'traffic_performance.csv', index=False)
    ranges = []
    for p in sorted(merged.P.unique()):
        system = build_system(MixedTrafficFD(p), cfg)
        for mode in ['SF','OF','PI']:
            path = (args.pi_dir/f'trajectories/PI_P{p:g}/history.npz' if mode=='PI'
                    else args.nominal_dir/f'trajectories/P_{p:g}/{mode}.npz')
            r = u.read_npz(path)
            control = r['u_control'] if mode=='PI' else r['control']
            error = np.max(abs(control-system.input_scale*r['design_input'])) if mode=='PI' else np.nan
            ranges.append(dict(P=p,mode=mode,min_ms=control.min(),max_ms=control.max(),
                               peak_abs_ms=max(abs(control)),input_map_error_ms=error))
    pd.DataFrame(ranges).to_csv(out / 'input_ranges_ms.csv', index=False)
    print('Traffic-performance tables generated.', flush=True)


def main(experiment):
    parser = argparse.ArgumentParser(description=experiment.replace('_',' ').capitalize())
    parser.add_argument('--output-dir', type=Path, default=u.ROOT/'runs'/experiment)
    parser.add_argument('--n', type=int, default=200, help='Number of interior spatial nodes')
    parser.add_argument('--horizon', type=float, default=400., help='Simulation duration in seconds')
    parser.add_argument('--penetrations', type=float, nargs='+', default=u.PEN)
    parser.add_argument('--nominal-dir', type=Path, default=u.ROOT/'runs/nominal_experiment')
    parser.add_argument('--pi-dir', type=Path, default=u.ROOT/'runs/pi_controller_experiment')
    parser.add_argument('--skip-tuning', action='store_true', help='Use fixed recorded PI gains 0.1 and -0.002/s')
    parser.add_argument('--skip-validation', action='store_true', help='Skip the additional PI tolerance check')
    parser.add_argument('--noise-levels', type=float, nargs='+', default=[0.,.005,.01,.02])
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261007])
    args = parser.parse_args()
    if args.n<2 or args.horizon<=0:
        parser.error('n must be at least 2 and horizon must be positive')
    cfg = SimulationConfig(n=args.n, sim_horizon_s=args.horizon)
    out = u.prepare_output(args.output_dir)
    globals()[experiment](args, cfg, out)
    print(f'Results: {out}', flush=True)
