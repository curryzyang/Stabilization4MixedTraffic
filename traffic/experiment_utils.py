from pathlib import Path
from dataclasses import replace, asdict
import json
import numpy as np
import pandas as pd
from .simulation import SimulationConfig, run_state_feedback, run_output_feedback
from .performance_indices import compute_traffic_performance_indices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs'
PEN = [0., .2, .4, .6, .8, 1.]
METRICS = ['J_fuel', 'fuel_L_per_veh_km', 'J_discomfort', 'J_TTT', 'J_delay']
SETTLING_TOLERANCE = 1e-4

def settling(t,norm,tolerance=SETTLING_TOLERANCE):
    above=np.flatnonzero(norm>=tolerance)
    if not len(above): return 0.
    j=above[-1]+1
    return float(t[j]) if j<len(t) else np.nan

def fields(r,mode):
    return (r['rho'],r['v'],r['norm'],r['u_control']) if mode in ['OL','SF'] else (r['rho_of'],r['v_of'],r['norm_of'],r['u_control_of'])

def row(fd,cfg,r,mode):
    rho,v,norm,control=fields(r,mode)
    m=compute_traffic_performance_indices(rho,v,r['x'],r['t'],fd.v_free_kmh/3.6)
    s=r['system'];tf=cfg.road_length_m/s.vs+cfg.road_length_m/(s.linearization_term-s.vs)
    return {'P':getattr(fd,'penetration_rate',np.nan),'mode':mode,'gap_CAV_m':getattr(fd,'s_av_m',np.nan),
        'headway_CAV_s':getattr(fd,'h_av_s',np.nan),'equilibrium_kmh':s.vs*3.6,
        'theoretical_bound_s':tf*(2 if mode=='OF' else 1),'settling_s':settling(r['t'],norm),
        'max_control_kmh':float(np.max(abs(control))*3.6),'rho_min':float(rho.min()),'v_min':float(v.min()),
        **m}

def run_case(fd,cfg):
    ol=run_state_feedback(fd,replace(cfg,ctrl_on=False),initial_perturbation=.1)
    of=run_output_feedback(fd,cfg,initial_perturbation=.1)
    sf={'system':of['system'],'t':of['t'],'x':of['x'],'rho':of['rho_sf'],'v':of['v_sf'],
        'norm':of['norm_sf'],'u_control':of['u_control_sf'],'metadata':of['metadata'],'internal_t':of['internal_t']}
    return {'OL':ol,'SF':sf,'OF':of}

def save_runs(p,results,scenario=None):
    folder=OUT/'trajectories'/(scenario if scenario else f'P_{p:g}');folder.mkdir(parents=True,exist_ok=True)
    for mode,r in results.items():
        rho,v,norm,control=fields(r,mode)
        np.savez_compressed(folder/f'{mode}.npz',t=r['t'],x=r['x'],rho=rho,v=v,norm=norm,
                            control=control,internal_t=r['internal_t'],observer_error=r.get('observer_error',np.array([])))
        (folder/f'{mode}_metadata.json').write_text(json.dumps(r['metadata'],indent=2))

def percentages(df):
    for i,r in df.iterrows():
        ol=df[(df.P==r.P)&(df['mode']=='OL')].iloc[0]
        for k in ['J_fuel','fuel_L_per_veh_km','J_discomfort','J_TTT','J_delay']:
            df.loc[i,k+'_pct']=100*(r[k]/ol[k]-1) if abs(ol[k])>1e-12 else np.nan
    return df

def archive(r,name):
    folder=OUT/'trajectories'/name
    folder.mkdir(parents=True,exist_ok=True)
    arrays={k:v for k,v in r.items() if k not in ['system','metadata']}
    np.savez_compressed(folder/'history.npz',**arrays)
    (folder/'metadata.json').write_text(json.dumps(r['metadata'],indent=2))

def summarize(r,label):
    s=r['system'];t=r['t'];act=r['metadata']['limiter'];u=r['design_input']
    m=compute_traffic_performance_indices(r['rho'],r['v'],r['x'],t,s.fd.v_free_kmh/3.6)
    phases=r['metadata']['limiter_phases']
    duration=sum(p['end_s']-p['start_s'] for p in phases if p['kind']=='ramp')
    rate=max(abs(np.diff(u)/np.diff(t)))
    bounds=act['kind']!='ideal'
    return dict(P=s.fd.penetration_rate,mode=r['metadata']['mode'],label=label,
                kp=r['metadata']['kp'],ki_per_s=r['metadata']['ki_per_s'],
                equilibrium_kmh=s.vs*3.6,input_scale_d=s.input_scale,
                settling_s=settling(t,r['norm']),final_norm=r['norm'][-1],
                peak_norm=r['norm'].max(),rho_min=r['rho'].min(),v_min=r['v'].min(),
                design_input_min_ms=u.min(),design_input_max_ms=u.max(),
                max_physical_input_kmh=max(abs(r['u_control']))*3.6,
                magnitude_active_pct=100*np.trapz(r['magnitude_active'],t)/t[-1],
                rate_active_pct=100*duration/t[-1] if act['kind']=='magnitude_rate' else np.nan,
                max_sampled_rate_ms_per_s=rate,
                max_rate_violation_ms_per_s=max(0.,rate-act['rate_ms_per_s']) if act['kind']=='magnitude_rate' else np.nan,
                max_amplitude_violation_ms=max(0.,act['lower_ms']-u.min(),u.max()-act['upper_ms']) if bounds else np.nan,
                boundary_map_max_error=r['boundary_map_error'].max(),
                observer_boundary_max_error=r['observer_boundary_error'].max(),
                nominal_feedback_identity_max_error=r['nominal_feedback_identity_error'].max(),**m)

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def summarize_noise(r,experiment,label):
    s=r['system'];t=r['t'];tail=t>=min(300., .75*t[-1])
    max_slope=float(np.max(np.abs(np.diff(r['applied_speed']*3.6)/np.diff(t))))
    act=r['metadata']['actuator'];lower,upper=r['metadata']['speed_bounds_ms']
    metric=compute_traffic_performance_indices(r['rho'],r['v'],r['x'],t,s.fd.v_free_kmh/3.6)
    record=dict(experiment=experiment,label=label,P=s.fd.penetration_rate,mode=r['metadata']['mode'],
        equilibrium_kmh=s.vs*3.6,settling_s=settling(t,r['norm']),
        final_norm=float(r['norm'][-1]),peak_norm=float(r['norm'].max()),
        tail_norm_rms=float(np.sqrt(np.trapz(r['norm'][tail]**2,t[tail])/(t[-1]-t[tail][0]))),
        observer_tail_rms=float(np.sqrt(np.trapz(r['observer_error'][tail]**2,t[tail])/(t[-1]-t[tail][0]))) if 'observer_error' in r else np.nan,
        peak_observer_error=float(r['observer_error'].max()) if 'observer_error' in r else np.nan,
        physical_input_total_variation_kmh=float(np.abs(np.diff(r['u_control']*3.6)).sum()),
        applied_speed_total_variation_kmh=float(np.abs(np.diff(r['applied_speed']*3.6)).sum()),
        applied_speed_min_kmh=float(r['applied_speed'].min()*3.6),
        applied_speed_max_kmh=float(r['applied_speed'].max()*3.6),
        magnitude_active_fraction=float(np.trapz(r['magnitude_active'],t)/(t[-1]-t[0])),
        rate_active_fraction=float(np.trapz(r['rate_active'],t)/(t[-1]-t[0])),
        max_actual_rate_kmh_per_s=float(np.nanmax(abs(r['applied_rate']))*3.6) if np.isfinite(r['applied_rate']).any() else np.nan,
        max_sampled_speed_slope_kmh_per_s=max_slope,
        sampled_rate_violation_kmh_per_s=max(0.,max_slope-act['rate_kmh_per_s']) if act['kind']=='magnitude_rate' else np.nan,
        speed_bound_violation_kmh=max(0.,lower-r['applied_speed'].min(),r['applied_speed'].max()-upper)*3.6 if act['kind'] in ('magnitude','magnitude_rate') else np.nan,
        boundary_map_max_error=float(np.max(r['boundary_map_error'])),
        target_residual_max_error=float(np.nanmax(r['target_residual_error'])) if np.isfinite(r['target_residual_error']).any() else np.nan,
        observer_boundary_max_error=float(np.nanmax(r['observer_boundary_error'])) if np.isfinite(r['observer_boundary_error']).any() else np.nan,
        rho_min=float(r['rho'].min()),v_min=float(r['v'].min()),**metric)
    record.update(r['metadata']['actuator']);record.update(r['metadata']['sensor'])
    return record

def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_clean(value), indent=2, allow_nan=False) + '\n')


def prepare_output(path):
    global OUT
    OUT = Path(path).resolve()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Choose a new output directory; results already exist: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def write_configuration(out, cfg, **settings):
    write_json(out / 'configuration.json', dict(simulation=asdict(cfg),
        initial_density='rho_star*(1+0.1*sin(3*pi*x/L))',
        initial_speed='v_star*(1-0.1*sin(3*pi*x/L))',
        observer_initial_perturbation=0, settling_threshold_sqrt_m=1e-4,
        output_interval_s=cfg.dt_s, integration='adaptive RK45 with CFL-limited maximum step',
        **settings))
