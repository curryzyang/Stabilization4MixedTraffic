"""Reproduce the all-1133-cell Huber fit from locally generated screened cells.

This repeats fitting/cluster bootstrap/plotting from locally generated Edie
cells. It cannot independently validate raw cleaning or coverage screening.
Use tgsim_calibration.py for the optional complete raw-data path.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from tgsim_calibration import fit, metrics, plot_fd

BASE = Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cells',type=Path,default=BASE/'runs/tgsim_calibration/aggregated_traffic_states.csv')
    ap.add_argument('--output-dir',type=Path,default=BASE/'runs/tgsim_refit')
    ap.add_argument('--plots', action='store_true', help='Generate optional local figures')
    args=ap.parse_args();out=args.output_dir
    if not args.cells.is_file():
        ap.error('Run tgsim_calibration.py first or supply --cells /path/to/aggregated_traffic_states.csv.')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing results: {out}')
    out.mkdir(parents=True,exist_ok=True)
    states=pd.read_csv(args.cells)
    required={'run_index','time_bin','space_bin','k_veh_per_km_per_lane','q_veh_per_h_per_lane','v_kmh','coverage_proxy_fraction'}
    if not required.issubset(states.columns):
        raise ValueError(f'Missing screened-cell columns: {required-set(states.columns)}')
    if len(states)!=1133:
        raise ValueError('This reference refit expects exactly 1133 screened cells')
    vf=55*1.609344
    mean_length=8.167128906249998
    kj=1000/(mean_length+2.)
    k=states.k_veh_per_km_per_lane.to_numpy();q=states.q_veh_per_h_per_lane.to_numpy()
    w=fit(k,q,vf,kj);kc=w*kj/(vf+w);capacity=vf*kc
    values=metrics(k,q,vf,w,kj)
    row={'dataset':'I-294_L1_SB_fixed_vf_kj_all_cells','fit_status':'ok','v_f_kmh':vf,'w_kmh':w,
        'k_j_veh_per_km_per_lane':kj,'k_c_veh_per_km_per_lane':kc,'q_max_veh_per_h_per_lane':capacity,**values}
    pd.DataFrame([row]).to_csv(out/'triangular_fd_parameters.csv',index=False)
    groups=[g.index.to_numpy() for _,g in states.groupby(['run_index','time_bin'])]
    rng=np.random.default_rng(20260930);boot=[]
    for _ in range(300):
        inds=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);g=states.loc[inds]
        boot.append(fit(g.k_veh_per_km_per_lane.to_numpy(),g.q_veh_per_h_per_lane.to_numpy(),vf,kj))
    pd.DataFrame({'replicate':np.arange(1,301),'w_kmh':boot}).to_csv(out/'bootstrap_wave_speeds.csv',index=False)
    ci=np.percentile(boot,[2.5,97.5]).tolist()
    report={'fixed_vf_kmh':vf,'fixed_kj_veh_km':kj,'average_vehicle_length_m':mean_length,
        'standstill_gap_m':2.,'w_kmh':w,'kc_veh_km':kc,'qmax_veh_h':capacity,
        'retained_cells':len(states),'fitted_cells':len(states),'all_sample_metrics':values,
        'fit_loss':'Huber','fit_scale_veh_h_lane':100,'fit_w_bounds_kmh':[1,80],
        'evaluation_scope':'in-sample fit; no independent or held-out validation',
        'cell_input':str(args.cells),'cells_sha256':hashlib.sha256(args.cells.read_bytes()).hexdigest(),
        'preprocessing_scope':'screened Edie cells supplied; raw cleaning and aggregation not rerun',
        'bootstrap_seed':20260930,'bootstrap_replicates':300,'bootstrap_clusters':len(groups),
        'w_block_bootstrap_95_CI':ci,'qmax_95_CI':[vf*z*kj/(vf+z) for z in ci]}
    (out/'calibration_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    states['predicted_q']=np.minimum(vf*k,w*np.maximum(kj-k,0))
    states['residual_q']=states.predicted_q-q;states['used_for_fit']=True
    states.to_csv(out/'residuals.csv',index=False)
    if args.plots: plot_fd(states,row,out)
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
