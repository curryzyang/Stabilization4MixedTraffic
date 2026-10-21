from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, gzip
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic.edie_aggregation import _segment_parts

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'runs/tgsim_calibration'

def metrics(k,q,vf,w,kj):
    pred=np.minimum(vf*k,w*np.maximum(kj-k,0));e=pred-q
    return {'rmse_q':float(np.sqrt(np.mean(e*e))),'mae_q':float(np.mean(abs(e))),
        'r2':float(1-np.sum(e*e)/np.sum((q-q.mean())**2)),'mean_residual':float(e.mean()),
        'observed_peak_q':float(q.max()),'observations_above_capacity':int(np.sum(q>vf*w*kj/(vf+w)))}

def fit(k,q,vf,kj):
    f=least_squares(lambda z:np.minimum(vf*k,z[0]*np.maximum(kj-k,0))-q,[15],bounds=([1],[80]),loss='huber',f_scale=100)
    return float(f.x[0])

def plot_fd(states,params,output_dir):
    """Render the saved all-cell fit; no fitting."""
    k=states.k_veh_per_km_per_lane.to_numpy()
    vf=float(params['v_f_kmh']);w=float(params['w_kmh'])
    kj=float(params['k_j_veh_per_km_per_lane'])
    kc=float(params['k_c_veh_per_km_per_lane'])
    capacity=float(params['q_max_veh_per_h_per_lane'])
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
        'axes.titlesize':10,'legend.fontsize':8,'figure.dpi':150,
        'savefig.dpi':300,'axes.linewidth':.8,'xtick.direction':'out',
        'ytick.direction':'out','grid.color':'#D8D8D8',
        'grid.linewidth':.5,'grid.alpha':.55,'pdf.fonttype':42,'ps.fonttype':42})
    xx=np.linspace(1e-6,kj,400);yy=np.minimum(vf*xx,w*np.maximum(kj-xx,0))
    for kind,name,label,scatter_color,line_color in [
        ('q','calibrated_flow_density_fd','Flow (veh/h/lane)','#3B6EA8','#C87A3D'),
        ('v','calibrated_speed_density_fd','Speed (km/h)','#4B8F72','#7B6AAE')]:
        fig,ax=plt.subplots(figsize=(5.8,3.8))
        values=states.q_veh_per_h_per_lane if kind=='q' else states.v_kmh
        ax.scatter(k,values,s=8,alpha=.28,edgecolors='none',color=scatter_color,label='Observed states')
        ax.plot(xx,yy if kind=='q' else yy/xx,color=line_color,linewidth=2.2,label='Triangular FD')
        if kind=='q':
            ax.scatter([kc],[capacity],s=34,color='#B85C5C',zorder=3,label='Capacity')
        else:
            ax.axvline(kc,color='#4A4A4A',linewidth=1,linestyle='--')
        ax.set(xlabel='Density (veh/km/lane)',ylabel=label)
        ax.grid(True);ax.legend(frameon=False);fig.tight_layout()
        for ext in ['pdf','png']:fig.savefig(output_dir/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--vf-kmh',type=float,default=55*1.609344)
    ap.add_argument('--raw-data', type=Path, default=BASE/'data/tgsim/i294_l1_trajectories.csv.gz')
    ap.add_argument('--plots', action='store_true', help='Generate optional local figures')
    ap.add_argument('--plot-only',action='store_true',help='Redraw existing cells and fitted parameters without fitting or changing numerical results')
    ap.add_argument('--output-dir',default=str(OUT));args=ap.parse_args()
    globals()['OUT']=Path(args.output_dir)
    if args.plot_only:
        states=pd.read_csv(OUT/'aggregated_traffic_states.csv')
        params=pd.read_csv(OUT/'triangular_fd_parameters.csv').iloc[0]
        plot_fd(states,params,OUT)
        print(f'Redrew both FD plots from {len(states)} saved cells and saved parameters; no fitting performed.')
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing results: {OUT}')
    path=args.raw_data
    if not path.is_file():
        ap.error('TGSIM data are not included. Supply --raw-data /path/to/file.csv or file.csv.gz; see docs/TGSIM_DATA.md.')
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(path);raw['is_acc']=raw.acc.astype(str).str.strip().str.lower().eq('yes')
    census=raw.groupby(['run_index','id']).is_acc.max()
    report={'raw_rows':len(raw),'unique_trajectory_ids':len(census),'ACC_trajectory_ids':int(census.sum()),
        'ACC_trajectory_share_percent':100*float(census.mean()),'raw_sha256':hashlib.sha256(gzip.open(path, 'rb').read() if path.suffix=='.gz' else path.read_bytes()).hexdigest(),
        'reported_road_length_m':4800,'direction':'southbound (lanes -2,-3,-4,-5)',
        'fixed_vf_kmh':args.vf_kmh,'fixed_vf_basis':'55 mph normal Central Tri-State regulatory scenario; 2021 temporary work-zone signs not independently identified',
        'coverage_method':'time-varying tracked-vehicle envelope trimmed by 20 m; proxy for aerial coverage, not measured camera FOV',
        'lane_normalization':'fixed four lanes, including lanes with zero observed vehicles',
        'fit_parameters':'w only; vf and kj are fixed, not jointly calibrated'}
    raw.groupby('run_index').agg(tmin=('time','min'),tmax=('time','max'),xmin=('xloc_kf','min'),xmax=('xloc_kf','max')).to_csv(OUT/'run_extents.csv')
    d=raw[raw.lane_kf.isin([-2,-3,-4,-5])&raw.speed_kf.between(0,160/3.6)&raw.acceleration_kf.abs().le(8)].copy()
    d=d.sort_values(['run_index','id','time']);report['southbound_clean_rows']=len(d)
    lengths=d.groupby(['run_index','id']).length_smoothed.mean();lengths=lengths[lengths.between(2,30)]
    length=float(lengths.mean());kj=1000/(length+2)
    report.update(average_vehicle_length_m=length,standstill_gap_m=2.,fixed_kj_veh_km=kj)
    # Travel totals split exactly across time and space bin boundaries.
    g=d.groupby(['run_index','id']);d['next_t']=g.time.shift(-1);d['next_x']=g.xloc_kf.shift(-1);d['next_lane']=g.lane_kf.shift(-1)
    dt=d.next_t-d.time;dx=d.next_x-d.xloc_kf
    valid=dt.between(.001,.5)&d.next_lane.eq(d.lane_kf)&dx.ge(0)
    report['excluded_lane_crossing_segments']=int((d.next_lane.notna()&d.next_lane.ne(d.lane_kf)).sum())
    report['valid_segments']=int(valid.sum());travel={}
    for run,t0,t1,x0,x1 in d.loc[valid,['run_index','time','next_t','xloc_kf','next_x']].itertuples(index=False,name=None):
        for tb,xb,tt,dd in _segment_parts(t0,t1,x0,x1,15.,20.,0.):
            z=travel.setdefault((int(run),tb,xb),[0.,0.]);z[0]+=tt;z[1]+=dd
    # Per-frame envelope; require full proxy exposure of a 15 s x 20 m cell.
    area={};frame=d.groupby(['run_index','time']).xloc_kf.agg(['min','max']).reset_index()
    for run,f in frame.groupby('run_index'):
        a=f.to_numpy();times=a[:,1]
        for j in range(len(a)-1):
            t0,t1=times[j:j+2];step=t1-t0
            if not (.001<step<=.5): continue
            lo=max(a[j,2],a[j+1,2])+20;hi=min(a[j,3],a[j+1,3])-20
            if hi<=lo: continue
            cuts=[t0]+[z for z in np.arange((np.floor(t0/15)+1)*15,t1,15)]+[t1]
            for left,right in zip(cuts[:-1],cuts[1:]):
                tb=int(np.floor((left+right)/2/15))
                for xb in range(int(lo//20),int(hi//20)+1):
                    width=max(0,min(hi,(xb+1)*20)-max(lo,xb*20))
                    key=(int(run),tb,xb);area[key]=area.get(key,0)+width*(right-left)
    rows=[]
    for (run,tb,xb),(tt,dd) in travel.items():
        exposure=area.get((run,tb,xb),0)
        # Travel totals at the edges may include periods outside the conservative
        # envelope. Only full-exposure cells are used to avoid inflating density.
        if exposure<.999*15*20: continue
        k=1000*tt/(4*15*20);q=3600*dd/(4*15*20)
        if not (0<k<kj and q>=0): continue
        rows.append({'run_index':run,'time_bin':tb,'space_bin':xb,'k_veh_per_km_per_lane':k,
                     'q_veh_per_h_per_lane':q,'v_kmh':q/k,'coverage_proxy_fraction':exposure/(15*20)})
    states=pd.DataFrame(rows);states.to_csv(OUT/'aggregated_traffic_states.csv',index=False)
    print('FULL-COVERAGE PROXY CELLS',len(states),flush=True)
    # All valid cells now contribute residuals to the same bounded Huber fit.
    # The original temporal split is deliberately absent from fitting and plots.
    k=states.k_veh_per_km_per_lane.to_numpy();q=states.q_veh_per_h_per_lane.to_numpy();vf=args.vf_kmh
    w=fit(k,q,vf,kj);kc=w*kj/(vf+w);capacity=vf*kc
    report.update(retained_cells=len(states),fitted_cells=len(states),
        fitting_sample='all valid retained cells; no training/held-out/buffer split',
        fit_loss='Huber',fit_scale_veh_h_lane=100,fit_w_bounds_kmh=[1,80],
        evaluation_scope='in-sample descriptive fit; no independent validation',
        cell_duration_s=15,cell_length_m=20,lane_count=4)
    report['all_sample_metrics']=metrics(k,q,vf,w,kj)
    groups=[z.index.to_numpy() for _,z in states.groupby(['run_index','time_bin'])]
    rng=np.random.default_rng(20260930);boot=[]
    for _ in range(300):
        inds=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);z=states.loc[inds]
        boot.append(fit(z.k_veh_per_km_per_lane.to_numpy(),z.q_veh_per_h_per_lane.to_numpy(),vf,kj))
    pd.DataFrame({'replicate':np.arange(1,301),'w_kmh':boot}).to_csv(OUT/'bootstrap_wave_speeds.csv',index=False)
    report.update(bootstrap_seed=20260930,bootstrap_replicates=300,bootstrap_clusters=len(groups),
        bootstrap_sampling='all retained (run_index,time_bin) clusters with replacement')
    report['w_block_bootstrap_95_CI']=np.percentile(boot,[2.5,97.5]).tolist()
    lo,hi=report['w_block_bootstrap_95_CI'];report['qmax_95_CI']=[vf*z*kj/(vf+z) for z in [lo,hi]]
    ps=[]
    for v in [45*1.609344,55*1.609344]:
        ww=fit(k,q,v,kj);ps.append({'fixed_vf_kmh':v,'fitted_w_kmh':ww,
            'evaluation_scope':'all-sample in-sample',**metrics(k,q,v,ww,kj)})
    pd.DataFrame(ps).to_csv(OUT/'policy_speed_sensitivity.csv',index=False)
    row={'dataset':'I-294_L1_SB_fixed_vf_kj_all_cells','fit_status':'ok','v_f_kmh':vf,'w_kmh':w,
         'k_j_veh_per_km_per_lane':kj,'k_c_veh_per_km_per_lane':kc,'q_max_veh_per_h_per_lane':capacity,**report['all_sample_metrics']}
    pd.DataFrame([row]).to_csv(OUT/'triangular_fd_parameters.csv',index=False)
    report.update(w_kmh=w,kc_veh_km=kc,qmax_veh_h=capacity)
    states['predicted_q']=np.minimum(vf*states.k_veh_per_km_per_lane,w*np.maximum(kj-states.k_veh_per_km_per_lane,0))
    states['residual_q']=states.predicted_q-states.q_veh_per_h_per_lane
    states['used_for_fit']=True
    states.to_csv(OUT/'residuals.csv',index=False)
    high=states[states.q_veh_per_h_per_lane>=states.q_veh_per_h_per_lane.quantile(.9)]
    report['high_flow_residuals']=metrics(high.k_veh_per_km_per_lane.to_numpy(),high.q_veh_per_h_per_lane.to_numpy(),vf,w,kj)
    report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), BASE/'traffic/edie_aggregation.py']}
    if args.plots: plot_fd(states,row,OUT)
    (OUT/'calibration_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
