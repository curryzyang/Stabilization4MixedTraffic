"""Generate optional figures from the saved numerical data; never run simulations."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from traffic.fundamental_diagram import MixedTrafficFD
from traffic.experiment_utils import read_npz, PEN
from experiments.tgsim_calibration import plot_fd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'runs/full_experiment')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs/figures')
    args = parser.parse_args()
    data, out = args.results_dir, args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new figure directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'legend.fontsize':8})

    def save(fig, name):
        for extension in ['pdf','png']:
            fig.savefig(out/f'{name}.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)

    def line(curves, name, ylabel, key='norm', logarithmic=False):
        fig, ax = plt.subplots(figsize=(5.8,3.8))
        for label, r in curves.items():
            values = r[key]
            ax.plot(r['t'],np.maximum(values,1e-12) if logarithmic else values,label=label,lw=1.2)
        ax.set(xlabel='Time (s)',ylabel=ylabel)
        if logarithmic:
            ax.set_yscale('log')
            ax.axhline(1e-4,color='gray',ls=':',lw=.8)
        ax.grid(alpha=.2);ax.legend();save(fig,name)

    def surface(r, key, name, label):
        fig=plt.figure(figsize=(5.8,4.2));ax=fig.add_subplot(111,projection='3d')
        x,t=np.meshgrid(r['x'][::4],r['t'][::40])
        ax.plot_surface(x,t,r[key][::40,::4],cmap='viridis',linewidth=0,rasterized=True)
        ax.set(xlabel='Position (m)',ylabel='Time (s)',zlabel=label)
        ax.view_init(25,-130);save(fig,name)

    for quantity,name,label in [('flow','mixed_flow_density','Flow (veh/h/lane)'),
                                 ('speed','mixed_speed_density','Speed (km/h)')]:
        fig,ax=plt.subplots(figsize=(5.8,3.8));density=np.linspace(.001,1000/7,700)
        for p in PEN:
            ax.plot(density,getattr(MixedTrafficFD(p),quantity)(density),label=f'P={p:g}')
        ax.set(xlabel='Density (veh/km/lane)',ylabel=label)
        ax.grid(alpha=.2);ax.legend(ncol=2);save(fig,name)

    nominal={mode:read_npz(data/f'nominal_experiment/trajectories/P_0.6/{mode}.npz')
             for mode in ['OL','SF','OF']}
    nominal['PI']=read_npz(data/'pi_controller_experiment/trajectories/PI_P0.6/history.npz')
    for mode in ['SF','OF','PI']:
        for key,label in [('rho','Density (veh/km/lane)'),('v','Speed (km/h)')]:
            surface(nominal[mode],key,f'nominal_{mode.lower()}_{key}',label)
    line(nominal,'controller_comparison',r'$E_{plant}$ ($\sqrt{m}$)',logarithmic=True)
    controls={mode:dict(r,physical_input=r['u_control'] if mode=='PI' else r['control'])
              for mode,r in nominal.items() if mode!='OL'}
    line(controls,'physical_boundary_inputs',r'$U_b$ (m/s)',key='physical_input')

    limited={}
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.8),sharex=True,sharey=True)
    for i,mode in enumerate(['SF','OF']):
        for j,kind in enumerate(['magnitude','magnitude_rate']):
            r=read_npz(data/f'actuator_experiment/trajectories/actuator_{mode}_{kind}/history.npz')
            limited[mode,kind]=r;ax=axes[i,j]
            ax.plot(r['t'],r['nominal_design_input'],label='Nominal')
            ax.plot(r['t'],r['design_input'],label='Applied')
            for bound in [-.3,.3]:ax.axhline(bound,color='gray',ls=':',lw=.7)
            ax.set(title=f'{mode}: '+kind.replace('_',' + '),xlim=(0,300),
                   xlabel='Time (s)',ylabel='$U$ (m/s)')
            ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();save(fig,'actuator_commands')
    for mode in ['SF','OF']:
        line({kind.replace('_',' + '):limited[mode,kind] for kind in ['magnitude','magnitude_rate']},
             f'actuator_norm_{mode.lower()}',r'$E_{plant}$ ($\sqrt{m}$)',logarithmic=True)

    noise={f'{100*sigma:g}%':read_npz(data/f'noise_experiment/trajectories/noise_sigma{sigma:g}_seed20261007/history.npz')
           for sigma in [0.,.005,.01,.02]}
    line(noise,'noise_regulation',r'$E_{plant}$ ($\sqrt{m}$)',logarithmic=True)
    line(noise,'noise_observer',r'$E_{obs}$ ($\sqrt{m}$)',key='observer_error',logarithmic=True)

    calibration=data/'tgsim_calibration'
    if not (calibration/'triangular_fd_parameters.csv').is_file() or not (
            data/'tgsim_control_experiment/trajectories/empirical/OF.npz').is_file():
        print(f'Simulation figures generated in {out}; complete TGSIM results are not available in {data}.')
        return
    plot_fd(pd.read_csv(calibration/'aggregated_traffic_states.csv'),
            pd.read_csv(calibration/'triangular_fd_parameters.csv').iloc[0],out)
    calibrated={mode:read_npz(data/f'tgsim_control_experiment/trajectories/empirical/{mode}.npz')
                for mode in ['OL','SF','OF']}
    for mode in ['SF','OF']:
        for key,label in [('rho','Density (veh/km/lane)'),('v','Speed (km/h)')]:
            surface(calibrated[mode],key,f'tgsim_{mode.lower()}_{key}',label)
    line(calibrated,'tgsim_regulation',r'$E_{plant}$ ($\sqrt{m}$)',logarithmic=True)
    print(f'Figures generated in {out}; source numerical data were not modified.')


if __name__=='__main__':
    main()
