"""Reproduce the archived mesh, time-sampling, RK-tolerance, and tau checks."""
from pathlib import Path
from dataclasses import replace
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from traffic import experiment_utils as suite
from traffic.simulation import SimulationConfig
from traffic.fundamental_diagram import MixedTrafficFD


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=suite.ROOT/'runs/numerical_convergence_experiment')
    args=parser.parse_args();out=suite.prepare_output(args.output_dir)
    cfg=SimulationConfig();fd=MixedTrafficFD(.6);rows=[]
    for n in [100,200,400]:
        scenario=replace(cfg,n=n);results=suite.run_case(fd,scenario)
        suite.save_runs(.6,results,scenario=f'mesh_{n}/P_0.6')
        rows.extend({'n':n,**suite.row(fd,scenario,r,m)} for m,r in results.items())
        print('Mesh:',n,flush=True)
    pd.DataFrame(rows).to_csv(out/'mesh_convergence.csv',index=False)
    rows=[]
    for label,scenario in [('nominal',cfg),('tight_RK',replace(cfg,rtol=1e-9,atol=1e-11)),
                          ('half_output',replace(cfg,dt_s=.05)),
                          ('tau30',replace(cfg,tau_s=30)),('tau90',replace(cfg,tau_s=90))]:
        results=suite.run_case(fd,scenario)
        suite.save_runs(.6,results,scenario=f'validation_{label}/P_0.6')
        rows.extend({'test':label,**suite.row(fd,scenario,r,m)} for m,r in results.items())
        print('Numerical validation:',label,flush=True)
    pd.DataFrame(rows).to_csv(out/'time_and_tau_validation.csv',index=False)
    suite.write_configuration(out,cfg,penetration=.6,meshes=[100,200,400],adaptation_times_s=[30,60,90])


if __name__=='__main__':
    main()
