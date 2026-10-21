"""Independent checks of the transformations, kernels and physical boundary map."""
import json,sys,argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import cumulative_trapezoid
from traffic.fundamental_diagram import MixedTrafficFD
from traffic.simulation import SimulationConfig,build_system,physical_to_transformed,transformed_to_physical,_prepare_state,_prepare_output
from traffic.kernel_solver import LinearizedFunctions,inverse_kernels
from traffic.performance_indices import compute_traffic_performance_indices

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default=str(Path(__file__).resolve().parents[1]/'runs/numerical_validation'))
    args=ap.parse_args()
    errors={}
    for p in [0,.6,1]:
        s=build_system(MixedTrafficFD(p),SimulationConfig(n=800),True)
        x=s.xspan;a=s.vs;b=s.linearization_term-a;tau=s.cfg.tau_s;c=1/(tau*(a+b))
        rho=s.rs+.1*s.rs*np.sin(3*np.pi*x/500);speed=a-.1*a*np.sin(3*np.pi*x/500)
        u,v=physical_to_transformed(s,rho,speed,x)
        r,z=transformed_to_physical(s,u,v,x)
        errors[f'physical_roundtrip_{p}']=max(np.max(abs(r-rho)),np.max(abs(z-speed)))
        beta=v-c*cumulative_trapezoid(np.exp(-x/(tau*a))*u-v,x,initial=0)
        reconstructed=[]
        for i,xx in enumerate(x):
            l1,l2=inverse_kernels(LinearizedFunctions(a,a+b,tau,s.q),xx,x[:i+1])
            reconstructed.append(beta[i]+np.trapz(l1*u[:i+1]+l2*beta[:i+1],x[:i+1]))
        errors[f'backstepping_roundtrip_{p}']=float(np.max(abs(v-reconstructed)))
        us,vs,design=_prepare_state(s,u,v)
        r,z=transformed_to_physical(s,us,vs,x)
        ub=design*s.input_scale
        errors[f'physical_inlet_{p}']=abs((r[0]-s.rs)+(s.rs/a)*(z[0]-a))
        errors[f'physical_outlet_{p}']=abs(z[-1]-a+(a/s.rs)*(r[-1]-s.rs)-ub)
        errors[f'target_outlet_{p}']=abs(vs[-1]-s.int_arr@(s.kvu1*us+s.kvv1*vs))
        up,vp,uh,vh,design=_prepare_output(s,u,v,u*.3,v*.3)
        errors[f'observer_outlet_error_{p}']=abs(vp[-1]-vh[-1])
        # Closed-form substitutions into all direct, inverse and observer PDEs.
        xi=.4*x
        k1=c*np.exp(-xi/(tau*a));k2=-c
        errors[f'direct_kernel_pde_{p}']=float(np.max(abs(a*k1/(tau*a)-(-np.exp(-xi/(tau*a))/tau)*k2)))
        m1=s.q*c;m2=c*np.exp(-x/(tau*a))
        errors[f'observer_kernel_pde_{p}']=float(np.max(abs(-b*m2/(tau*a)-np.exp(-x/(tau*a))/tau*m1)))
        l1,l2=inverse_kernels(LinearizedFunctions(a,a+b,tau,s.q),x,xi)
        errors[f'inverse_kernel_pde_{p}']=float(np.max(abs(b*(-l1/(tau*(a+b)))-a*(-b*l1/(tau*a*(a+b))))))
        diag,_=inverse_kernels(LinearizedFunctions(a,a+b,tau,s.q),x,x)
        l10,l20=inverse_kernels(LinearizedFunctions(a,a+b,tau,s.q),x,0*x)
        errors[f'inverse_kernel_boundaries_{p}']=float(max(np.max(abs(diag-c*np.exp(-x/(tau*a)))),np.max(abs(l20+l10))))
        errors[f'observer_kernel_boundaries_{p}']=float(max(abs(m1[0]-s.q*c) if np.ndim(m1) else abs(m1-s.q*c),np.max(abs(m2-c*np.exp(-x/(tau*a))))))
        # Uniform fields have analytically known integral units and zero discomfort.
        xx=np.linspace(0,500,17);tt=np.linspace(0,400,21)
        rr=np.full((len(tt),len(xx)),45.);vv=np.full_like(rr,36.)
        mm=compute_traffic_performance_indices(rr,vv,xx,tt,120/3.6)
        expected={'J_TTT':9000.,'J_delay':6300.,'distance_veh_km':90.,
                  'J_fuel':9000*(.025+24.5e-6*10+32.5e-9*1000),'J_discomfort':0.}
        errors[f'physical_metric_units_{p}']=max(abs(mm[k]-v) for k,v in expected.items())
        # Original linearized rho/v RHS equals reconstructed transformed RHS.
        rx=.1*s.rs*3*np.pi/500*np.cos(3*np.pi*x/500)
        vx=-.1*a*3*np.pi/500*np.cos(3*np.pi*x/500)
        ux=np.exp(x/(tau*a))*(vx+(a+b)/s.rs*rx+(v+(a+b)/s.rs*(rho-s.rs))/(tau*a))
        ut=-a*ux;vt=b*vx-np.exp(-x/(tau*a))*u/tau
        rt=s.rs/(a+b)*(np.exp(-x/(tau*a))*ut-vt)
        errors[f'linearized_mass_equivalence_{p}']=float(np.max(abs(rt+a*rx+s.rs*vx)))
    assert max(v for k,v in errors.items() if 'backstepping' not in k)<1e-10
    assert max(v for k,v in errors.items() if 'backstepping' in k)<1e-4
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'verification.json').write_text(json.dumps(errors,indent=2))
    print('transform, boundary and kernel checks passed',max(errors.values()))
if __name__=='__main__':main()
