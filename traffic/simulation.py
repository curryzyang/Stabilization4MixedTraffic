"""Physical-domain simulation of the nominal LINEARIZED ARZ system.

Numerical state u is bar w, not density; v is the speed perturbation.
All physical reconstruction, boundary actuation and metrics use SI units.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp
from .kernel_solver import LinearizedFunctions

@dataclass
class SimulationConfig:
    n: int = 200
    n_g: int = 200
    road_length_m: float = 500.0
    sim_horizon_s: float = 400.0
    dt_s: float = 0.1  # saved output interval, NOT the RK45 internal step
    tau_s: float = 60.0
    rs_veh_per_km: float = 45.0
    ctrl_on: bool = True
    linearization_branch: str = 'congested'
    rtol: float = 1e-7
    atol: float = 1e-9
    cfl: float = 0.4
    density_amplitude_veh_per_km: float | None = None  # optional legacy override

@dataclass
class System:
    cfg: SimulationConfig
    fd: object
    vs: float
    rs: float
    linearization_term: float
    q: float
    xspan: np.ndarray
    xspan_t: np.ndarray
    tspan: np.ndarray
    delta: float
    int_arr: np.ndarray
    lam: np.ndarray
    mu: np.ndarray
    c_1: np.ndarray
    c_2: np.ndarray
    kvu1: np.ndarray
    kvv1: np.ndarray
    p1: np.ndarray | None = None
    p2: np.ndarray | None = None

    @property
    def scale(self):
        return np.exp(-self.xspan/(self.cfg.tau_s*self.vs))
    @property
    def boundary_reflection(self):
        return self.scale[-1]/self.q
    @property
    def input_scale(self):
        return (self.linearization_term-self.vs)/self.linearization_term


def build_system(fd, cfg, with_observer=False):
    rs=cfg.rs_veh_per_km/1000
    vs=fd.operating_speed_ms(rs)
    s=fd.linearization_term(rs)
    if not (vs>0 and s>vs):
        raise ValueError('Operating point must be strictly congested and heterodirectional')
    if abs(float(fd.speed(cfg.rs_veh_per_km))/3.6-vs)>1e-8:
        raise ValueError('Selected equilibrium is inconsistent with the diagram branch')
    x=np.linspace(0,cfg.road_length_m,cfg.n+2)
    dx=cfg.road_length_m/(cfg.n+1)
    t=np.arange(0,cfg.sim_horizon_s+cfg.dt_s/2,cfg.dt_s)
    if t[-1]!=cfg.sim_horizon_s: t=np.r_[t[t<cfg.sim_horizon_s],cfg.sim_horizon_s]
    weights=dx*np.r_[.5,np.ones(cfg.n),.5]
    q=(vs-s)/vs
    fun=LinearizedFunctions(vs,s,cfg.tau_s,q,cfg.road_length_m)
    c=1/(cfg.tau_s*s)
    return System(cfg,fd,vs,rs,s,q,x,x[1:-1],t,dx,weights,
        fun.lam(x),fun.mu(x),fun.c_1(x),fun.c_2(x),
        c*np.exp(-x/(cfg.tau_s*vs)),np.full_like(x,-c),
        np.full_like(x,(s-vs)*c) if with_observer else None,
        -vs*c*np.exp(-x/(cfg.tau_s*vs)) if with_observer else None)


def physical_to_transformed(sys, rho_m, speed_ms, x):
    dv=np.asarray(speed_ms)-sys.vs
    w=dv+(sys.linearization_term/sys.rs)*(np.asarray(rho_m)-sys.rs)
    return np.exp(np.asarray(x)/(sys.cfg.tau_s*sys.vs))*w,dv


def transformed_to_physical(sys, u, v, x):
    w=np.exp(-np.asarray(x)/(sys.cfg.tau_s*sys.vs))*np.asarray(u)
    rho=sys.rs+sys.rs/sys.linearization_term*(w-np.asarray(v))
    return rho,sys.vs+np.asarray(v)


def initial_uv(sys, perturbation=.1, velocity_amplitude_mps=None):
    shape=np.sin(3*np.pi*sys.xspan_t/sys.cfg.road_length_m)
    amplitude=perturbation*sys.vs if velocity_amplitude_mps is None else velocity_amplitude_mps
    density_amplitude=(perturbation*sys.rs if sys.cfg.density_amplitude_veh_per_km is None
                       else sys.cfg.density_amplitude_veh_per_km/1000)
    rho=sys.rs+density_amplitude*shape
    speed=sys.vs-amplitude*shape
    return physical_to_transformed(sys,rho,speed,sys.xspan_t)


def _augment(arr,n,components):
    core=arr.reshape((components,n)).T
    aug=np.empty((n+2,components));aug[1:-1]=core
    aug[0]=2*core[0]-core[1];aug[-1]=2*core[-1]-core[-2]
    return aug


def _prepare_state(sys,u,v):
    u,v=u.copy(),v.copy()
    u[0]=sys.q*v[0]
    if sys.cfg.ctrl_on:
        # Solve the trapezoidal boundary term, including v(L), algebraically.
        v[-1]=0
        integral=sys.int_arr@(sys.kvu1*u+sys.kvv1*v)
        v[-1]=integral/(1-sys.int_arr[-1]*sys.kvv1[-1])
    else:
        v[-1]=sys.boundary_reflection*u[-1]
    design=v[-1]-sys.boundary_reflection*u[-1]
    return u,v,float(design)


def _prepare_output(sys,up,vp,uh,vh):
    up,vp,uh,vh=[a.copy() for a in (up,vp,uh,vh)]
    up[0]=sys.q*vp[0];uh[0]=sys.q*vh[0]
    vh[-1]=0
    integral=sys.int_arr@(sys.kvu1*uh+sys.kvv1*vh)
    # Manuscript law uses estimated endpoint; observer boundary uses y=up(L).
    rhs=integral+sys.boundary_reflection*(up[-1]-uh[-1])
    vp[-1]=rhs/(1-sys.int_arr[-1]*sys.kvv1[-1])
    vh[-1]=vp[-1]
    design=vp[-1]-sys.boundary_reflection*up[-1]
    return up,vp,uh,vh,float(design)


def _derivatives_pair(sys,u,v):
    ux=np.r_[(u[1]-u[0])/sys.delta,(u[:-3]-4*u[1:-2]+3*u[2:-1])/(2*sys.delta)]
    vx=np.r_[(-3*v[1:-2]+4*v[2:-1]-v[3:])/(2*sys.delta),(v[-1]-v[-2])/sys.delta]
    return -sys.vs*ux,(sys.linearization_term-sys.vs)*vx+sys.c_2[1:-1]*u[1:-1]


def state_rhs(t,z,sys):
    aug=_augment(z[:2*sys.cfg.n],sys.cfg.n,2)
    u,v,_=_prepare_state(sys,aug[:,0],aug[:,1])
    return np.concatenate(_derivatives_pair(sys,u,v))


def output_rhs(t,z,sys):
    n=sys.cfg.n;aug=_augment(z[:6*n],n,6)
    us,vs,_=_prepare_state(sys,aug[:,0],aug[:,1])
    up,vp,uh,vh,_=_prepare_output(sys,*[aug[:,i] for i in range(2,6)])
    ust,vst=_derivatives_pair(sys,us,vs)
    upt,vpt=_derivatives_pair(sys,up,vp)
    uht,vht=_derivatives_pair(sys,uh,vh)
    innovation=up[-1]-uh[-1]
    return np.r_[ust,vst,upt,vpt,uht+sys.p1[1:-1]*innovation,vht+sys.p2[1:-1]*innovation]


def _solve(sys,rhs,z0):
    max_step=sys.cfg.cfl*sys.delta/max(sys.vs,sys.linearization_term-sys.vs)
    sol=solve_ivp(lambda t,z:rhs(t,z,sys),(0,sys.cfg.sim_horizon_s),z0,
                  method='RK45',rtol=sys.cfg.rtol,atol=sys.cfg.atol,max_step=max_step,dense_output=True)
    if not sol.success: raise RuntimeError(sol.message)
    history=sol.t.copy();z=sol.sol(sys.tspan)
    meta={**asdict(sys.cfg),'method':'RK45','internal_steps':len(history)-1,
          'max_step_s':max_step,'nfev':sol.nfev,'min_internal_step_s':float(np.diff(history).min()),
          'max_internal_step_s':float(np.diff(history).max()),'kernels':'exact physical-domain constant-coefficient kernels'}
    return z,history,meta


def _norm(sys,rho,speed):
    return np.sqrt(((rho/sys.rs-1)**2+(speed/sys.vs-1)**2)@sys.int_arr)


def run_state_feedback(fd,cfg,initial_perturbation=.1,initial_velocity_amplitude_mps=None):
    sys=build_system(fd,cfg)
    u0,v0=initial_uv(sys,initial_perturbation,initial_velocity_amplitude_mps)
    z,history,meta=_solve(sys,state_rhs,np.r_[u0,v0])
    us=[];vs=[];controls=[]
    for zz in z.T:
        a=_augment(zz,cfg.n,2);u,v,c=_prepare_state(sys,a[:,0],a[:,1])
        us.append(u);vs.append(v);controls.append(c)
    u,v=np.asarray(us),np.asarray(vs)
    rho,speed=transformed_to_physical(sys,u,v,sys.xspan)
    return {'system':sys,'t':sys.tspan,'x':sys.xspan,'rho':rho*1000,'v':speed*3.6,
            'u_control':np.asarray(controls)*sys.input_scale,'design_control':np.asarray(controls),
            'norm':_norm(sys,rho,speed),'raw_state':z,'bar_w':u,'delta_v':v,
            'internal_t':history,'metadata':meta}


def run_output_feedback(fd,cfg,initial_perturbation=.1,initial_velocity_amplitude_mps=None):
    sys=build_system(fd,cfg,True)
    u0,v0=initial_uv(sys,initial_perturbation,initial_velocity_amplitude_mps)
    z,history,meta=_solve(sys,output_rhs,np.r_[u0,v0,u0,v0,np.zeros(2*cfg.n)])
    arrays=[[] for _ in range(6)];cs=[];co=[]
    for zz in z.T:
        a=_augment(zz,cfg.n,6)
        us,vs,c=_prepare_state(sys,a[:,0],a[:,1])
        up,vp,uh,vh,d=_prepare_output(sys,*[a[:,i] for i in range(2,6)])
        for ls,val in zip(arrays,[us,vs,up,vp,uh,vh]): ls.append(val)
        cs.append(c);co.append(d)
    us,vs,up,vp,uh,vh=[np.asarray(a) for a in arrays]
    rs,ss=transformed_to_physical(sys,us,vs,sys.xspan)
    rp,sp=transformed_to_physical(sys,up,vp,sys.xspan)
    rh,sh=transformed_to_physical(sys,uh,vh,sys.xspan)
    err=np.sqrt((((rp-rh)/sys.rs)**2+((sp-sh)/sys.vs)**2)@sys.int_arr)
    return {'system':sys,'t':sys.tspan,'x':sys.xspan,'rho_sf':rs*1000,'v_sf':ss*3.6,
      'rho_of':rp*1000,'v_of':sp*3.6,'rho_hat':rh*1000,'v_hat':sh*3.6,
      'u_control_sf':np.asarray(cs)*sys.input_scale,'u_control_of':np.asarray(co)*sys.input_scale,
      'design_control_of':np.asarray(co),'norm_sf':_norm(sys,rs,ss),'norm_of':_norm(sys,rp,sp),
      'observer_error':err,'raw_state':z,'internal_t':history,'metadata':meta}
