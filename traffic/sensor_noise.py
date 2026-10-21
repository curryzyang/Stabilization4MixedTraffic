"""Physical boundary-channel implementation used by the sensor-noise experiment.

These experiments retain the nominal linearized plant. No robustness theorem is
implied. All feedback is evaluated during the ODE solve, not clipped afterwards.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .simulation import (build_system, initial_uv, _augment, _derivatives_pair,
                         _solve, _norm, transformed_to_physical)


@dataclass(frozen=True)
class Actuator:
    kind: str = 'ideal'  # ideal / magnitude / tracking / magnitude_rate
    lower_fraction: float = .9
    upper_fraction: float = 1.1
    rate_kmh_per_s: float = .5
    tracking_gain_per_s: float = 1.


@dataclass(frozen=True)
class Sensor:
    relative_sigma: float = 0.
    seed: int = 20261007
    knot_interval_s: float = 1.
    bias_density_fraction: float = 0.
    bias_speed_fraction: float = 0.


def noise_record(sys, sensor):
    """Independent rho/v errors, fixed before RK evaluation; linear interpolation.

    Sigma describes Gaussian knot values, not the time RMS after interpolation.
    Both physical errors are multiplied by the exact physical-to-y map.
    """
    t=np.arange(0,sys.cfg.sim_horizon_s+sensor.knot_interval_s/2,
                sensor.knot_interval_s)
    if t[-1]<sys.cfg.sim_horizon_s: t=np.r_[t,sys.cfg.sim_horizon_s]
    rng=np.random.default_rng(sensor.seed)
    physical=rng.normal(size=(2,len(t)))
    drho=(sensor.relative_sigma*physical[0]+sensor.bias_density_fraction)*sys.rs
    dv=(sensor.relative_sigma*physical[1]+sensor.bias_speed_fraction)*sys.vs
    # Feasible start: sensors are initially exact; first interval ramps to error.
    drho[0]=0;dv[0]=0
    noise=np.exp(sys.cfg.road_length_m/(sys.cfg.tau_s*sys.vs))*(dv+sys.linearization_term/sys.rs*drho)
    return t,drho,dv,noise


def run_boundary_feedback(fd,cfg,mode='SF',actuator=None,sensor=None,
                          kp=.1,ki_per_s=-.001,initial_perturbation=.1):
    actuator=actuator or Actuator();sensor=sensor or Sensor()
    if mode not in ('SF','OF','PI'): raise ValueError(mode)
    if actuator.kind not in ('ideal','magnitude','tracking','magnitude_rate'): raise ValueError(actuator.kind)
    dynamic=actuator.kind in ('tracking','magnitude_rate')
    sys=build_system(fd,cfg,mode=='OF');n=cfg.n;components=4 if mode=='OF' else 2
    low=actuator.lower_fraction*sys.vs;high=actuator.upper_fraction*sys.vs
    if not low<=sys.vs<=high: raise ValueError('Equilibrium outside actuator range')
    rate=actuator.rate_kmh_per_s/3.6
    if rate<=0 or actuator.tracking_gain_per_s<=0: raise ValueError('Invalid actuator dynamics')
    nt,nrho,nv,ny=noise_record(sys,sensor)
    R=sys.boundary_reflection;d=sys.input_scale;weights=sys.int_arr
    u0,v0=initial_uv(sys,initial_perturbation)
    core=np.r_[u0,v0,np.zeros(2*n)] if mode=='OF' else np.r_[u0,v0]
    extra=[]
    if mode=='PI': extra.append(0.)
    if dynamic: extra.append(sys.vs)
    z0=np.r_[core,extra]

    def traces(t,z):
        a=_augment(z[:components*n],n,components)
        u,v=a[:,0].copy(),a[:,1].copy();u[0]=sys.q*v[0]
        uh,vh=(a[:,2].copy(),a[:,3].copy()) if mode=='OF' else (None,None)
        noise=float(np.interp(t,nt,ny));y=u[-1]+noise
        if mode=='OF': uh[0]=sys.q*vh[0]
        # First obtain the ideal algebraic boundary solution.
        if mode=='SF':
            vv=v.copy();vv[-1]=0
            ideal_delta=(weights@(sys.kvu1*u+sys.kvv1*vv))/(1-weights[-1]*sys.kvv1[-1])
        elif mode=='OF':
            vh[-1]=0
            ideal_hat=(weights@(sys.kvu1*uh+sys.kvv1*vh)+R*(y-uh[-1]))/(1-weights[-1]*sys.kvv1[-1])
            ideal_delta=ideal_hat-R*noise
        else:
            ideal_delta=kp*v[0]+ki_per_s*z[components*n]
        ideal_speed=sys.vs+ideal_delta
        if actuator.kind=='ideal': applied_speed=ideal_speed
        elif actuator.kind=='magnitude': applied_speed=float(np.clip(ideal_speed,low,high))
        else: applied_speed=float(z[-1])
        v[-1]=applied_speed-sys.vs
        Uapp=v[-1]-R*u[-1]
        if mode=='OF':
            # Same actual input and same measured y in boundary and innovation.
            vh[-1]=R*y+Uapp
            Unom=weights@(sys.kvu1*uh+sys.kvv1*vh)-R*uh[-1]
        elif mode=='SF': Unom=weights@(sys.kvu1*u+sys.kvv1*v)-R*u[-1]
        else: Unom=None
        rho_boundary=sys.rs+sys.rs/sys.linearization_term*(sys.scale[-1]*u[-1]-v[-1])
        nominal_speed=(sys.vs-sys.vs/sys.rs*(rho_boundary-sys.rs)+d*Unom
                       if mode!='PI' else ideal_speed)
        command=(float(nominal_speed) if actuator.kind=='tracking'
                 else float(np.clip(nominal_speed,low,high)))
        raw_rate=actuator.tracking_gain_per_s*(command-applied_speed)
        applied_rate=(float(np.clip(raw_rate,-rate,rate)) if actuator.kind=='magnitude_rate'
                      else float(raw_rate) if actuator.kind=='tracking' else np.nan)
        innovation=y-uh[-1] if mode=='OF' else 0.
        beta=v[-1]-weights@(sys.kvu1*u+sys.kvv1*v)
        target_error=abs(beta-(applied_speed-nominal_speed)/d) if mode=='SF' else np.nan
        # OF observer boundary differs by R*noise, never forced equal under noise.
        observer_bc_error=abs(vh[-1]-(R*y+Uapp)) if mode=='OF' else np.nan
        return u,v,uh,vh,dict(applied_speed=applied_speed,nominal_speed=nominal_speed,
            ideal_speed=ideal_speed,applied_input=d*Uapp,design_input=Uapp,
            nominal_design_input=Unom if Unom is not None else Uapp,
            applied_rate=applied_rate,magnitude_active=float(nominal_speed<low-1e-10 or nominal_speed>high+1e-10) if actuator.kind in ('magnitude','magnitude_rate') else 0.,
            rate_active=float(abs(raw_rate)>rate+1e-10) if actuator.kind=='magnitude_rate' else 0.,
            measurement=y,measurement_noise=noise,innovation=innovation,
            boundary_map_error=abs(applied_speed-sys.vs+sys.vs/sys.rs*(rho_boundary-sys.rs)-d*Uapp),
            target_residual_error=target_error,observer_boundary_error=observer_bc_error)

    def rhs(t,z,unused):
        u,v,uh,vh,record=traces(t,z)
        ut,vt=_derivatives_pair(sys,u,v);answer=[ut,vt]
        if mode=='OF':
            uht,vht=_derivatives_pair(sys,uh,vh)
            answer += [uht+sys.p1[1:-1]*record['innovation'],vht+sys.p2[1:-1]*record['innovation']]
        if mode=='PI': answer.append(np.array([v[0]]))
        if dynamic: answer.append(np.array([record['applied_rate']]))
        return np.concatenate(answer)

    z,history,metadata=_solve(sys,rhs,z0)
    arrays=[[] for _ in range(components)];records=[]
    for t,zz in zip(sys.tspan,z.T):
        u,v,uh,vh,record=traces(t,zz)
        for a,value in zip(arrays,[u,v,uh,vh]): a.append(value)
        records.append(record)
    u,v=map(np.asarray,arrays[:2]);rho,speed=transformed_to_physical(sys,u,v,sys.xspan)
    extra_record={k:np.asarray([r[k] for r in records]) for k in records[0]}
    metadata.update(mode=mode,actuator=asdict(actuator),sensor=asdict(sensor),
                    P=getattr(fd,'penetration_rate',None),fd_parameters=asdict(fd),
                    equilibrium_speed_ms=sys.vs,equilibrium_density_veh_m=sys.rs,
                    S_ms=sys.linearization_term,input_scale_d=d,
                    kp=kp,ki_per_s=ki_per_s,initial_perturbation=initial_perturbation,
                    actuator_initial_speed_ms=sys.vs if dynamic else None,
                    noise_model='independent physical Gaussian knot errors; continuous piecewise-linear interpolation',
                    speed_bounds_ms=[low,high])
    result=dict(system=sys,t=sys.tspan,x=sys.xspan,rho=rho*1000,v=speed*3.6,
                norm=_norm(sys,rho,speed),u_control=extra_record['applied_input'],
                internal_t=history,metadata=metadata,raw_state=z,bar_w=u,delta_v=v,
                noise_knots_t=nt,noise_knots_rho_veh_m=nrho,noise_knots_v_ms=nv,
                noise_knots_y=ny,**extra_record)
    if mode=='OF':
        uh,vh=map(np.asarray,arrays[2:]);rh,sh=transformed_to_physical(sys,uh,vh,sys.xspan)
        result.update(rho_hat=rh*1000,v_hat=sh*3.6,
            observer_error=np.sqrt((((rho-rh)/sys.rs)**2+((speed-sh)/sys.vs)**2)@weights))
    return result


def run_sensor_noise(fd, cfg, sensor):
    """Run OF with ideal actuation and physical density/speed sensor errors."""
    return run_boundary_feedback(fd, cfg, mode="OF", sensor=sensor)
