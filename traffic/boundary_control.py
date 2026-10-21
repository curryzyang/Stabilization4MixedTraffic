"""Design-input PI controller and ideal amplitude/slew limiter.

The linearized plant and exact kernels are reused without modifying production
sources. Slew limiting uses hybrid tracking/ramp phases: no tracking gain or
independent first-order time constant is present.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp
from .simulation import (build_system, initial_uv, _augment, _derivatives_pair,
                            _solve, _norm, transformed_to_physical)


@dataclass(frozen=True)
class InputLimiter:
    kind: str = 'ideal'
    lower_ms: float = -.3
    upper_ms: float = .3
    rate_ms_per_s: float = .03
    initial_input_ms: float = 0.


def run(fd, cfg, mode='SF', limiter=None, kp=.5, ki_per_s=.01):
    limiter = limiter or InputLimiter()
    if mode not in ('SF', 'OF', 'PI'):
        raise ValueError(mode)
    if limiter.kind not in ('ideal', 'magnitude', 'magnitude_rate'):
        raise ValueError(limiter.kind)
    if mode == 'PI' and limiter.kind != 'ideal':
        raise ValueError('PI is not part of the constrained-input experiments')
    if not limiter.lower_ms <= limiter.initial_input_ms <= limiter.upper_ms:
        raise ValueError('Infeasible limiter initial input')
    if limiter.rate_ms_per_s <= 0:
        raise ValueError('Invalid slew rate')
    sys = build_system(fd, cfg, mode == 'OF')
    n = cfg.n
    components = 4 if mode == 'OF' else 2
    size = components*n
    R, d, weights = sys.boundary_reflection, sys.input_scale, sys.int_arr
    feedthrough = weights[-1]*sys.kvv1[-1]
    u0, v0 = initial_uv(sys, .1)
    core = np.r_[u0, v0, np.zeros(2*n)] if mode == 'OF' else np.r_[u0, v0]
    if mode == 'PI':
        core = np.r_[core, 0.]

    def base(z):
        a = _augment(z[:size], n, components)
        u, v = a[:, 0].copy(), a[:, 1].copy()
        u[0] = sys.q*v[0]
        uh, vh = ((a[:, 2].copy(), a[:, 3].copy()) if mode == 'OF' else (None, None))
        if mode == 'OF':
            uh[0] = sys.q*vh[0]
        return u, v, uh, vh

    def free_command(z):
        u, v, uh, vh = base(z)
        if mode == 'PI':
            return float(kp*v[0] + ki_per_s*z[size])
        if mode == 'SF':
            v[-1] = 0.
            intercept = weights@(sys.kvu1*u+sys.kvv1*v) + (feedthrough-1)*R*u[-1]
        else:
            vh[-1] = 0.
            intercept = weights@(sys.kvu1*uh+sys.kvv1*vh) + feedthrough*R*u[-1] - R*uh[-1]
        return float(intercept/(1-feedthrough))

    def target(z):
        return float(np.clip(free_command(z), limiter.lower_ms, limiter.upper_ms))

    def traces(z, applied=None):
        u, v, uh, vh = base(z)
        free = free_command(z)
        if applied is None:
            applied = free if limiter.kind == 'ideal' else target(z)
        v[-1] = R*u[-1]+applied
        if mode == 'OF':
            vh[-1] = R*u[-1]+applied
            nominal = float(weights@(sys.kvu1*uh+sys.kvv1*vh)-R*uh[-1])
        elif mode == 'SF':
            nominal = float(weights@(sys.kvu1*u+sys.kvv1*v)-R*u[-1])
        else:
            nominal = free
        innovation = u[-1]-uh[-1] if mode == 'OF' else 0.
        return u, v, uh, vh, nominal, innovation, free

    def core_rhs(t, z, applied=None):
        u, v, uh, vh, nominal, innovation, free = traces(z, applied)
        answer = list(_derivatives_pair(sys, u, v))
        if mode == 'OF':
            uht, vht = _derivatives_pair(sys, uh, vh)
            answer += [uht+sys.p1[1:-1]*innovation, vht+sys.p2[1:-1]*innovation]
        if mode == 'PI':
            answer.append(np.array([v[0]]))
        return np.concatenate(answer)

    def target_rate(t, z):
        free = free_command(z)
        derivative = free_command(core_rhs(t, z, target(z)))
        if free < limiter.lower_ms or free > limiter.upper_ms:
            return 0.
        if free <= limiter.lower_ms and derivative < 0:
            return 0.
        if free >= limiter.upper_ms and derivative > 0:
            return 0.
        return derivative

    phases = []
    saved_rate = np.full(len(sys.tspan), np.nan)
    if limiter.kind != 'magnitude_rate':
        z, history, metadata = _solve(sys, lambda t, z, unused: core_rhs(t, z), core)
        phase_codes = np.zeros(len(sys.tspan), dtype=int)
    else:
        # Track the clipped nominal command exactly whenever its derivative is
        # feasible; otherwise ramp at +/- r until the target is caught again.
        # This is the ideal continuous slew limiter (a differential inclusion),
        # implemented by event-driven smooth phases rather than sign chatter.
        current = np.r_[core, limiter.initial_input_ms]
        current_t = 0.
        rate = limiter.rate_ms_per_s
        gap = target(current)-current[-1]
        state = ('ramp', float(np.sign(gap))) if abs(gap)>1e-10 else ('track', 0.)
        max_step = cfg.cfl*sys.delta/max(sys.vs, sys.linearization_term-sys.vs)
        histories, nfev = [], 0
        z = np.empty((len(current), len(sys.tspan)))
        phase_codes = np.zeros(len(sys.tspan), dtype=int)
        for phase_index in range(1000):
            kind, direction = state
            if kind == 'track':
                current[-1] = target(current)
                slope = target_rate(current_t, current)
                if abs(slope)>rate*(1+1e-9):
                    state = ('ramp', float(np.sign(slope)))
                    continue
                def rhs(t, zz):
                    return np.r_[core_rhs(t, zz, target(zz)), target_rate(t, zz)]
                def event(t, zz):
                    return rate-abs(target_rate(t, zz))
                event.direction = -1
            else:
                def rhs(t, zz):
                    return np.r_[core_rhs(t, zz, zz[-1]), direction*rate]
                def event(t, zz):
                    return direction*(target(zz)-zz[-1])
                event.direction = -1
            event.terminal = True
            sol = solve_ivp(rhs, (current_t, cfg.sim_horizon_s), current,
                            method='RK45', rtol=cfg.rtol, atol=cfg.atol,
                            max_step=max_step, dense_output=True, events=event)
            if not sol.success:
                raise RuntimeError(sol.message)
            end = float(sol.t[-1])
            if end <= current_t+1e-11:
                raise RuntimeError(f'Limiter phase did not advance at {end:g}: {state}')
            mask = (sys.tspan>=current_t-1e-10)&(sys.tspan<=end+1e-10)
            sampled = sol.sol(sys.tspan[mask])
            if kind == 'track':
                sampled[-1] = [target(zz) for zz in sampled.T]
            z[:, mask] = sampled
            phase_codes[mask] = 1 if kind == 'ramp' else 0
            saved_rate[mask] = (direction*rate if kind == 'ramp' else
                                [target_rate(t, zz) for t, zz in zip(sys.tspan[mask], sampled.T)])
            phases.append(dict(start_s=current_t, end_s=end, kind=kind, direction=direction))
            histories.append(sol.t)
            nfev += sol.nfev
            current, current_t = sol.y[:, -1], end
            if end >= cfg.sim_horizon_s-1e-9:
                break
            if kind == 'ramp':
                current[-1] = target(current)
                slope = target_rate(end, current)
                state = ('track', 0.) if abs(slope)<=rate*(1+1e-8) else ('ramp', float(np.sign(slope)))
            else:
                # In tracking phases the applied command is algebraic. The
                # redundant integrated U state can accumulate tiny drift;
                # reset it to the actual tracked trace before a ramp starts.
                current[-1] = target(current)
                # At a clipping corner the feasible target derivative can jump
                # from zero; use the underlying command derivative for its
                # direction, even if event interpolation lands just outside.
                raw_slope = free_command(core_rhs(end, current, target(current)))
                state = ('ramp', float(np.sign(raw_slope)))
        else:
            raise RuntimeError('Too many hybrid limiter phases')
        history = np.unique(np.concatenate(histories))
        metadata = {**asdict(cfg), 'method':'RK45 event-driven ideal slew limiter',
                    'internal_steps':len(history)-1, 'max_step_s':max_step, 'nfev':nfev,
                    'min_internal_step_s':float(np.diff(history).min()),
                    'max_internal_step_s':float(np.diff(history).max())}

    fields, records = [[] for _ in range(components)], []
    for t, zz in zip(sys.tspan, z.T):
        applied = float(zz[-1]) if limiter.kind == 'magnitude_rate' else None
        u, v, uh, vh, nominal, innovation, free = traces(zz, applied)
        applied = v[-1]-R*u[-1]
        for arr, value in zip(fields, [u, v, uh, vh]):
            arr.append(value)
        rho_L = sys.rs+sys.rs/sys.linearization_term*(sys.scale[-1]*u[-1]-v[-1])
        records.append(dict(design_input=applied, nominal_design_input=nominal,
                            free_design_input=free, applied_input=d*applied,
                            physical_boundary_speed_ms=sys.vs+v[-1],
                            magnitude_active=float(nominal<limiter.lower_ms-1e-10 or nominal>limiter.upper_ms+1e-10) if limiter.kind!='ideal' else 0.,
                            boundary_map_error=abs(v[-1]+sys.vs/sys.rs*(rho_L-sys.rs)-d*applied),
                            observer_boundary_error=abs(vh[-1]-(R*u[-1]+applied)) if mode=='OF' else 0.,
                            nominal_feedback_identity_error=abs(nominal-(free+feedthrough*(applied-free))) if mode!='PI' else 0.))
    u, v = map(np.asarray, fields[:2])
    rho, speed = transformed_to_physical(sys, u, v, sys.xspan)
    recorded = {k:np.asarray([r[k] for r in records]) for k in records[0]}
    metadata.update(mode=mode, limiter=asdict(limiter), fd_parameters=asdict(fd),
                    P=fd.penetration_rate, equilibrium_speed_ms=sys.vs,
                    equilibrium_density_veh_m=sys.rs, S_ms=sys.linearization_term,
                    input_scale_d=d, boundary_reflection_R=R,
                    kp=kp if mode=='PI' else None, ki_per_s=ki_per_s if mode=='PI' else None,
                    initial_density='rho_star*(1+.1*sin(3*pi*x/L))',
                    initial_speed='v_star*(1-.1*sin(3*pi*x/L))',
                    input_map='U_b=d*U; delta_v(L)=R*bar_w(L)+U',
                    limiter_phases=phases, limiter_algorithm='exact command tracking or saturated-slope ramp; event-driven; no tracking time constant')
    result = dict(system=sys, t=sys.tspan, x=sys.xspan, rho=rho*1000, v=speed*3.6,
                  norm=_norm(sys,rho,speed), u_control=recorded['applied_input'],
                  raw_state=z, bar_w=u, delta_v=v, internal_t=history,
                  metadata=metadata, applied_rate=saved_rate, rate_active=phase_codes, **recorded)
    if mode == 'PI':
        result['integral_error'] = z[size]
    if mode == 'OF':
        uh, vh = map(np.asarray,fields[2:])
        rh, sh = transformed_to_physical(sys,uh,vh,sys.xspan)
        result.update(rho_hat=rh*1000, v_hat=sh*3.6,
                      observer_error=np.sqrt((((rho-rh)/sys.rs)**2+((speed-sh)/sys.vs)**2)@weights))
    return result
