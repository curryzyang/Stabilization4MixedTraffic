import numpy as np

def _segment_parts(t0: float, t1: float, x0: float, x1: float, dt_cell: float, dx_cell: float, x_origin: float):
    tau = 0.0
    dt = t1 - t0
    dx = x1 - x0
    while tau < 1.0 - 1e-12:
        t = t0 + tau * dt
        x = x0 + tau * dx
        t_bin = int(np.floor(t / dt_cell))
        x_bin = int(np.floor((x - x_origin) / dx_cell))
        next_tau = 1.0

        t_boundary = (t_bin + 1) * dt_cell
        if dt > 0:
            tau_t = (t_boundary - t0) / dt
            if tau < tau_t < next_tau:
                next_tau = tau_t

        if dx > 0:
            x_boundary = x_origin + (x_bin + 1) * dx_cell
            tau_x = (x_boundary - x0) / dx
            if tau < tau_x < next_tau:
                next_tau = tau_x
        elif dx < 0:
            x_boundary = x_origin + x_bin * dx_cell
            tau_x = (x_boundary - x0) / dx
            if tau < tau_x < next_tau:
                next_tau = tau_x

        if next_tau <= tau:
            next_tau = min(1.0, tau + 1e-9)
        frac = next_tau - tau
        part_dt = frac * dt
        if part_dt > 0:
            yield t_bin, x_bin, part_dt, abs(frac * dx)
        tau = next_tau
