from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _infer_grid_spacings(x: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    if x.ndim != 1 or t.ndim != 1:
        raise ValueError("x and t must be one-dimensional arrays")
    if len(x) < 2 or len(t) < 2:
        raise ValueError("x and t need at least 2 points to evaluate indices")
    dx = float(np.mean(np.diff(x)))
    dt = float(np.mean(np.diff(t)))
    return dx, dt


def compute_traffic_performance_indices(rho_km, v_kmh, x, t, max_speed_mps):
    """All metrics use the SAME trapezoidal space/time domain in physical SI states.

    Fuel model is the manuscript's phenomenological model with L/s output.
    Discomfort uses a^2+(1 s)^2*(partial_t a)^2 for consistent dimensions.
    """
    rho=np.asarray(rho_km,dtype=float)/1000
    v=np.asarray(v_kmh,dtype=float)/3.6
    x,t=np.asarray(x),np.asarray(t)
    if rho.shape!=v.shape or rho.shape!=(len(t),len(x)):
        raise ValueError('Physical field/grid shapes do not agree')
    if rho.min()<=0 or v.min()<0:
        raise ValueError('Linearized trajectory left the positive physical-state domain')
    trap=getattr(np,'trapezoid',np.trapz)
    def integral(f): return float(trap(trap(f,x=x,axis=1),x=t,axis=0))
    vt=np.gradient(v,t,axis=0,edge_order=2)
    vx=np.gradient(v,x,axis=1,edge_order=2)
    a=vt+v*vx
    at=np.gradient(a,t,axis=0,edge_order=2)
    fuel=np.maximum(0,.025+24.5e-6*v+32.5e-9*v**3+125e-6*v*a)
    ttt=integral(rho);dist=integral(rho*v)
    total=integral(rho*fuel)
    return {'J_fuel':total,'fuel_L_per_veh_km':total/(dist/1000),
            'J_discomfort':integral(rho*(a*a+at*at)),
            'J_TTT':ttt,'J_delay':ttt-dist/max_speed_mps,
            'distance_veh_km':dist/1000,
            'accel_rms_mps2':np.sqrt(integral(rho*a*a)/ttt)}


def save_performance_tables(
    output_dir: Path,
    penetration_rates: Iterable[float],
    baseline_metrics: dict[str, float],
    metrics_by_p: dict[float, dict[str, float]],
    table_name: str,
) -> tuple[Path, Path]:
    """Legacy single-baseline exporter; unused by the matched revision pipeline."""

    rows = [
        {
            "penetration": "0 (pure HV, open-loop)",
            "P": 0.0,
            **{k: v for k, v in baseline_metrics.items()},
            **{f"{k}_pct": 0.0 for k in baseline_metrics},
        }
    ]
    for p in sorted(metrics_by_p):
        metrics = metrics_by_p[p]
        label = f"{p:g}"
        if p == 0:
            label = "0 (pure HV, closed-loop)"
        row = {"penetration": label, "P": p}
        for key, value in metrics.items():
            base = baseline_metrics[key]
            row[key] = value
            row[f"{key}_pct"] = 0.0 if base == 0 else 100.0 * (value - base) / base
        rows.append(row)

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_abs = output_dir / f"{table_name}_performance.csv"
    csv_pct = output_dir / f"{table_name}_performance_pct.csv"

    df_abs = df[["penetration", "P", "J_fuel", "J_discomfort", "J_TTT", "J_delay"]]
    df_abs.to_csv(csv_abs, index=False)

    pct_cols = ["penetration", "P", "J_fuel_pct", "J_discomfort_pct", "J_TTT_pct", "J_delay_pct"]
    df_pct = df[pct_cols]
    df_pct.to_csv(csv_pct, index=False)

    tex = output_dir / f"{table_name}_performance.tex"
    tex.write_text(
        "\\begin{table*}[tbp]\n"
        "\\begin{center}\n"
        "\\caption{" + f"Traffic performance metrics under different CAV penetration rates: {table_name}" + "}\n"
        "\\begin{tabular}{r c c c c c c c c}\n"
        "\\hline\n"
        "Penetration rate $P_{CAV}$ & $J_{fuel}$ & $\\Delta J_{fuel}(\\%)$ & "
        "$J_{discomfort}$ & $\\Delta J_{discomfort}(\\%)$ & $J_{TTT}$ & $\\Delta J_{TTT}(\\%)$ "
        "& $J_{delay}$ & $\\Delta J_{delay}(\\%)$\\\\\n"
        "\\hline\n"
        + "\n".join(
            [
                (
                    f"\\text{{{row['penetration']}}} & {row['J_fuel']:.4e} & {row['J_fuel_pct']:+.2f}\\% & "
                    f"{row['J_discomfort']:.4e} & {row['J_discomfort_pct']:+.2f}\\% & "
                    f"{row['J_TTT']:.4e} & {row['J_TTT_pct']:+.2f}\\% & "
                    f"{row['J_delay']:.4e} & {row['J_delay_pct']:+.2f}\\% \\\\"
                )
                for row in rows
            ]
        )
        + "\\hline\n"
        "\\end{tabular}\n"
        "\\label{tab:performance_" + table_name + "}\n"
        "\\end{center}\n"
        "\\end{table*}\n",
        encoding="utf-8",
    )

    return csv_abs, tex
