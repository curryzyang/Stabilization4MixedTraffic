from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TriangularFD:
    dataset: str
    v_free_kmh: float
    w_kmh: float
    k_jam: float
    k_critical: float
    q_capacity: float
    rmse_q: Optional[float] = None
    mae_q: Optional[float] = None
    r2: Optional[float] = None

    @property
    def v_free_ms(self) -> float:
        return self.v_free_kmh / 3.6

    @property
    def congestion_slope_ms_per_veh_m(self) -> float:
        # The congested branch is q = w * (k_j - k), v = q / k.
        # With k in veh/km and v in km/h, dv/dk = -w*k_j/k^2.
        # Converting k to veh/m and v to m/s multiplies this by 1000/3.6.
        return -(self.w_kmh * self.k_jam / 3.6) / 1000.0

    def flow(self, density_veh_per_km: np.ndarray | float) -> np.ndarray:
        k = np.asarray(density_veh_per_km, dtype=float)
        free = self.v_free_kmh * k
        congested = self.w_kmh * np.maximum(self.k_jam - k, 0.0)
        return np.minimum(free, congested)

    def speed(self, density_veh_per_km: np.ndarray | float) -> np.ndarray:
        k = np.asarray(density_veh_per_km, dtype=float)
        q = self.flow(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = q / k
        return np.where(k > 0, v, self.v_free_kmh)

    def local_speed_slope_ms(self, density_veh_per_m: float, branch: str = "auto") -> float:
        k_km = density_veh_per_m * 1000.0
        if branch == "free":
            return 0.0
        if branch == "congested" or (branch == "auto" and k_km >= self.k_critical):
            return self.congestion_slope_ms_per_veh_m / density_veh_per_m**2
        return 0.0

    def linearization_term(self, density_veh_per_m: float, branch: str = "congested") -> float:
        """Return -rho_s * dV/drho in m/s.

        The original Matlab model uses 1/(rho_s*hmix). For
        V=(1/rho-leff)/hmix, -rho_s*dV/drho equals that term. For the
        triangular FD we use the selected branch slope at the operating
        density. The default is the congested branch to avoid the zero-slope
        free-flow branch degeneracy at rho_s=45 veh/km for the provided data.
        """

        k_km = density_veh_per_m * 1000.0
        if branch == "free":
            dv_dk_kmh = 0.0
        else:
            dv_dk_kmh = -self.w_kmh * self.k_jam / max(k_km * k_km, 1e-12)
        dv_drho_ms = dv_dk_kmh * 1000.0 / 3.6
        return -density_veh_per_m * dv_drho_ms

    def operating_speed_ms(self, density_veh_per_m: float) -> float:
        return float(self.speed(density_veh_per_m * 1000.0)) / 3.6


@dataclass(frozen=True)
class MixedTrafficFD:
    penetration_rate: float
    v_free_kmh: float = 120.0
    s0_m: float = 2.0
    h_hv_s: float = 1.5
    h_av_s: float = 0.9
    vehicle_length_m: float = 5.0
    s_av_m: float = 2.0

    @property
    def dataset(self) -> str:
        return f"theoretical_P_{self.penetration_rate:.1f}"

    @property
    def hmix_s(self) -> float:
        p = self.penetration_rate
        return self.h_av_s * p + self.h_hv_s * (1.0 - p)

    @property
    def leff_m(self) -> float:
        p = self.penetration_rate
        return self.s0_m * (1.0 - p) + self.s_av_m * p + self.vehicle_length_m

    def operating_speed_ms(self, density_veh_per_m: float) -> float:
        return (1.0 / density_veh_per_m - self.leff_m) / self.hmix_s

    def linearization_term(self, density_veh_per_m: float, branch: str = "congested") -> float:
        return 1.0 / (density_veh_per_m * self.hmix_s)

    def flow(self, density_veh_per_km: np.ndarray | float) -> np.ndarray:
        k = np.asarray(density_veh_per_km, dtype=float)
        h_hours = self.hmix_s / 3600.0
        leff_km = self.leff_m / 1000.0
        rc = 1.0 / (self.v_free_kmh * h_hours + leff_km)
        free = self.v_free_kmh * k
        congested = (1.0 - k * leff_km) / h_hours
        return np.where(k <= rc, free, congested)

    def speed(self, density_veh_per_km: np.ndarray | float) -> np.ndarray:
        k = np.asarray(density_veh_per_km, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = self.flow(k) / k
        return np.where(k > 0, v, self.v_free_kmh)


def theoretical_fd_curves(penetration_rates: list[float] | np.ndarray, points: int = 1000) -> dict:
    curves = {}
    for p in penetration_rates:
        fd = MixedTrafficFD(float(p))
        rm = 1000.0 / fd.leff_m
        k = np.linspace(1e-6, rm, points)
        curves[float(p)] = {"density": k, "flow": fd.flow(k), "speed": fd.speed(k), "fd": fd}
    return curves


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No calibrated FD data was found. Expected one of: "
        + ", ".join(str(p) for p in paths)
    )


def load_triangular_fd(path: str | Path | None = None) -> TriangularFD:
    base = Path(__file__).resolve().parents[1]
    if path is None:
        path = _first_existing(
            [
                base / "data" / "triangular_fd_parameters.csv",
                base.parent / "Calibrated_Fundamental_Diagram" / "I-294_L1" / "triangular_fd_parameters.csv",
            ]
        )
    path = Path(path)
    params = pd.read_csv(path)
    if params.empty:
        raise ValueError(f"FD parameter file is empty: {path}")
    row = params.iloc[0]
    if str(row.get("fit_status", "ok")) != "ok":
        raise ValueError(f"FD fit status is not ok in {path}: {row.get('fit_status')}")
    return TriangularFD(
        dataset=str(row.get("dataset", path.parent.name)),
        v_free_kmh=float(row["v_f_kmh"]),
        w_kmh=float(row["w_kmh"]),
        k_jam=float(row["k_j_veh_per_km_per_lane"]),
        k_critical=float(row["k_c_veh_per_km_per_lane"]),
        q_capacity=float(row["q_max_veh_per_h_per_lane"]),
        rmse_q=_optional_float(row.get("rmse_q")),
        mae_q=_optional_float(row.get("mae_q")),
        r2=_optional_float(row.get("r2")),
    )


def _optional_float(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def load_traffic_states(path: str | Path | None = None) -> pd.DataFrame:
    base = Path(__file__).resolve().parents[1]
    if path is None:
        path = _first_existing(
            [
                base / "data" / "aggregated_traffic_states.csv",
                base.parent / "Calibrated_Fundamental_Diagram" / "I-294_L1" / "aggregated_traffic_states.csv",
            ]
        )
    states = pd.read_csv(path)
    required = {"k_veh_per_km_per_lane", "q_veh_per_h_per_lane", "v_kmh"}
    missing = required.difference(states.columns)
    if missing:
        raise ValueError(f"Traffic state file {path} is missing columns: {sorted(missing)}")
    return states


def export_fd_summary(fd: TriangularFD, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset": fd.dataset,
        "v_f_kmh": fd.v_free_kmh,
        "w_kmh": fd.w_kmh,
        "k_j_veh_per_km_per_lane": fd.k_jam,
        "k_c_veh_per_km_per_lane": fd.k_critical,
        "q_max_veh_per_h_per_lane": fd.q_capacity,
        "rmse_q": fd.rmse_q,
        "mae_q": fd.mae_q,
        "r2": fd.r2,
    }
    pd.DataFrame([row]).to_csv(output_dir / "calibrated_triangular_fd_parameters.csv", index=False)
    text = (
        f"Dataset: {fd.dataset}\n"
        "Triangular fundamental diagram\n\n"
        f"q(k) = min({fd.v_free_kmh:.6f} * k, {fd.w_kmh:.6f} * ({fd.k_jam:.6f} - k))\n"
        "v(k) = q(k) / k\n\n"
        "Units: k veh/km/lane, q veh/h/lane, v km/h\n"
        f"k_c = {fd.k_critical:.6f} veh/km/lane\n"
        f"q_max = {fd.q_capacity:.6f} veh/h/lane\n"
    )
    (output_dir / "calibrated_triangular_fd_equation.txt").write_text(text, encoding="utf-8")
