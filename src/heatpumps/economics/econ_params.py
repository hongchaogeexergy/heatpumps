
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class EconParams:
    # Cost indices / correction
    ref_year: str = "2013"
    current_year: str = "2023"
    cepci_ref: float = 567.3   # CEPCI 2013
    cepci_cur: float = 797.9   # CEPCI 2023

    # Electricity price & operating hours
    electricity_price_cent_per_kWh: float = 40.0  # cent/kWh
    tau_h_per_year: float = 5500.0               # full-load hours per year

    # Financials
    i_eff: float = 0.08   # effective interest rate
    r_n: float = 0.02     # cost escalation
    n_years: int = 20     # lifetime in years

    # O&M
    omc_relative: float = 0.03   # fraction of TCI for O&M (component-wise default)

    # Scale PEC → TCI
    pec_to_tci_factor: float = 6.32

    # HEX U-values / costing parameters (W/m²K)
    k_evap: float = 1500.0
    k_cond: float = 3500.0
    k_inter: float = 2200.0
    k_trans: float = 60.0
    k_econ: float = 1500.0
    k_misc: float = 50.0

    # Flash tank costing (volume-based placeholder defaults)
    flash_residence_time_s: float = 10.0
    flash_ref_cost: float = 15000.0
    flash_ref_volume_m3: float = 1.0
    flash_cost_exponent: float = 0.6
    flash_pressure_ref_bar: float = 10.0
    flash_pressure_exponent: float = 0.0
    flash_rho_default: float = 1000.0

    # Optional: extra knobs
    overrides: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------
    # Convenience properties
    # ---------------------------
    @property
    def tau(self) -> float:
        """Alias for tau_h_per_year (full load hours per year)."""
        return self.tau_h_per_year

    @property
    def n(self) -> int:
        """Alias for n_years (plant lifetime)."""
        return self.n_years

    @property
    def cepci_factor(self) -> float:
        """CEPCI cost correction factor (cur/ref)."""
        try:
            return float(self.cepci_cur) / float(self.cepci_ref)
        except Exception:
            return 1.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EconParams":
        # robust creation, accepts both str/int for years
        d = dict(d or {})
        if "ref_year" in d:
            d["ref_year"] = str(d["ref_year"])
        if "current_year" in d:
            d["current_year"] = str(d["current_year"])
        return EconParams(**d)

# Handy defaults you can import
econ_defaults = EconParams().__dict__
