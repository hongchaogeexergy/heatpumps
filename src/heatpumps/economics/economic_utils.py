
import logging
from exerpy import ExergoeconomicAnalysis
from heatpumps.economics.econ_params import EconParams
from heatpumps.economics.exerpy_costing import (
    build_costs,
    build_exergo_boundaries,
    build_Exe_Eco_Costs,
)

logger = logging.getLogger(__name__)


def run_full_economic_pipeline(hp, econ: EconParams):
    """
    Run PEC → TCI → OPEX → (optionally) exergoeconomic cost assignment.
    Uses build_costs(...) from exerpy_costing as the single cost source.
    """
    if not hasattr(hp, "ean") or hp.ean is None:
        raise RuntimeError("Heat pump has no ExergyAnalysis (hp.ean missing).")

    PEC, TCI, Z = build_costs(
        ean=hp.ean,
        hp=hp,
        k_evap=econ.k_evap,
        k_cond=econ.k_cond,
        k_inter=econ.k_inter,
        k_trans=econ.k_trans,
        k_econ=econ.k_econ,
        k_misc=econ.k_misc,
        flash_residence_time_s=econ.flash_residence_time_s,
        flash_ref_cost=econ.flash_ref_cost,
        flash_ref_volume_m3=econ.flash_ref_volume_m3,
        flash_cost_exponent=econ.flash_cost_exponent,
        flash_pressure_ref_bar=econ.flash_pressure_ref_bar,
        flash_pressure_exponent=econ.flash_pressure_exponent,
        flash_rho_default=econ.flash_rho_default,
        CEPCI_cur=econ.cepci_cur,
        CEPCI_ref=econ.cepci_ref,
        tci_factor=econ.pec_to_tci_factor,
        omc_rel=econ.omc_relative,
        i_eff=econ.i_eff,
        r_n=econ.r_n,
        n=econ.n_years,
        tau_h_per_year=econ.tau_h_per_year,
    )

    capex_equipment = dict(PEC)
    capex_breakdown = dict(TCI)
    opex_breakdown = dict(Z)

    capex_total = sum(capex_breakdown.values())
    opex_total = sum(opex_breakdown.values())

    ean = hp.ean
    fuel_inputs, internal_zero, product, loss = build_exergo_boundaries(ean, hp)
    boundaries = {
        "fuel": {"inputs": list(fuel_inputs), "outputs": []},
        "product": product,
        "loss": loss,
    }
    Exe_Eco_Costs = build_Exe_Eco_Costs(
        ean=ean,
        hp=hp,
        boundaries=boundaries,
        internal_zero_labels=internal_zero,
        elec_price_cent_kWh=float(econ.electricity_price_cent_per_kWh),
        Z_by_component_label=Z,
        set_product_and_loss_to_zero=True,
    )

    # Apply user overrides if provided
    for key, val in getattr(econ, "overrides", {}).items():
        Exe_Eco_Costs[key] = val

    # --- Exergoeconomic Analysis (skip if SimpleHeatExchanger present) ---
    exergoecon = {}
    has_simple_hex = any(
        c.__class__.__name__ == "SimpleHeatExchanger"
        for c in ean.components.values()
    )

    if has_simple_hex:
        logger.warning("Skipping ExergoeconomicAnalysis: SimpleHeatExchanger not supported by ExerPy yet.")
    else:
        try:
            exea = ExergoeconomicAnalysis(ean)
            exea.run(Exe_Eco_Costs=Exe_Eco_Costs, Tamb=ean.Tamb)
            res = exea.exergoeconomic_results()
            if isinstance(res, tuple) and len(res) == 4:
                df_comp, df_mat1, df_mat2, df_nonmat = res
                exergoecon = {
                    "df_comp": df_comp,
                    "df_mat1": df_mat1,
                    "df_mat2": df_mat2,
                    "df_nonmat": df_nonmat,
                }
            else:
                logger.warning("exergoeconomic_results did not return 4 DataFrames; skipping.")
        except Exception as e:
            logger.warning(f"[pipeline] exergoeconomic_results failed: {e}")
            exergoecon = {}

    return {
        "capex_equipment": capex_equipment,
        "capex_breakdown": capex_breakdown,
        "capex_total": capex_total,
        "opex_breakdown": opex_breakdown,
        "opex_total": opex_total,
        "totals": {"capex_total": capex_total, "opex_total": opex_total},
        "assumptions": {
            "pec_to_tci_factor": econ.pec_to_tci_factor,
            "omc_relative": econ.omc_relative,
            "i_eff": econ.i_eff,
            "r_n": econ.r_n,
            "n_years": econ.n_years,
            "cepci_ref": econ.cepci_ref,
            "cepci_cur": econ.cepci_cur,
            "k_evap": econ.k_evap,
            "k_cond": econ.k_cond,
            "k_inter": econ.k_inter,
            "k_trans": econ.k_trans,
            "k_econ": econ.k_econ,
            "k_misc": econ.k_misc,
        },
        "exergoeconomics": exergoecon,
    }
