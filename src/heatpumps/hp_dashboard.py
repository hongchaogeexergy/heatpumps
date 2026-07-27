import base64
import json
import os
from copy import deepcopy
from xml.sax.saxutils import escape

import darkdetect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from heatpumps import variables as var
from CoolProp.CoolProp import PropsSI as PSI
from heatpumps.simulation import run_design, run_partload
from streamlit import session_state as ss
from exerpy import ExergyAnalysis
from exerpy import ExergoeconomicAnalysis, EconomicAnalysis
from heatpumps.economics.exerpy_costing import build_costs, run_exergoeconomic_from_hp

from heatpumps.models.topology_diagram import build_graph_from_hp


PEC_HEX_OPTIONS = {
    "ommen": {
        "label": "Ommen et al. (Default)",
        "formula": r"PEC_{\mathrm{HEX}} = 15526 \left(\frac{A}{42}\right)^{0.80}",
        "reference": "Ommen et al.",
        "literature_year": "2015",
        "data_year": "2013",
        "cepci_ref": 567.0,
        "currency": "EUR",
    },
    "dai_cascade": {
        "label": "Dai et al. (Kaskaden-WT)",
        "formula": r"PEC_{\mathrm{HEX}} = 383.5 \cdot A^{0.65}",
        "reference": "Dai et al.",
        "literature_year": "2022",
        "data_year": "2015-2016",
        "cepci_ref": 555.0,
        "currency": "USD",
    },
    "shamoushaki_shell": {
        "label": "Shamoushaki (Rohrbündel)",
        "formula": r"PEC_{\mathrm{HEX}} = \log_{10}(A) - 0.06395 \cdot A^2 + 947.2 \cdot A + 227.9",
        "reference": "Shamoushaki et al.",
        "literature_year": "2021",
        "data_year": "2020",
        "cepci_ref": 596.0,
        "currency": "USD",
    },
    "shamoushaki_plate": {
        "label": "Shamoushaki (Platte)",
        "formula": r"PEC_{\mathrm{HEX}} = \log_{10}(A) + 0.2581 \cdot A^2 + 891.7 \cdot A + 26050",
        "reference": "Shamoushaki et al.",
        "literature_year": "2021",
        "data_year": "2020",
        "cepci_ref": 596.0,
        "currency": "USD",
    },
}

PEC_COMP_OPTIONS = {
    "ommen": {
        "label": "Ommen et al. (Default)",
        "formula": r"PEC_{\mathrm{comp}} = 19850 \left(\frac{\dot{V}_{in}}{279.8}\right)^{0.73}",
        "reference": "Ommen et al.",
        "literature_year": "2015",
        "data_year": "2013",
        "cepci_ref": 567.0,
        "currency": "EUR",
    },
    "shamoushaki_centrifugal": {
        "label": "Shamoushaki (Zentrifugalverdichter)",
        "formula": r"PEC_{\mathrm{comp}} = \log_{10}(\dot{W}_{P}) + 0.03867 \cdot \dot{W}_{P}^{2} + 4446.7 \cdot \dot{W}_{P} + 137800",
        "reference": "Shamoushaki et al.",
        "literature_year": "2021",
        "data_year": "2020",
        "cepci_ref": 596.0,
        "currency": "USD",
    },
    "shamoushaki_reciprocating": {
        "label": "Shamoushaki (Hubkolbenverdichter)",
        "formula": r"PEC_{\mathrm{comp}} = \log_{10}(\dot{W}_{P}) + 0.04147 \cdot \dot{W}_{P}^{2} + 454.8 \cdot \dot{W}_{P} + 181000",
        "reference": "Shamoushaki et al.",
        "literature_year": "2021",
        "data_year": "2020",
        "cepci_ref": 596.0,
        "currency": "USD",
    },
}

PEC_PUMP_OPTION = {
    "label": "Shamoushaki et al.",
    "formula": r"PEC_{\mathrm{pump}} = \log_{10}(\dot{W}_{P}) - 0.03195 \cdot \dot{W}_{P}^{2} + 467.2 \cdot \dot{W}_{P} + 20480",
    "reference": "Shamoushaki et al.",
    "literature_year": "2021",
    "data_year": "2020",
    "cepci_ref": 596.0,
    "currency": "USD",
}

PEC_FLASH_OPTIONS = {
    "ommen": {
        "label": "Ommen et al. (Default)",
        "formula": r"PEC_{\mathrm{flash}} = 1444 \left(\frac{V_{\mathrm{flash}}}{0.089}\right)^{0.63}",
        "reference": "Ommen et al.",
        "literature_year": "2015",
        "data_year": "2013",
        "cepci_ref": 567.0,
        "currency": "EUR",
    },
    "dai": {
        "label": "Dai et al.",
        "formula": r"PEC_{\mathrm{flash}} = 280.3 \cdot \dot{m}^{0.67}",
        "reference": "Dai et al.",
        "literature_year": "2022",
        "data_year": "2015-2016",
        "cepci_ref": 555.0,
        "currency": "USD",
    },
}

COST_METHOD_OPTIONS = {
    "standard": "Dashboard-Standard",
    "repo_hthp": "Tomasinelli et al. (2026)",
}


def _selected_pec_summary(costcalcparams, CEPCI_cur):
    usd_to_eur = float(costcalcparams.get('usd_to_eur', 0.93))
    hex_model = PEC_HEX_OPTIONS[costcalcparams.get('hex_cost_model', 'ommen')]
    comp_model = PEC_COMP_OPTIONS[costcalcparams.get('compressor_cost_model', 'ommen')]
    flash_model = PEC_FLASH_OPTIONS[costcalcparams.get('flash_cost_model', 'ommen')]
    rows = [
        {
            "Komponente": "Wärmeübertrager",
            "Auswahl": hex_model["label"],
            "Referenz": hex_model["reference"],
            "Literaturjahr": hex_model["literature_year"],
            "Datenjahr": hex_model["data_year"],
            "Basiswährung": hex_model["currency"],
            "Währungsfaktor": f"{usd_to_eur:.3f}" if hex_model["currency"] == "USD" else "1.000",
            "CEPCI ref": f"{hex_model['cepci_ref']:.1f}",
            "CEPCI-Faktor": f"{CEPCI_cur / hex_model['cepci_ref']:.3f}",
        },
        {
            "Komponente": "Verdichter",
            "Auswahl": comp_model["label"],
            "Referenz": comp_model["reference"],
            "Literaturjahr": comp_model["literature_year"],
            "Datenjahr": comp_model["data_year"],
            "Basiswährung": comp_model["currency"],
            "Währungsfaktor": f"{usd_to_eur:.3f}" if comp_model["currency"] == "USD" else "1.000",
            "CEPCI ref": f"{comp_model['cepci_ref']:.1f}",
            "CEPCI-Faktor": f"{CEPCI_cur / comp_model['cepci_ref']:.3f}",
        },
        {
            "Komponente": "Pumpe",
            "Auswahl": PEC_PUMP_OPTION["label"],
            "Referenz": PEC_PUMP_OPTION["reference"],
            "Literaturjahr": PEC_PUMP_OPTION["literature_year"],
            "Datenjahr": PEC_PUMP_OPTION["data_year"],
            "Basiswährung": PEC_PUMP_OPTION["currency"],
            "Währungsfaktor": f"{usd_to_eur:.3f}",
            "CEPCI ref": f"{PEC_PUMP_OPTION['cepci_ref']:.1f}",
            "CEPCI-Faktor": f"{CEPCI_cur / PEC_PUMP_OPTION['cepci_ref']:.3f}",
        },
        {
            "Komponente": "Flashtank",
            "Auswahl": flash_model["label"],
            "Referenz": flash_model["reference"],
            "Literaturjahr": flash_model["literature_year"],
            "Datenjahr": flash_model["data_year"],
            "Basiswährung": flash_model["currency"],
            "Währungsfaktor": f"{usd_to_eur:.3f}" if flash_model["currency"] == "USD" else "1.000",
            "CEPCI ref": f"{flash_model['cepci_ref']:.1f}",
            "CEPCI-Faktor": f"{CEPCI_cur / flash_model['cepci_ref']:.3f}",
        },
    ]
    return pd.DataFrame(rows)


def build_kosmadakis_project_cost_df(hp, PEC, kosmadakis_params, elec_price_cent_kWh, tau_h_per_year):
    """Estimate project cost and savings using the Kosmadakis dashboard assumptions."""
    comp_by_label = {
        getattr(comp, "label", ""): comp
        for comp in getattr(hp, "comps", {}).values()
    }

    def _component_class(label):
        comp = comp_by_label.get(label)
        return comp.__class__.__name__ if comp is not None else ""

    def _is_hex_label(label):
        cls = _component_class(label)
        label_l = str(label).lower()
        return (
            cls in {"Condenser", "HeatExchanger", "SimpleHeatExchanger", "DropletSeparator"}
            or any(token in label_l for token in ("evaporator", "condenser", "heat exchanger", "economizer"))
        )

    def _is_comp_label(label):
        return _component_class(label) == "Compressor"

    def _is_flash_label(label):
        cls = _component_class(label)
        return cls == "Drum" or "flash" in str(label).lower()

    def _safe_float(value, default=0.0):
        try:
            value = float(value)
        except Exception:
            return default
        return value if np.isfinite(value) else default

    pec_hex_sum = float(sum(val for lbl, val in PEC.items() if _is_hex_label(lbl)))
    pec_comp_sum = float(sum(val for lbl, val in PEC.items() if _is_comp_label(lbl)))
    pec_flash_sum = float(sum(val for lbl, val in PEC.items() if _is_flash_label(lbl)))
    base_equipment_cost = pec_hex_sum + pec_comp_sum + pec_flash_sum

    refrigerant_charge_kg = max(_safe_float(kosmadakis_params.get('refrigerant_charge_kg', 0.0)), 0.0)
    refrigerant_price_eur_kg = max(_safe_float(kosmadakis_params.get('refrigerant_price_eur_kg', 50.0), 50.0), 0.0)
    gas_price_cent_kWh = max(_safe_float(kosmadakis_params.get('gas_price_cent_kWh', 8.0), 8.0), 0.0)
    gas_boiler_efficiency = max(_safe_float(kosmadakis_params.get('gas_boiler_efficiency', 0.90), 0.90), 1e-9)
    piping_factor = max(_safe_float(kosmadakis_params.get('kos_piping_factor', 0.10), 0.10), 0.0)
    electrical_factor = max(_safe_float(kosmadakis_params.get('kos_electrical_factor', 0.10), 0.10), 0.0)
    project_factor = max(_safe_float(kosmadakis_params.get('kos_project_factor', 4.16), 4.16), 0.0)
    om_factor = max(_safe_float(kosmadakis_params.get('kos_om_factor', 0.02), 0.02), 0.0)
    discount_rate = max(_safe_float(kosmadakis_params.get('kos_discount_rate', 0.05), 0.05), 0.0)

    C_p_t = piping_factor * base_equipment_cost
    C_el_CI = electrical_factor * base_equipment_cost
    C_refrigerant = refrigerant_price_eur_kg * refrigerant_charge_kg
    C_total_kos = base_equipment_cost + C_refrigerant + C_p_t + C_el_CI
    C_project = project_factor * C_total_kos

    try:
        Q_out_W = getattr(hp, 'Q_out', None)
        if Q_out_W is None or (isinstance(Q_out_W, float) and np.isnan(Q_out_W)):
            if hasattr(hp, '_get_heat_output_W'):
                Q_out_W = hp._get_heat_output_W()
            else:
                Q_out_W = hp.comps['cons'].Q.val
        annual_heat_kWh = abs(float(Q_out_W)) / 1e3 * float(tau_h_per_year)
    except Exception:
        annual_heat_kWh = 0.0

    try:
        annual_el_kWh = abs(float(hp.conns['E0'].E.val)) / 1e3 * float(tau_h_per_year)
    except Exception:
        annual_el_kWh = 0.0

    C_g = annual_heat_kWh * gas_price_cent_kWh / 100.0 / gas_boiler_efficiency
    C_el_OP = annual_el_kWh * float(elec_price_cent_kWh) / 100.0
    C_O_and_M_project = om_factor * C_project
    E_s = C_g - C_el_OP - C_O_and_M_project

    if discount_rate <= 0.0:
        PBP = C_project / E_s if E_s > 0.0 else np.nan
    elif E_s > C_project and (1.0 - C_project / E_s) > 0.0:
        PBP = np.log(1.0 / (1.0 - C_project / E_s)) / np.log(1.0 + discount_rate)
    else:
        PBP = np.nan

    return pd.DataFrame([
        {"Größe": "Σ PEC_HEX", "Wert": pec_hex_sum, "Einheit": "EUR"},
        {"Größe": "Σ PEC_compressor", "Wert": pec_comp_sum, "Einheit": "EUR"},
        {"Größe": "PEC_flashtank", "Wert": pec_flash_sum, "Einheit": "EUR"},
        {"Größe": "C_p-t", "Wert": C_p_t, "Einheit": "EUR"},
        {"Größe": "C_el^CI", "Wert": C_el_CI, "Einheit": "EUR"},
        {"Größe": "C_refrigerant", "Wert": C_refrigerant, "Einheit": "EUR"},
        {"Größe": "C_gesamt", "Wert": C_total_kos, "Einheit": "EUR"},
        {"Größe": "C_project", "Wert": C_project, "Einheit": "EUR"},
        {"Größe": "Q_Nutz", "Wert": annual_heat_kWh, "Einheit": "kWh/a"},
        {"Größe": "W_el", "Wert": annual_el_kWh, "Einheit": "kWh/a"},
        {"Größe": "C_g", "Wert": C_g, "Einheit": "EUR/a"},
        {"Größe": "C_el^OP", "Wert": C_el_OP, "Einheit": "EUR/a"},
        {"Größe": "C_O&M", "Wert": C_O_and_M_project, "Einheit": "EUR/a"},
        {"Größe": "E_s", "Wert": E_s, "Einheit": "EUR/a"},
        {"Größe": "PBP", "Wert": PBP, "Einheit": "a"},
    ])


def switch2design():
    """Switch to design simulation tab."""
    ss.select = 'Auslegung'

def st_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make a dataframe safe for Streamlit Arrow serialization."""
    if df is None:
        return None

    d = df.copy()

    # Always materialize index into columns (Arrow often chokes on mixed index)
    d = d.reset_index(drop=False)

    # Sanitize column names (Arrow hates None)
    cols = []
    for i, c in enumerate(d.columns):
        if c is None or (isinstance(c, float) and np.isnan(c)):
            cols.append(f"col_{i}")
        else:
            cols.append(str(c))
    d.columns = cols

    # Convert any object columns to string to avoid mixed types like "TOT" + numbers
    for c in d.columns:
        if d[c].dtype == "object":
            d[c] = d[c].astype(str)

    return d


def _excel_xml_sheet_name(name: str) -> str:
    cleaned = "".join(ch for ch in str(name) if ch not in r'[]:*?/\\')
    return cleaned[:31] or "Sheet"


def _excel_xml_cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '<Cell/>'
    if isinstance(value, (np.integer, int)):
        return f'<Cell><Data ss:Type="Number">{int(value)}</Data></Cell>'
    if isinstance(value, (np.floating, float)) and np.isfinite(value):
        return f'<Cell><Data ss:Type="Number">{float(value)}</Data></Cell>'
    text = escape(str(value))
    return f'<Cell><Data ss:Type="String">{text}</Data></Cell>'


def _build_excel_xml_workbook(sheets: dict[str, pd.DataFrame]) -> bytes:
    workbook = [
        '<?xml version="1.0"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"',
        ' xmlns:o="urn:schemas-microsoft-com:office:office"',
        ' xmlns:x="urn:schemas-microsoft-com:office:excel"',
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">',
    ]

    for raw_name, df in sheets.items():
        sheet_name = escape(_excel_xml_sheet_name(raw_name))
        safe_df = st_safe_df(df if isinstance(df, pd.DataFrame) else pd.DataFrame(df))
        workbook.append(f'<Worksheet ss:Name="{sheet_name}"><Table>')

        workbook.append('<Row>')
        for col in safe_df.columns:
            workbook.append(_excel_xml_cell(col))
        workbook.append('</Row>')

        for row in safe_df.itertuples(index=False, name=None):
            workbook.append('<Row>')
            for value in row:
                workbook.append(_excel_xml_cell(value))
            workbook.append('</Row>')

        workbook.append('</Table></Worksheet>')

    workbook.append('</Workbook>')
    return "".join(workbook).encode("utf-8")


def _hard_reset_model_state():
    """Drop cached model artifacts when topology changes."""
    for key in (
        "hp",
        "hp_params",
        "partload_char",
        "exergy_boundaries",
        "exergoecon_results",
        "exergy_results",
    ):
        if key in ss:
            ss.pop(key)


def switch2partload():
    """Switch to partload simulation tab."""
    ss.select = 'Teillast'


def reset2design():
    """Reset session state and switch to design simulation tab."""
    keys = list(ss.keys())
    for key in keys:
        ss.pop(key)
    ss.select = 'Auslegung'


def info_df(label, refrigs):
    """Create Dataframe with info of chosen refrigerant."""
    df_refrig = pd.DataFrame(
        columns=['Typ', 'T_NBP', 'T_krit', 'p_krit', 'SK', 'ODP', 'GWP']
        )
    df_refrig.loc[label, 'Typ'] = refrigs[label]['type']
    df_refrig.loc[label, 'T_NBP'] = str(refrigs[label]['T_NBP'])
    df_refrig.loc[label, 'T_krit'] = str(refrigs[label]['T_crit'])
    df_refrig.loc[label, 'p_krit'] = str(refrigs[label]['p_crit'])
    df_refrig.loc[label, 'SK'] = refrigs[label]['ASHRAE34']
    df_refrig.loc[label, 'ODP'] = str(refrigs[label]['ODP'])
    df_refrig.loc[label, 'GWP'] = str(refrigs[label]['GWP100'])

    return df_refrig


def calc_limits(wf, prop, padding_rel, scale='lin'):
    """
    Calculate states diagram limits of given property.

    Parameters
    ----------

    wf : str
        Working fluid for which to filter heat pump simulation results.
    
    prop : str
        Fluid property to calculate limits for.

    padding_rel : float
        Padding from minimum and maximum value to axes limit in relation to
        full range between minimum and maximum.

    scale : str
        Either 'lin' or 'log'. Scale on with padding is applied. Defaults to
        'lin'.
    """
    if scale not in ['lin', 'log']:
        raise ValueError(
            f"Parameter 'scale' has to be either 'lin' or 'log'. '{scale}' is "
            + "not allowed."
            )

    wfmask = ss.hp.nw.results['Connection'][wf] == 1.0

    min_val = ss.hp.nw.results['Connection'].loc[wfmask, prop].min()
    max_val = ss.hp.nw.results['Connection'].loc[wfmask, prop].max()
    if scale == 'lin':
        delta_val = max_val - min_val
        ax_min_val = min_val - padding_rel * delta_val
        ax_max_val = max_val + padding_rel * delta_val
    elif scale == 'log':
        delta_val = np.log10(max_val) - np.log10(min_val)
        ax_min_val = 10 ** (np.log10(min_val) - padding_rel * delta_val)
        ax_max_val = 10 ** (np.log10(max_val) + padding_rel * delta_val)

    return ax_min_val, ax_max_val


def _append_param_row(rows, group, label, value):
    """Append a formatted parameter row if the value is available."""
    if value is None or value == '':
        return
    rows.append({'Bereich': group, 'Parameter': label, 'Wert': value})


def build_selected_params_df(params, hp_model, base_topology, model_name, process_type):
    """Create a compact overview of the user-selected inputs."""
    rows = []
    setup = params.get('setup', {})
    source_mode = str(setup.get('source_mode', 'fixed_delta_T'))
    sink_mode = str(setup.get('sink_mode', 'sensible'))
    split_mode = str(setup.get('cascade_split_mode', 't_mid'))

    _append_param_row(rows, 'Szenario', 'Grundtopologie', base_topology)
    _append_param_row(rows, 'Szenario', 'Modell', model_name)
    _append_param_row(rows, 'Szenario', 'Prozessart', process_type)

    if hp_model['nr_refrigs'] == 1:
        _append_param_row(
            rows, 'Kältemittel', 'Kreis',
            params.get('setup', {}).get('refrig')
            )
    else:
        _append_param_row(
            rows, 'Kältemittel', 'Niedertemperaturkreis',
            params.get('setup', {}).get('refrig1')
            )
        _append_param_row(
            rows, 'Kältemittel', 'Hochtemperaturkreis',
            params.get('setup', {}).get('refrig2')
            )

    ambient = params.get('ambient', {})
    _append_param_row(rows, 'Umgebung', 'Temperatur', f"{ambient.get('T')} °C")
    _append_param_row(rows, 'Umgebung', 'Druck', f"{ambient.get('p')} bar")

    source_ff = params.get('B1', {})
    source_bf = params.get('B2', {})
    _append_param_row(rows, 'Wärmequelle', 'Modus', source_mode)
    _append_param_row(rows, 'Wärmequelle', 'Vorlauf', f"{source_ff.get('T')} °C")
    if source_mode == 'fixed_delta_T':
        _append_param_row(rows, 'Wärmequelle', 'Rücklauf', f"{source_bf.get('T')} °C")
        _append_param_row(
            rows, 'Wärmequelle', 'Delta T',
            f"{setup.get('source_delta_T', np.nan)} K"
        )
    else:
        _append_param_row(
            rows, 'Wärmequelle', 'Startwert Rücklauf',
            f"{source_bf.get('T')} °C"
        )
        _append_param_row(
            rows, 'Wärmequelle', 'Massenstrom',
            f"{setup.get('m_source', source_ff.get('m'))} kg/s"
        )
    if 'p' in source_ff:
        _append_param_row(
            rows, 'Wärmequelle', 'Eintrittsdruck',
            f"{source_ff.get('p')} bar"
            )
    _append_param_row(
        rows, 'Wärmequelle', 'Quellenpumpe modelliert',
        'Ja' if bool(setup.get('use_source_pump', True)) else 'Nein'
    )

    sink_rf = params.get('C1', {})
    sink_ff = params.get('C3', {})
    _append_param_row(rows, 'Wärmesenke', 'Modus', sink_mode)
    if sink_mode == 'steam':
        _append_param_row(
            rows, 'Wärmesenke', 'Dampftemperatur',
            f"{setup.get('T_steam', sink_rf.get('T'))} °C"
        )
        _append_param_row(
            rows, 'Wärmesenke', 'Dampfmassenstrom',
            f"{setup.get('m_steam', sink_rf.get('m'))} kg/s"
        )
        _append_param_row(
            rows, 'Wärmesenke', 'Sattdruck',
            f"{sink_rf.get('p', sink_ff.get('p'))} bar"
        )
    else:
        _append_param_row(rows, 'Wärmesenke', 'Rücklauf', f"{sink_rf.get('T')} °C")
        if 'C2' in params:
            _append_param_row(
                rows, 'Wärmesenke', 'Zwischenzustand',
                f"{params['C2'].get('T')} °C"
                )
        _append_param_row(rows, 'Wärmesenke', 'Vorlauf', f"{sink_ff.get('T')} °C")
        if 'p' in sink_ff:
            _append_param_row(
                rows, 'Wärmesenke', 'Druck',
                f"{sink_ff.get('p')} bar"
                )
    _append_param_row(
        rows, 'Wärmesenke', 'Senkenpumpe modelliert',
        'Ja' if bool(setup.get('use_sink_pump', True)) else 'Nein'
    )

    if 'A0' in params and 'p' in params['A0']:
        _append_param_row(
            rows, 'Prozess', 'Hochdruck',
            f"{params['A0']['p']} bar"
            )

    cons = params.get('cons', {})
    if 'Q' in cons:
        _append_param_row(
            rows, 'Prozess', 'Heizleistung Soll',
            f"{abs(cons['Q']) / 1e6:.2f} MW"
            )

    if 'global_ttd_main_hex' in setup:
        _append_param_row(
            rows, 'Hauptwärmeübertrager', 'Globaler Minimalabstand',
            f"{float(setup['global_ttd_main_hex']):.1f} K"
        )
    if supports_explicit_suction_superheat(hp_model) and 'dT_sup' in setup:
        _append_param_row(
            rows, 'Kreisprozess', 'Verdichtersaugüberhitzung',
            f"{float(setup['dT_sup']):.1f} K"
        )
    if supports_explicit_subcooling(hp_model) and 'dT_sub' in setup:
        _append_param_row(
            rows, 'Kreisprozess', 'Unterkühlung',
            f"{float(setup['dT_sub']):.1f} K"
        )
    calibration_mode = str(setup.get('calibration_mode', 'rip'))
    if (
        supports_rip_factor(hp_model)
        and calibration_mode != 'a_target'
        and 'rip_factor' in setup
    ):
        _append_param_row(
            rows, 'Kreisprozess', 'Zwischendruckfaktor RIP',
            f"{float(setup['rip_factor']):.3f}"
        )
    if calibration_mode == 'a_target':
        _append_param_row(
            rows, 'Kreisprozess', 'Kalibriermodus',
            'A_target -> RIP'
        )
        if 'A_target' in setup:
            _append_param_row(
                rows, 'Kreisprozess', 'A_target',
                f"{float(setup['A_target']):.3f}"
            )
    if supports_explicit_injection_superheat(hp_model) and 'dT_sup_inj' in setup:
        _append_param_row(
            rows, 'Kreisprozess', 'Einspritz-Überhitzung',
            f"{float(setup['dT_sup_inj']):.1f} K"
        )
    if hp_model['nr_refrigs'] == 2 and 't_mid' in setup:
        _append_param_row(
            rows, 'Zwischenwärmeübertrager', 'T_mid',
            f"{float(setup['t_mid']):.2f} °C"
        )
    if hp_model['nr_refrigs'] == 2:
        _append_param_row(
            rows, 'Zwischenwärmeübertrager', 'Splitmodus', split_mode
        )
    if hp_model['nr_refrigs'] == 2 and 't_mid_fraction' in setup:
        _append_param_row(
            rows, 'Zwischenwärmeübertrager', 'Splitfaktor α',
            f"{float(setup['t_mid_fraction']):.3f}"
        )
    if hp_model['nr_refrigs'] == 2 and 'lift_share' in setup:
        _append_param_row(
            rows, 'Zwischenwärmeübertrager', 'Lift Share',
            f"{float(setup['lift_share']):.3f}"
        )
    if 'motor_eta' in setup:
        _append_param_row(
            rows, 'System', 'Motorwirkungsgrad',
            f"{float(setup['motor_eta']) * 100:.1f} %"
        )
    if 'skip_ommen_check' in setup:
        _append_param_row(
            rows, 'System', 'Ommen-Check deaktiviert',
            'Ja' if bool(setup['skip_ommen_check']) else 'Nein'
        )

    for comp_key in (
        'comp', 'comp1', 'comp2',
        'HT_comp', 'LT_comp', 'HT_comp1', 'HT_comp2', 'LT_comp1', 'LT_comp2'
    ):
        if comp_key in params and 'eta_s' in params[comp_key]:
            _append_param_row(
                rows, 'Verdichter', comp_key,
                f"{params[comp_key]['eta_s'] * 100:.0f} %"
                )

    for ihx_key in ('ihx', 'ihx1', 'ihx2'):
        if ihx_key in params and 'dT_sh' in params[ihx_key]:
            _append_param_row(
                rows, 'Interne Wärmeübertragung', ihx_key,
                f"{params[ihx_key]['dT_sh']} K"
                )

    return pd.DataFrame(rows)


def build_literature_metrics_export_df(hp, hp_model_name):
    """Create a one-row export table with literature-oriented KPIs."""
    if hp is None or not hasattr(hp, 'get_literature_comparison_metrics'):
        return pd.DataFrame()

    metrics = hp.get_literature_comparison_metrics()
    if not metrics:
        return pd.DataFrame()

    return pd.DataFrame([
        {
            'Kennzahl': 'A_inj',
            'Wert': (
                f"A_inj ≈ {float(metrics.get('A_inj', np.nan)):.3f}"
                if np.isfinite(metrics.get('A_inj', np.nan)) else '—'
            )
        },
        {
            'Kennzahl': 'T_discharge_final',
            'Wert': (
                f"T_discharge_final ≈ "
                f"{float(metrics.get('T_discharge_final_C', np.nan)):.2f} °C"
                if np.isfinite(metrics.get('T_discharge_final_C', np.nan)) else '—'
            )
        },
        {
            'Kennzahl': 'VHC',
            'Wert': (
                f"VHC ≈ {float(metrics.get('VHC_MJ_m3', np.nan)):.2f} MJ/m³"
                if np.isfinite(metrics.get('VHC_MJ_m3', np.nan)) else '—'
            )
        }
    ])


def supports_general_boundary_modes(hp_model_name):
    """Return whether general HTHP boundary modes are exposed in the UI."""
    return hp_model_name in {'cascade', 'cascade_mbhx', 'cascade_2ihx'}


def _load_hthp_reference_cases():
    """Load local literature reference cases for HTHP comparisons."""
    ref_path = os.path.join(_get_input_param_dir(), 'hthp_reference_cases.json')
    if not os.path.exists(ref_path):
        return {}
    try:
        data = _load_input_json(ref_path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def get_reference_case_data(params):
    """Return literature reference metadata for the current parameter set."""
    ref_id = str(params.get('setup', {}).get('reference_id', '')).strip()
    if not ref_id:
        return None
    case = _load_hthp_reference_cases().get(ref_id)
    return case if isinstance(case, dict) else None


def _format_reference_value(value, unit='-', digits=2):
    """Format a numeric value with unit for the reference comparison tables."""
    try:
        value = float(value)
    except Exception:
        return '—'
    if not np.isfinite(value):
        return '—'
    if unit == '-' or not unit:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} {unit}"


def build_reference_comparison_df(hp, params):
    """Create a literature comparison table for supported reference cases."""
    case = get_reference_case_data(params)
    if case is None or hp is None or not hasattr(hp, 'get_reference_case_metrics'):
        return pd.DataFrame()

    current = hp.get_reference_case_metrics() or {}
    rows = []

    for metric in case.get('metrics', []):
        if not isinstance(metric, dict):
            continue

        key = metric.get('key')
        label = metric.get('label', key)
        unit = metric.get('unit', '-')
        digits = int(metric.get('digits', 2))

        try:
            ref_value = float(metric.get('reference', np.nan))
        except Exception:
            ref_value = np.nan
        try:
            model_value = float(current.get(key, np.nan))
        except Exception:
            model_value = np.nan

        abs_dev = np.nan
        rel_dev_pct = np.nan
        if np.isfinite(ref_value) and np.isfinite(model_value):
            abs_dev = model_value - ref_value
            if not np.isclose(ref_value, 0.0):
                rel_dev_pct = abs_dev / ref_value * 100.0

        rows.append({
            'Kennzahl': label,
            'Referenz': _format_reference_value(ref_value, unit=unit, digits=digits),
            'Modell': _format_reference_value(model_value, unit=unit, digits=digits),
            'Abweichung absolut': _format_reference_value(abs_dev, unit=unit, digits=digits),
            'Abweichung relativ': (
                f"{rel_dev_pct:+.{digits}f} %"
                if np.isfinite(rel_dev_pct) else '—'
            ),
        })

    return pd.DataFrame(rows)


def build_reference_internal_states_export_df(hp, params):
    """Create an export table with internal thermodynamic reference metrics."""
    case = get_reference_case_data(params)
    if case is None or hp is None or not hasattr(hp, 'get_reference_case_metrics'):
        return pd.DataFrame()

    current = hp.get_reference_case_metrics() or {}
    metric_lookup = {}
    for metric in case.get('metrics', []):
        if isinstance(metric, dict) and metric.get('key'):
            metric_lookup[str(metric['key'])] = metric

    internal_metric_order = [
        'm_cycle1_kg_s',
        'm_cycle2_kg_s',
        'V_dot_c1_m3_h',
        'V_dot_c2_m3_h',
        'p_high_c1_bar',
        'T_disch_c1_C',
        'p_high_c2_bar',
        'T_disch_c2_C',
    ]

    rows = []
    for key in internal_metric_order:
        metric = metric_lookup.get(key)
        if metric is None:
            continue

        label = metric.get('label', key)
        unit = metric.get('unit', '-')
        digits = int(metric.get('digits', 2))

        try:
            ref_value = float(metric.get('reference', np.nan))
        except Exception:
            ref_value = np.nan
        try:
            model_value = float(current.get(key, np.nan))
        except Exception:
            model_value = np.nan

        abs_dev = np.nan
        rel_dev_pct = np.nan
        if np.isfinite(ref_value) and np.isfinite(model_value):
            abs_dev = model_value - ref_value
            if not np.isclose(ref_value, 0.0):
                rel_dev_pct = abs_dev / ref_value * 100.0

        rows.append({
            'Kennzahl': label,
            'Einheit': unit,
            'Referenz': ref_value,
            'Modell': model_value,
            'Abweichung absolut': abs_dev,
            'Abweichung relativ [%]': rel_dev_pct,
        })

    return pd.DataFrame(rows)


def build_reference_published_df(params):
    """Create a table with published reference values not yet compared live."""
    case = get_reference_case_data(params)
    if case is None:
        return pd.DataFrame()

    rows = []
    for metric in case.get('published_only_metrics', []):
        if not isinstance(metric, dict):
            continue
        rows.append({
            'Kennzahl': metric.get('label', ''),
            'Veröffentlicht': _format_reference_value(
                metric.get('reference', np.nan),
                unit=metric.get('unit', '-'),
                digits=int(metric.get('digits', 2)),
            ),
        })

    return pd.DataFrame(rows)


def get_source_mode_from_params(params):
    """Return the selected source mode from raw params."""
    return str(params.get('setup', {}).get('source_mode', 'fixed_delta_T'))


def get_sink_mode_from_params(params):
    """Return the selected sink mode from raw params."""
    return str(params.get('setup', {}).get('sink_mode', 'sensible'))


def get_source_delta_T_from_params(params):
    """Return the configured source-side temperature drop in K."""
    setup = params.get('setup', {})
    if 'source_delta_T' in setup:
        return float(setup['source_delta_T'])
    return float(params['B1']['T']) - float(params['B2']['T'])


def get_source_cold_design_temp_from_params(params):
    """Return the source outlet temperature used for cascade setup."""
    if get_source_mode_from_params(params) == 'fixed_delta_T':
        return float(params['B2']['T'])
    return float(params['B1']['T']) - get_source_delta_T_from_params(params)


def get_sink_hot_target_temp_from_params(params):
    """Return the target hot sink temperature for the selected sink mode."""
    if get_sink_mode_from_params(params) == 'steam':
        return float(params.get('setup', {}).get('T_steam', params['C1']['T']))
    return float(
        params.get('C2', {}).get(
            'T', params.get('C3', {}).get('T', params['C1']['T'])
        )
    )


def get_cascade_t_mid_bounds(params):
    """Return physical bounds for the cascade intermediate temperature."""
    t_source_cold = get_source_cold_design_temp_from_params(params)
    t_sink_hot = get_sink_hot_target_temp_from_params(params)
    inter_ttd = float(params.get('inter', {}).get('ttd_u', 0.0))
    t_mid_min = t_source_cold + inter_ttd / 2
    t_mid_max = t_sink_hot - inter_ttd / 2
    return t_source_cold, t_sink_hot, t_mid_min, t_mid_max


def supports_explicit_suction_superheat(hp_model):
    """Return whether a model supports a separate suction superheat input."""
    return hp_model['nr_refrigs'] == 1 and hp_model['nr_ihx'] == 0


def supports_explicit_subcooling(hp_model):
    """Return whether a model supports a separate liquid subcooling input."""
    return (
        supports_explicit_suction_superheat(hp_model)
        and hp_model['process_type'] == 'subcritical'
    )


def supports_rip_factor(hp_model):
    """Return whether a model uses an adjustable intermediate pressure."""
    return hp_model['nr_refrigs'] == 1 and hp_model['comp_var'] is not None


def supports_explicit_injection_superheat(hp_model):
    """Return whether a model supports explicit injection superheat input."""
    return (
        hp_model['base_topology'] == 'Economizer'
        and hp_model['nr_refrigs'] == 1
        and hp_model['nr_ihx'] == 0
        and hp_model['econ_type'] == 'closed'
    )


def supports_a_target_calibration(hp_model_name):
    """Return whether a model supports A-target literature calibration."""
    return hp_model_name in {'flash', 'econ_closed'}


def get_main_hex_ttd_targets(params):
    """Return all main heat exchanger TTD parameters eligible for global control."""
    targets = []
    for comp_key, field in (
        ('evap', 'ttd_l'),
        ('cond', 'ttd_u'),
        ('inter', 'ttd_u'),
        ('trans', 'ttd_l'),
        ('econ', 'ttd_l'),
        ('econ1', 'ttd_l'),
        ('econ2', 'ttd_l'),
    ):
        if comp_key in params and field in params[comp_key]:
            targets.append((comp_key, field))
    return targets


def apply_global_main_hex_ttd(params, value):
    """Apply one global minimum temperature difference to all main HEX targets."""
    value = float(value)
    for comp_key, field in get_main_hex_ttd_targets(params):
        params[comp_key][field] = value
    params.setdefault('setup', {})
    params['setup']['global_ttd_main_hex'] = value


def _get_input_param_dir():
    """Return absolute path to the model input parameter directory."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'models', 'input')
    )


def _load_input_json(json_path):
    """Load a JSON file from disk."""
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _apply_dashboard_preset_defaults(params):
    """
    Seed dashboard-side cost and finance defaults from a loaded preset.

    Values are only written if no user state exists yet, so manual edits remain
    intact until the parameter preset is changed (which already resets widget
    state via the model signature logic).
    """
    for key, value in params.get('costcalcparams', {}).items():
        ss.setdefault(key, value)
    for key, value in params.get('econ_params', {}).items():
        ss.setdefault(key, value)


def _restore_cost_finance_widget_state(params):
    """
    Restore sidebar cost/finance widget state from the last valid run.

    Streamlit occasionally reinitializes a cluster of numeric widgets to their
    minimum values on rerun. If that happens, recover the last valid values
    from cached UI state or fall back to preset/default values.
    """
    cost_defaults = {
        'cost_method': 'standard',
        'current_year': '2025',
        'analysis_year': 2025,
        'hx_area_method': 'q_lmtd',
        'compressor_eta_vol': 1.0,
        'compressor_eta_vol_lt': 1.0,
        'compressor_eta_vol_ht': 1.0,
        'elec_price_cent_kWh': 40.0,
        'b1_cost_eur_per_GJ': 0.0,
        'tau_h_per_year': 5500.0,
        'usd_to_eur': 0.93,
        'hex_cost_model': 'ommen',
        'compressor_cost_model': 'ommen',
        'flash_cost_model': 'ommen',
        'include_pumps_in_pec': True,
        'k_evap': 1500,
        'k_cond': 3500,
        'k_inter': 2200,
        'k_ihx': 1500,
        'k_trans': 60,
        'k_misc': 50,
        'residence_time': 10,
    }
    kosmadakis_defaults = {
        'gas_price_cent_kWh': 8.0,
        'gas_boiler_efficiency': 0.90,
        'refrigerant_charge_kg': 0.0,
        'refrigerant_price_eur_kg': 50.0,
        'kos_piping_factor': 0.10,
        'kos_electrical_factor': 0.10,
        'kos_project_factor': 4.16,
        'kos_om_factor': 0.02,
        'kos_discount_rate': 0.05,
    }
    econ_defaults = {
        'i_eff': 0.08,
        'r_n': 0.02,
        'r_n_om': 0.02,
        'r_n_el': 0.02,
        'n': 20,
        'omc_rel': 0.03,
        'tci_factor': 6.32,
        'install_factor': 4.16,
    }

    preset_costs = params.get('costcalcparams', {}) or {}
    preset_econ = params.get('econ_params', {}) or {}
    cached_costs = ss.get('costcalcparams', {}) or {}
    cached_econ = ss.get('econ_ui_params', {}) or {}
    cached_kosmadakis = ss.get('kosmadakis_params', {}) or {}

    def _looks_like_min_reset(costs=None, econ=None):
        costs = costs or ss
        econ = econ or ss
        try:
            return (
                float(costs.get('k_misc', cost_defaults['k_misc'])) == 0.0
                and float(costs.get('residence_time', cost_defaults['residence_time'])) == 0.0
                and float(econ.get('i_eff', econ_defaults['i_eff'])) == 0.0
                and float(econ.get('r_n', econ_defaults['r_n'])) == 0.0
                and float(econ.get('r_n_om', econ_defaults['r_n_om'])) == 0.0
                and float(econ.get('r_n_el', econ_defaults['r_n_el'])) == 0.0
                and int(econ.get('n', econ_defaults['n'])) == 1
                and float(econ.get('omc_rel', econ_defaults['omc_rel'])) == 0.0
                and float(econ.get('tci_factor', econ_defaults['tci_factor'])) == 0.0
                and float(econ.get('install_factor', econ_defaults['install_factor'])) == 0.0
            )
        except Exception:
            return False

    force_restore = _looks_like_min_reset()
    cached_invalid = _looks_like_min_reset(cached_costs, cached_econ)
    try:
        kosmadakis_invalid = float(ss.get('kos_project_factor', kosmadakis_defaults['kos_project_factor'])) == 0.0
    except Exception:
        kosmadakis_invalid = False

    for key, default in cost_defaults.items():
        if not cached_invalid and key in cached_costs:
            target = cached_costs[key]
        else:
            target = preset_costs.get(key, default)
        if force_restore or key not in ss:
            ss[key] = target

    for key, default in kosmadakis_defaults.items():
        if not kosmadakis_invalid and key in cached_kosmadakis:
            target = cached_kosmadakis[key]
        else:
            target = preset_costs.get(key, default)
        if force_restore or kosmadakis_invalid or key not in ss:
            ss[key] = target

    for key, default in econ_defaults.items():
        if not cached_invalid and key in cached_econ:
            target = cached_econ[key]
        else:
            target = preset_econ.get(key, default)
        if force_restore or key not in ss:
            ss[key] = target


def _session_value_kwargs(key, default):
    """Return a widget `value` kwarg only if the session state is unset."""
    if key in ss:
        return {}
    return {'value': default}


def _session_index_kwargs(key, default):
    """Return a widget `index` kwarg only if the session state is unset."""
    if key in ss:
        return {}
    return {'index': default}


def resolve_topology_svg_path(src_path, hp_model_name_topology, *, is_dark=False,
                              labeled=False):
    """Return the best matching static topology SVG path or ``None``."""
    topologies_dir = os.path.join(src_path, 'img', 'topologies')
    suffix = '_label' if labeled else ''
    variants = [str(hp_model_name_topology)]

    # Experimental models may intentionally reuse the static diagram of their
    # closest stable base model until a dedicated SVG exists.
    if variants[0].endswith('_mbhx'):
        variants.append(variants[0].removesuffix('_mbhx'))

    filenames = []
    if is_dark:
        filenames.extend([
            f'hp_{name}{suffix}_dark.svg' for name in variants
        ])
    filenames.extend([
        f'hp_{name}{suffix}.svg' for name in variants
    ])

    for filename in filenames:
        path = os.path.join(topologies_dir, filename)
        if os.path.exists(path):
            return path
    return None


CURATED_MODEL_PRESETS = {
    'cascade': [
        {
            'filename': 'params_hthp_cascade_ref_r290_r600.json',
            'label': 'HTHP-Referenzfall - R290/R600 - LS 0.4 - Tsrc 40 °C - Dampf 110 °C',
        },
        {
            'filename': 'params_hthp_cascade_ref_r290_r600_standard_cost.json',
            'label': 'HTHP-Basisreferenzfall - R290/R600 - LS 0.4 - Tsrc 40 °C - Dampf 110 °C | Dashboard-Standardkosten',
        },
        {
            'filename': 'params_hthp_cascade_ref_r717_r600.json',
            'label': 'HTHP-Referenzfall - R717/R600 - LS 0.4 - Tsrc 40 °C - Dampf 110 °C',
        },
        {
            'filename': 'params_hthp_cascade_ref_r717_r600_standard_cost.json',
            'label': 'HTHP-Referenzfall - R717/R600 - LS 0.4 - Tsrc 40 °C - Dampf 110 °C | Dashboard-Standardkosten',
        },
        {
            'filename': 'params_hthp_cascade_ref_r290_r600_ls30_tsrc40.json',
            'label': 'HTHP-Referenzfall - R290/R600 - LS 0.3 - Tsrc 40 °C - Dampf 110 °C',
        },
        {
            'filename': 'params_hthp_cascade_ref_r290_r600_ls30_tsrc40_standard_cost.json',
            'label': 'HTHP-Referenzfall - R290/R600 - LS 0.3 - Tsrc 40 °C - Dampf 110 °C | Dashboard-Standardkosten',
        },
        {
            'filename': 'params_hthp_cascade_ref_r290_r600_ls40_tsrc30.json',
            'label': 'HTHP-Referenzfall - R290/R600 - LS 0.4 - Tsrc 30 °C - Dampf 110 °C',
        },
        {
            'filename': 'params_hthp_cascade_ref_r290_r600_ls40_tsrc30_standard_cost.json',
            'label': 'HTHP-Referenzfall - R290/R600 - LS 0.4 - Tsrc 30 °C - Dampf 110 °C | Dashboard-Standardkosten',
        },
    ],
    'cascade_mbhx': [
        {
            'filename': 'params_hthp_cascade_mbhx_ref_r290_r600.json',
            'label': 'HTHP-Referenzfall - R290/R600 - LS 0.4 - Tsrc 40 °C - Dampf 110 °C | MBHX',
        },
    ],
    'cascade_2ihx': [
        {
            'filename': 'params_hthp_cascade_2ihx_compare_r290_r600.json',
            'label': 'Topologievergleich - Kaskade 2 IHX - R290/R600 - LS 0.4 - Tsrc 40 °C - Dampf 110 °C',
        },
        {
            'filename': 'params_hthp_cascade_2ihx_compare_r717_r600.json',
            'label': 'Topologievergleich - Kaskade 2 IHX - R717/R600 - LS 0.4 - Tsrc 40 °C - Dampf 110 °C',
        },
        {
            'filename': 'params_hthp_cascade_2ihx_compare_r290_r600_ls30_tsrc40.json',
            'label': 'Topologievergleich - Kaskade 2 IHX - R290/R600 - LS 0.3 - Tsrc 40 °C - Dampf 110 °C',
        },
        {
            'filename': 'params_hthp_cascade_2ihx_compare_r290_r600_ls40_tsrc30.json',
            'label': 'Topologievergleich - Kaskade 2 IHX - R290/R600 - LS 0.4 - Tsrc 30 °C - Dampf 110 °C',
        },
    ],
    'flash': [
        {
            'filename': 'params_yang2024_flash.json',
            'label': 'Yang 2024 - Reales Verhalten',
        },
        {
            'filename': 'params_yang2024_flash_literature_idealized.json',
            'label': 'Yang 2024 - Idealisiert',
        },
    ],
    'econ_closed': [
        {
            'filename': 'params_yang2024_econ_closed.json',
            'label': 'Yang 2024 - Reales Verhalten',
        },
        {
            'filename': 'params_yang2024_econ_closed_literature_idealized.json',
            'label': 'Yang 2024 - Idealisiert',
        },
    ],
}


def get_param_presets_for_model(hp_model_name):
    """Return the standard parameter file and matching custom presets."""
    input_dir = _get_input_param_dir()
    standard_filename = f'params_hp_{hp_model_name}.json'
    standard_path = os.path.join(input_dir, standard_filename)
    standard_params = _load_input_json(standard_path)
    standard_type = standard_params.get('setup', {}).get('type')

    curated_presets = []
    for preset_meta in CURATED_MODEL_PRESETS.get(hp_model_name, []):
        json_path = os.path.join(input_dir, preset_meta['filename'])
        if not os.path.exists(json_path):
            continue

        try:
            params = _load_input_json(json_path)
        except Exception:
            continue

        setup = params.get('setup', {})
        if not isinstance(setup, dict):
            continue
        if setup.get('type') != standard_type:
            continue

        curated_presets.append({
            'label': preset_meta['label'],
            'filename': preset_meta['filename'],
            'path': json_path,
            'is_standard': False,
            'setup_name': str(
                setup.get('name') or os.path.splitext(preset_meta['filename'])[0]
            ),
        })

    presets = [{
        'label': 'Standardparameter',
        'filename': standard_filename,
        'path': standard_path,
        'is_standard': True,
        'setup_name': standard_params.get('setup', {}).get('name', ''),
    }]

    if curated_presets:
        presets.extend(curated_presets)
        return presets

    custom_presets = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.json'):
            continue
        if filename in {
            standard_filename,
            'CEPCI.json',
            'state_diagram_config.json',
        }:
            continue

        json_path = os.path.join(input_dir, filename)
        try:
            params = _load_input_json(json_path)
        except Exception:
            continue

        setup = params.get('setup', {})
        if not isinstance(setup, dict):
            continue
        if setup.get('type') != standard_type:
            continue

        setup_name = str(setup.get('name') or os.path.splitext(filename)[0])
        custom_presets.append({
            'label': f'{setup_name} ({filename})',
            'filename': filename,
            'path': json_path,
            'is_standard': False,
            'setup_name': setup_name,
        })

    presets.extend(
        sorted(custom_presets, key=lambda preset: preset['label'].lower())
    )
    return presets


def _reset_design_widget_state(keep_keys):
    """Clear design widget state while preserving selection controls."""
    for key in list(ss.keys()):
        if key not in keep_keys:
            ss.pop(key)


def img_to_base64(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


@st.dialog("Kontaktdaten")
def footer():
    st.markdown(f"""
        <div style='font-size: 1.0em;'>
            <div style='margin-bottom: 0.5em;'>
                <strong>Jonas Freißmann</strong>
                <img src="https://avatars.githubusercontent.com/u/57762052?v=4" width="32" style="margin: 0 10px;"><br>
            </div>
            <p style="margin-bottom: 0.3em;">jonas.freissmann@web.de</p>
            <a href="mailto:jonas.freissmann@web.de" style="text-decoration: none;">
                <img src="data:image/svg+xml;base64,{mail64}" width="32" style="margin: 10px 10px 10px 0;">
            </a>
            <a href="https://orcid.org/0009-0007-6432-5479" target="_blank" style="text-decoration: none;">
                <img src="data:image/svg+xml;base64,{orcid64}" width="29" style="margin: 0 10px;">
            </a>
            <a href="https://github.com/jfreissmann" target="_blank" style="text-decoration: none;">
                <img src="data:image/svg+xml;base64,{github64}" width="30" style="margin: 0 10px;">
            </a>
            <a href="https://www.linkedin.com/in/jonas-frei%C3%9Fmann-8a6401368/" target="_blank" style="text-decoration: none;">
                <img src="data:image/svg+xml;base64,{linkedin64}" width="35" style="margin: 0 10px;">
            </a><br><br><br>
            <div style='margin-bottom: 0.5em;'>
                <strong>Malte Fritz</strong>
                <img src="https://avatars.githubusercontent.com/u/35224977?v=4" width="32" style="margin: 0 10px;"><br>
            </div>
            <p style="margin-bottom: 0.3em;">malte.fritz@web.de</p>
            <a href="mailto:malte.fritz@web.de" style="text-decoration: none;">
                <img src="data:image/svg+xml;base64,{mail64}" width="32" style="margin: 10px 10px 10px 0;">
            </a>
            <a href="https://orcid.org/my-orcid?orcid=0009-0001-5843-0973" target="_blank" style="text-decoration: none;">
                <img src="data:image/svg+xml;base64,{orcid64}" width="29" style="margin: 0 10px;">
            </a>
            <a href="https://github.com/maltefritz" target="_blank" style="text-decoration: none;">
                <img src="data:image/svg+xml;base64,{github64}" width="30" style="margin: 0 10px;">
            </a>
            <a href="https://www.linkedin.com/in/malte-fritz-515259100" target="_blank" style="text-decoration: none;">
                <img src="data:image/svg+xml;base64,{linkedin64}" width="35" style="margin: 0 10px;">
            </a>
        </div><br>
        """, unsafe_allow_html=True)


src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
icon_path = os.path.join(src_path, 'img', 'icons')

# %% MARK: Initialisation
refrigpath = os.path.join(src_path, 'refrigerants.json')
with open(refrigpath, 'r', encoding='utf-8') as file:
    refrigerants = json.load(file)

st.set_page_config(
    layout='wide',
    page_title='heatpumps',
    page_icon=os.path.join(icon_path, 'page_icon_ZNES.png')
    )

is_dark = darkdetect.isDark()

# %% MARK: Sidebar
with st.sidebar:
    if is_dark:
        logo = os.path.join(src_path, 'img', 'Logo_ZNES_mitUnisV2_dark.svg')
    else:
        logo = os.path.join(src_path, 'img', 'Logo_ZNES_mitUnisV2.svg')
    st.image(logo, use_container_width=True)

    mode = st.selectbox(
        'Auswahl Modus', ['Start', 'Auslegung', 'Teillast'],
        key='select', label_visibility='hidden'
        )

    st.markdown("""---""")

    # %% MARK: Design
    if mode == 'Auslegung':
        ss.rerun_req = True
        st.header('Auslegung der Wärmepumpe')

        with st.expander('Setup', expanded=True):
            base_topology = st.selectbox(
                'Grundtopologie',
                var.base_topologies,
                index=0, key='base_topology'
            )

            models = []
            for model, mdata in var.hp_models.items():
                if mdata['base_topology'] == base_topology:
                    if mdata['process_type'] != 'transcritical':
                        models.append(mdata['display_name'])

            model_name = st.selectbox(
                'Wärmepumpenmodell', models, index=0, key='model'
            )

            process_type = st.radio(
                'Prozessart', options=('subkritisch', 'transkritisch'),
                horizontal=True
            )

            if process_type == 'transkritisch':
                model_name = f'{model_name} | Transkritisch'

            for model, mdata in var.hp_models.items():
                correct_base = mdata['base_topology'] == base_topology
                correct_model_name = mdata['display_name'] == model_name
                if correct_base and correct_model_name:
                    hp_model = mdata
                    hp_model_name = model
                    if 'trans' in hp_model_name:
                        hp_model_name_topology = hp_model_name.replace(
                            '_trans', ''
                            )
                    else:
                        hp_model_name_topology = hp_model_name
                    break

            preset_options = get_param_presets_for_model(hp_model_name)
            preset_labels = {
                preset['path']: preset['label'] for preset in preset_options
            }
            preset_filenames = {
                preset['path']: preset['filename'] for preset in preset_options
            }
            preset_paths = [preset['path'] for preset in preset_options]
            if ss.get('param_preset_path') not in preset_paths:
                ss.param_preset_path = preset_paths[0]

            selected_param_path = st.selectbox(
                'Parametersatz',
                options=preset_paths,
                format_func=lambda path: preset_labels[path],
                key='param_preset_path',
                help='Standardparameter oder kompatible JSON-Presets fuer '
                     'das aktuell gewaehlte Modell laden.'
            )

            st.caption(
                'Geladene Parameterdatei: '
                + preset_filenames[selected_param_path]
            )

            # Reset widget state when topology/model/preset changes.
            model_signature = (
                f"{base_topology}|{model_name}|{process_type}|"
                + f"{hp_model_name}|{selected_param_path}"
            )
            if ss.get("hp_model_signature") != model_signature:
                _reset_design_widget_state({
                    'select',
                    'base_topology',
                    'model',
                    'param_preset_path',
                })
                ss.hp_model_signature = model_signature

            parampath = selected_param_path
            params = deepcopy(_load_input_json(parampath))
            _apply_dashboard_preset_defaults(params)
            _restore_cost_finance_widget_state(params)
        if hp_model['nr_ihx'] == 1:
            with st.expander('Interne Wärmerübertragung'):
                params['ihx']['dT_sh'] = st.slider(
                    'Überhitzung/Unterkühlung', value=5,
                    min_value=0, max_value=25, format='%d°C',
                    key='dT_sh')
        if hp_model['nr_ihx'] > 1:
            with st.expander('Interne Wärmerübertragung'):
                dT_ihx = {}
                for i in range(1, hp_model['nr_ihx']+1):
                     dT_ihx[i] = st.slider(
                        f'Nr. {i}: Überhitzung/Unterkühlung', value=5,
                        min_value=0, max_value=25, format='%d°C',
                        key=f'dT_ihx{i}'
                        )
                     params[f'ihx{i}']['dT_sh'] = dT_ihx[i]

        with st.expander('Kältemittel'):
            if hp_model['nr_refrigs'] == 1:
                refrig_index = None
                for ridx, (rlabel, rdata) in enumerate(refrigerants.items()):
                    if rlabel == params['setup']['refrig']:
                        refrig_index = ridx
                        break
                    elif rdata['CP'] == params['setup']['refrig']:
                        refrig_index = ridx
                        break

                refrig_label = st.selectbox(
                    'Kältemittel', refrigerants.keys(), index=refrig_index,
                    key='refrigerant', label_visibility='hidden'
                    )
                params['setup']['refrig'] = refrigerants[refrig_label]['CP']
                params['fluids']['wf'] = refrigerants[refrig_label]['CP']
                df_refrig = info_df(refrig_label, refrigerants)

            elif hp_model['nr_refrigs'] == 2:
                refrig2_index = None
                for ridx, (rlabel, rdata) in enumerate(refrigerants.items()):
                    if rlabel == params['setup']['refrig2']:
                        refrig2_index = ridx
                        break
                    elif rdata['CP'] == params['setup']['refrig2']:
                        refrig2_index = ridx
                        break

                refrig2_label = st.selectbox(
                    'Kältemittel (Hochtemperaturkreis)', refrigerants.keys(),
                    index=refrig2_index, key='refrigerant2'
                    )
                params['setup']['refrig2'] = refrigerants[refrig2_label]['CP']
                params['fluids']['wf2'] = refrigerants[refrig2_label]['CP']
                df_refrig2 = info_df(refrig2_label, refrigerants)

                refrig1_index = None
                for ridx, (rlabel, rdata) in enumerate(refrigerants.items()):
                    if rlabel == params['setup']['refrig1']:
                        refrig1_index = ridx
                        break
                    elif rdata['CP'] == params['setup']['refrig1']:
                        refrig1_index = ridx
                        break

                refrig1_label = st.selectbox(
                    'Kältemittel (Niedertemperaturkreis)', refrigerants.keys(),
                    index=refrig1_index, key='refrigerant1'
                    )
                params['setup']['refrig1'] = refrigerants[refrig1_label]['CP']
                params['fluids']['wf1'] = refrigerants[refrig1_label]['CP']
                df_refrig1 = info_df(refrig1_label, refrigerants)


        if hp_model['nr_refrigs'] == 1:
            T_crit = int(np.floor(refrigerants[refrig_label]['T_crit']))
            p_crit = int(np.floor(refrigerants[refrig_label]['p_crit']))
        elif hp_model['nr_refrigs'] == 2:
            T_crit = int(np.floor(refrigerants[refrig2_label]['T_crit']))
            p_crit = int(np.floor(refrigerants[refrig2_label]['p_crit']))

        ss.T_crit = T_crit
        ss.p_crit = p_crit

        if 'trans' in hp_model_name:
            with st.expander('Traskritischer Druck'):
                params['A0']['p'] = st.slider(
                    'Wert in bar', min_value=ss.p_crit,
                    value=params['A0']['p'], max_value=300, format='%d bar',
                    key='p_trans_out'
                    )

        with st.expander('Thermische Nennleistung'):
            params['cons']['Q'] = st.number_input(
                'Wert in MW', value=abs(params['cons']['Q']/1e6),
                step=0.1, key='Q_N'
                )
            params['cons']['Q'] *= -1e6

        with st.expander('Wärmequelle'):
            params.setdefault('setup', {})
            if supports_general_boundary_modes(hp_model_name):
                source_mode_options = {
                    'Fester Temperaturabfall': 'fixed_delta_T',
                    'Fester Massenstrom': 'fixed_mass_flow',
                }
                current_source_mode = get_source_mode_from_params(params)
                current_source_mode_label = next(
                    (
                        label for label, value in source_mode_options.items()
                        if value == current_source_mode
                    ),
                    'Fester Temperaturabfall'
                )
                selected_source_mode_label = st.radio(
                    'Wärmequelle',
                    options=list(source_mode_options.keys()),
                    index=list(source_mode_options.keys()).index(
                        current_source_mode_label
                    ),
                    key='source_mode_selector'
                )
                params['setup']['source_mode'] = source_mode_options[
                    selected_source_mode_label
                ]

            params['B1']['T'] = st.slider(
                'Temperatur Vorlauf', min_value=0, max_value=T_crit,
                value=params['B1']['T'], format='%d°C', key='T_heatsource_ff'
                )
            params['B1']['p'] = st.number_input(
                'Druck Wärmequelle [bar]', min_value=0.1, max_value=200.0,
                value=float(params['B1'].get('p', 1.013)),
                step=0.1, format='%.3f', key='p_heatsource'
            )
            if supports_general_boundary_modes(hp_model_name):
                params['setup']['use_source_pump'] = st.checkbox(
                    'Quellenpumpe modellieren',
                    value=bool(params['setup'].get('use_source_pump', True)),
                    key='use_source_pump_checkbox'
                )
                default_delta = float(get_source_delta_T_from_params(params))
                params['setup']['source_delta_T'] = st.number_input(
                    'Temperaturabfall Quelle [K]',
                    min_value=0.1,
                    max_value=100.0,
                    value=float(max(default_delta, 0.1)),
                    step=0.5,
                    format='%.1f',
                    key='source_delta_T_input'
                )
                params['B2']['T'] = (
                    float(params['B1']['T']) - float(params['setup']['source_delta_T'])
                )
                if params['setup']['source_mode'] == 'fixed_mass_flow':
                    params['setup']['m_source'] = st.number_input(
                        'Quellmassenstrom [kg/s]',
                        min_value=0.01,
                        max_value=500.0,
                        value=float(
                            params['setup'].get(
                                'm_source', params['B1'].get('m', 15.0)
                            )
                        ),
                        step=0.5,
                        format='%.2f',
                        key='m_source_input'
                    )
                    params['B1']['m'] = params['setup']['m_source']
                    st.caption(
                        'Der Quellruecklauf ist in diesem Modus ein '
                        + 'Rechenergebnis. Der Temperaturabfall dient als '
                        + 'Startwert fuer T_mid und die Initialisierung.'
                    )
                else:
                    st.caption(
                        f"Resultierender Quellruecklauf: {params['B2']['T']:.1f} °C"
                    )
            else:
                params['B2']['T'] = st.slider(
                    'Temperatur Rücklauf', min_value=0, max_value=T_crit,
                    value=params['B2']['T'], format='%d°C', key='T_heatsource_bf'
                    )

            invalid_temp_diff = params['B2']['T'] >= params['B1']['T']
            if invalid_temp_diff:
                st.error(
                    'Die Rücklauftemperatur muss niedriger sein, als die '
                    + 'Vorlauftemperatur.'
                    )
            params['setup']['waste_heat_further_usage'] = st.checkbox(
                'Waste heat further usage berücksichtigen',
                value=bool(params['setup'].get('waste_heat_further_usage', False)),
                help='Falls aktiviert, wird der Rücklauf der Wärmequelle in der '
                     'Exergieanalyse als weiter nutzbar betrachtet, sofern er '
                     'nicht unter die Umgebungstemperatur fällt.'
            )

        with st.expander('Wärmesenke'):
            T_max_sink = T_crit
            if 'trans' in hp_model_name or supports_general_boundary_modes(hp_model_name):
                T_max_sink = 200  # °C -- Ad hoc value, maybe find better one
            params.setdefault('C2', {})
            params.setdefault('C3', {})

            if supports_general_boundary_modes(hp_model_name):
                sink_mode_options = {
                    'Normales Heizen': 'sensible',
                    'Dampferzeugung': 'steam',
                }
                current_sink_mode = get_sink_mode_from_params(params)
                current_sink_mode_label = next(
                    (
                        label for label, value in sink_mode_options.items()
                        if value == current_sink_mode
                    ),
                    'Normales Heizen'
                )
                selected_sink_mode_label = st.radio(
                    'Senkenrandbedingung',
                    options=list(sink_mode_options.keys()),
                    index=list(sink_mode_options.keys()).index(
                        current_sink_mode_label
                    ),
                    key='sink_mode_selector'
                )
                params['setup']['sink_mode'] = sink_mode_options[
                    selected_sink_mode_label
                ]

                if params['setup']['sink_mode'] == 'steam':
                    params['setup']['use_sink_pump'] = False
                    params['setup']['T_steam'] = st.number_input(
                        'Dampftemperatur [°C]',
                        min_value=50.0,
                        max_value=200.0,
                        value=float(params['setup'].get('T_steam', 110.0)),
                        step=1.0,
                        format='%.1f',
                        key='T_steam_input'
                    )
                    params['setup']['m_steam'] = st.number_input(
                        'Dampfmassenstrom [kg/s]',
                        min_value=0.01,
                        max_value=50.0,
                        value=float(params['setup'].get('m_steam', 0.5)),
                        step=0.05,
                        format='%.2f',
                        key='m_steam_input'
                    )
                    sink_pressure = PSI(
                        'P', 'Q', 0,
                        'T', float(params['setup']['T_steam']) + 273.15,
                        params['fluids']['si']
                    ) * 1e-5
                    params['C1']['T'] = float(params['setup']['T_steam'])
                    params['C2']['T'] = float(params['setup']['T_steam'])
                    params['C3']['T'] = float(params['setup']['T_steam'])
                    params['C1']['p'] = sink_pressure
                    params['C2']['p'] = sink_pressure
                    params['C3']['p'] = sink_pressure
                    params['C1']['m'] = float(params['setup']['m_steam'])
                    st.caption(
                        f"Sattdruck bei {params['setup']['T_steam']:.1f} °C: "
                        + f'{sink_pressure:.3f} bar. Die Senkenpumpe ist in '
                        + 'diesem Modus deaktiviert.'
                    )
                else:
                    params['setup']['use_sink_pump'] = st.checkbox(
                        'Senkenpumpe modellieren',
                        value=bool(params['setup'].get('use_sink_pump', True)),
                        key='use_sink_pump_checkbox'
                    )
                    params['C3']['T'] = st.slider(
                        'Temperatur Vorlauf', min_value=0, max_value=T_max_sink,
                        value=params['C3']['T'], format='%d°C', key='T_consumer_ff'
                    )
                    params['C2']['T'] = params['C3']['T']
                    params['C1']['T'] = st.slider(
                        'Temperatur Rücklauf', min_value=0, max_value=T_max_sink,
                        value=params['C1']['T'], format='%d°C', key='T_consumer_bf'
                    )
                    sink_pressure = st.number_input(
                        'Druck Wärmesenke [bar]', min_value=0.1, max_value=200.0,
                        value=float(
                            params.get('C3', {}).get(
                                'p', params['C1'].get('p', 10.0)
                            )
                        ),
                        step=0.1, format='%.3f', key='p_consumer'
                    )
                    params['C3']['p'] = sink_pressure
                    params['C1']['p'] = sink_pressure
            else:
                params['C3']['T'] = st.slider(
                    'Temperatur Vorlauf', min_value=0, max_value=T_max_sink,
                    value=params['C3']['T'], format='%d°C', key='T_consumer_ff'
                )
                if 'C2' in params:
                    params['C2']['T'] = params['C3']['T']
                params['C1']['T'] = st.slider(
                    'Temperatur Rücklauf', min_value=0, max_value=T_max_sink,
                    value=params['C1']['T'], format='%d°C', key='T_consumer_bf'
                )
                sink_pressure = st.number_input(
                    'Druck Wärmesenke [bar]', min_value=0.1, max_value=200.0,
                    value=float(params.get('C3', {}).get('p', params['C1'].get('p', 10.0))),
                    step=0.1, format='%.3f', key='p_consumer'
                )
                params.setdefault('C3', {})
                params['C3']['p'] = sink_pressure
                params['C1']['p'] = sink_pressure

            if get_sink_mode_from_params(params) == 'sensible':
                invalid_temp_diff = params['C1']['T'] >= params['C3']['T']
                if invalid_temp_diff:
                    st.error(
                        'Die Rücklauftemperatur muss niedriger sein, als die '
                        + 'Vorlauftemperatur.'
                    )
            invalid_temp_diff = get_sink_hot_target_temp_from_params(params) <= params['B1']['T']
            if invalid_temp_diff:
                st.error(
                    'Die Temperatur der Wärmesenke muss höher sein, als die '
                    + 'der Wärmequelle.'
                )

        main_hex_ttd_targets = get_main_hex_ttd_targets(params)
        if main_hex_ttd_targets:
            with st.expander('Hauptwärmeübertrager'):
                default_global_ttd = params.get('setup', {}).get(
                    'global_ttd_main_hex'
                )
                if default_global_ttd is None:
                    default_global_ttd = params[
                        main_hex_ttd_targets[0][0]
                    ][main_hex_ttd_targets[0][1]]

                global_ttd = st.slider(
                    'Globaler minimaler Temperaturabstand',
                    min_value=0.5,
                    max_value=30.0,
                    value=float(default_global_ttd),
                    step=0.5,
                    format='%.1f K',
                    key='global_main_hex_ttd'
                )
                apply_global_main_hex_ttd(params, global_ttd)

                component_names = {
                    'evap': 'Verdampfer',
                    'cond': 'Kondensator',
                    'inter': 'Zwischenwärmeübertrager',
                    'trans': 'Gaskühler',
                    'econ': 'Economizer',
                    'econ1': 'Economizer 1',
                    'econ2': 'Economizer 2',
                }
                active_components = [
                    component_names.get(comp_key, comp_key)
                    for comp_key, _ in main_hex_ttd_targets
                ]
                st.caption(
                    'Der Wert wird auf folgende Hauptwärmeübertrager '
                    + 'angewendet: '
                    + ', '.join(active_components)
                    + '.'
                )

        if (
            supports_explicit_suction_superheat(hp_model)
            or supports_explicit_subcooling(hp_model)
            or supports_rip_factor(hp_model)
            or supports_explicit_injection_superheat(hp_model)
            or supports_a_target_calibration(hp_model_name)
        ):
            with st.expander('Erweiterte Kreisprozessparameter'):
                params.setdefault('setup', {})

                if supports_explicit_suction_superheat(hp_model):
                    params['setup']['dT_sup'] = st.slider(
                        'Verdichtersaugüberhitzung',
                        min_value=0.0,
                        max_value=30.0,
                        value=float(params['setup'].get('dT_sup', 0.0)),
                        step=0.5,
                        format='%.1f K',
                        key='dT_sup_cycle'
                    )

                if supports_explicit_subcooling(hp_model):
                    params['setup']['dT_sub'] = st.slider(
                        'Unterkühlung',
                        min_value=0.0,
                        max_value=30.0,
                        value=float(params['setup'].get('dT_sub', 0.0)),
                        step=0.5,
                        format='%.1f K',
                        key='dT_sub_cycle'
                    )
                    st.caption(
                        'Die Unterkühlung wird für subkritische '
                        + 'Ein-Kältemittel-Modelle ohne IHX berücksichtigt, '
                        + 'also z. B. auch bei Flashtank und geschlossenem '
                        + 'Economizer.'
                    )

                if (
                    supports_rip_factor(hp_model)
                    or supports_a_target_calibration(hp_model_name)
                ):
                    calibration_options = {'RIP direkt vorgeben': 'rip'}
                    if supports_a_target_calibration(hp_model_name):
                        calibration_options['A_target kalibrieren'] = 'a_target'

                    current_calibration_mode = params['setup'].get(
                        'calibration_mode', 'rip'
                    )
                    calibration_label = next(
                        (
                            label
                            for label, value in calibration_options.items()
                            if value == current_calibration_mode
                        ),
                        'RIP direkt vorgeben'
                    )
                    selected_calibration_label = st.radio(
                        'Literatur-Kalibrierung',
                        options=list(calibration_options.keys()),
                        index=list(calibration_options.keys()).index(
                            calibration_label
                        ),
                        key='cycle_calibration_mode'
                    )
                    params['setup']['calibration_mode'] = calibration_options[
                        selected_calibration_label
                    ]

                if (
                    supports_rip_factor(hp_model)
                    and params['setup'].get('calibration_mode', 'rip') != 'a_target'
                ):
                    params['setup']['rip_factor'] = st.slider(
                        'Zwischendruckfaktor RIP',
                        min_value=0.5,
                        max_value=1.5,
                        value=float(params['setup'].get('rip_factor', 1.0)),
                        step=0.01,
                        format='%.2f',
                        key='rip_factor_cycle'
                    )

                if (
                    supports_a_target_calibration(hp_model_name)
                    and params['setup'].get('calibration_mode') == 'a_target'
                ):
                    params['setup']['A_target'] = st.slider(
                        'Einspritzmassenanteil A_target',
                        min_value=0.05,
                        max_value=1.20,
                        value=float(params['setup'].get('A_target', 0.30)),
                        step=0.01,
                        format='%.2f',
                        key='a_target_cycle'
                    )

                if supports_explicit_injection_superheat(hp_model):
                    params['setup']['dT_sup_inj'] = st.slider(
                        'Einspritz-Überhitzung',
                        min_value=0.0,
                        max_value=30.0,
                        value=float(params['setup'].get('dT_sup_inj', 0.0)),
                        step=0.5,
                        format='%.1f K',
                        key='dT_sup_inj_cycle'
                    )

                caption_parts = []
                if supports_explicit_suction_superheat(hp_model):
                    caption_parts.append(
                        'Die Verdichtersaugüberhitzung beschreibt die zusätzliche '
                        'Erwärmung des Kältemittels vor dem ersten Verdichter.'
                    )
                if supports_explicit_subcooling(hp_model):
                    caption_parts.append(
                        'Die Unterkühlung beschreibt die zusätzliche Abkühlung '
                        'der Flüssigkeit nach dem Kondensator.'
                    )
                if supports_rip_factor(hp_model):
                    if params['setup'].get('calibration_mode', 'rip') == 'a_target':
                        caption_parts.append(
                            'Im Literaturmodus wird RIP automatisch so kalibriert, '
                            'dass der gewünschte Einspritzmassenanteil A_target '
                            'möglichst gut erreicht wird.'
                        )
                    else:
                        caption_parts.append(
                            'RIP = 1.00 entspricht dem geometrischen Mitteldruck. '
                            'Kleinere oder größere Werte verschieben die '
                            'Lastaufteilung zwischen erster und zweiter '
                            'Verdichterstufe. Zusätzliche Druckverluste in den '
                            'Bauteilen können das sichtbare Druckniveau weiter '
                            'verschieben.'
                        )
                if supports_explicit_injection_superheat(hp_model):
                    caption_parts.append(
                        'Die Einspritz-Überhitzung beschreibt den zusätzlichen '
                        'Temperaturabstand der Einspritzleitung zum Siedepunkt. '
                        'Sie wird aktuell nur für geschlossene Economizer-Topologien '
                        'angeboten.'
                    )
                if caption_parts:
                    st.caption(' '.join(caption_parts))

        if supports_general_boundary_modes(hp_model_name):
            with st.expander('Allgemeine Systemparameter'):
                params.setdefault('setup', {})
                params['setup']['motor_eta'] = st.slider(
                    'Motorwirkungsgrad',
                    min_value=50.0,
                    max_value=100.0,
                    value=float(params['setup'].get('motor_eta', 0.98) * 100),
                    step=0.5,
                    format='%.1f%%',
                    key='motor_eta_slider'
                ) / 100.0
                params['setup']['skip_ommen_check'] = st.checkbox(
                    'Ommen-Druckgrenzen ignorieren',
                    value=bool(params['setup'].get('skip_ommen_check', False)),
                    key='skip_ommen_check_checkbox'
                )

        if hp_model['nr_refrigs'] == 2:
            with st.expander('Zwischenwärmeübertrager'):
                params.setdefault('setup', {})
                (
                    t_source_cold, t_sink_hot, t_mid_min, t_mid_max
                ) = get_cascade_t_mid_bounds(params)

                t_mid_min = float(np.round(t_mid_min, 1))
                t_mid_max = float(np.round(t_mid_max, 1))

                if t_mid_max <= t_mid_min:
                    st.warning(
                        'Für die aktuellen Quell- und Senkentemperaturen '
                        + 'konnte kein sinnvoller Bereich für T_mid bestimmt '
                        + 'werden.'
                    )
                else:
                    if supports_general_boundary_modes(hp_model_name):
                        split_mode_options = {
                            'T_mid direkt': 't_mid',
                            'Lift Share': 'lift_share',
                        }
                        current_split_mode = str(
                            params['setup'].get('cascade_split_mode', 't_mid')
                        )
                        current_split_mode_label = next(
                            (
                                label
                                for label, value in split_mode_options.items()
                                if value == current_split_mode
                            ),
                            'T_mid direkt'
                        )
                        selected_split_mode_label = st.radio(
                            'Aufteilung Temperaturhub',
                            options=list(split_mode_options.keys()),
                            index=list(split_mode_options.keys()).index(
                                current_split_mode_label
                            ),
                            key='cascade_split_mode_selector'
                        )
                        params['setup']['cascade_split_mode'] = (
                            split_mode_options[selected_split_mode_label]
                        )

                    if params['setup'].get('cascade_split_mode', 't_mid') == 'lift_share':
                        source_hot = float(params['B1']['T'])
                        inter_ttd = float(params.get('inter', {}).get('ttd_u', 0.0))
                        gross_lift = max(t_sink_hot - source_hot, 1e-9)
                        lift_max = (
                            t_sink_hot - inter_ttd / 2.0 - source_hot
                        ) / gross_lift
                        lift_max = float(min(max(lift_max, 0.0), 0.99))
                        if lift_max <= 0:
                            st.warning(
                                'Für die aktuellen Randbedingungen konnte kein '
                                + 'gueltiger Lift-Share-Bereich bestimmt werden.'
                            )
                        else:
                            default_lift_share = float(
                                params['setup'].get('lift_share', 0.5)
                            )
                            params['setup']['lift_share'] = st.slider(
                                'Lift Share',
                                min_value=0.0,
                                max_value=lift_max,
                                value=float(
                                    np.clip(default_lift_share, 0.0, lift_max)
                                ),
                                step=0.01,
                                format='%.2f',
                                key='lift_share_slider'
                            )
                            params['setup']['T34'] = (
                                source_hot
                                + params['setup']['lift_share']
                                * (t_sink_hot - source_hot)
                            )
                            params['setup']['t_mid'] = (
                                params['setup']['T34'] + inter_ttd / 2.0
                            )
                            params['setup']['t_mid_fraction'] = (
                                (params['setup']['t_mid'] - t_source_cold)
                                / max(t_sink_hot - t_source_cold, 1e-9)
                            )
                            st.caption(
                                f"T_34 = {params['setup']['T34']:.2f} °C, "
                                + f"T_mid = {params['setup']['t_mid']:.2f} °C, "
                                + f"α = {params['setup']['t_mid_fraction']:.3f}."
                            )
                    else:
                        default_t_mid = params['setup'].get('t_mid')
                        if default_t_mid is None:
                            default_fraction = float(
                                params['setup'].get('t_mid_fraction', 0.5)
                            )
                            default_t_mid = (
                                t_source_cold
                                + default_fraction * (t_sink_hot - t_source_cold)
                            )

                        default_t_mid = float(
                            np.clip(default_t_mid, t_mid_min, t_mid_max)
                        )

                        params['setup']['t_mid'] = st.slider(
                            'Mittlere Temperatur T_mid',
                            min_value=t_mid_min,
                            max_value=t_mid_max,
                            value=default_t_mid,
                            step=0.5,
                            format='%.1f°C',
                            key='T_mid'
                        )

                        params['setup']['t_mid_fraction'] = (
                            (params['setup']['t_mid'] - t_source_cold)
                            / max(t_sink_hot - t_source_cold, 1e-9)
                        )
                        source_hot = float(params['B1']['T'])
                        inter_ttd = float(params.get('inter', {}).get('ttd_u', 0.0))
                        params['setup']['T34'] = (
                            params['setup']['t_mid'] - inter_ttd / 2.0
                        )
                        params['setup']['lift_share'] = (
                            (params['setup']['T34'] - source_hot)
                            / max(t_sink_hot - source_hot, 1e-9)
                        )

                        st.caption(
                            'Die Eingabe bestimmt die Lage von T_mid zwischen '
                            + 'kaltem Quellende und warmem Senkenende. Je größer '
                            + 'α, desto höher liegt T_mid und desto mehr '
                            + 'Temperaturhub übernimmt der Niedertemperaturkreis. '
                            + f"Aktuell: α = {params['setup']['t_mid_fraction']:.3f}, "
                            + f"Lift Share = {params['setup']['lift_share']:.3f}."
                        )

        with st.expander('Verdichter'):
            nr_refrigs = hp_model['nr_refrigs']
            if hp_model['comp_var'] is None and nr_refrigs == 1:
                params['comp']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_s$', min_value=0, max_value=100,
                    step=1, value=int(params['comp']['eta_s']*100),
                    format='%d%%'
                    ) / 100
            elif hp_model['comp_var'] is not None and nr_refrigs == 1:
                params['comp1']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_{s,1}$', min_value=0, max_value=100,
                    step=1, value=int(params['comp1']['eta_s']*100),
                    format='%d%%'
                    ) / 100
                params['comp2']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_{s,2}$', min_value=0, max_value=100,
                    step=1, value=int(params['comp2']['eta_s']*100),
                    format='%d%%'
                    ) / 100
            elif hp_model['comp_var'] is None and nr_refrigs == 2:
                params['HT_comp']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_{s,HTK}$', min_value=0, max_value=100,
                    step=1, value=int(params['HT_comp']['eta_s']*100),
                    format='%d%%'
                    ) / 100
                params['LT_comp']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_{s,NTK}$', min_value=0, max_value=100,
                    step=1, value=int(params['LT_comp']['eta_s']*100),
                    format='%d%%'
                    ) / 100
            elif hp_model['comp_var'] is not None and nr_refrigs == 2:
                params['HT_comp1']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_{s,HTK,1}$', min_value=0,
                    max_value=100, step=1, 
                    value=int(params['HT_comp1']['eta_s']*100), format='%d%%'
                    ) / 100
                params['HT_comp2']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_{s,HTK,2}$', min_value=0,
                    max_value=100, step=1,
                    value=int(params['HT_comp2']['eta_s']*100), format='%d%%'
                    ) / 100
                params['LT_comp1']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_{s,NTK,1}$', min_value=0,
                    max_value=100, step=1,
                    value=int(params['LT_comp1']['eta_s']*100), format='%d%%'
                    ) / 100
                params['LT_comp2']['eta_s'] = st.slider(
                    'Wirkungsgrad $\eta_{s,NTK,2}$', min_value=0,
                    max_value=100, step=1,
                    value=int(params['LT_comp2']['eta_s']*100), format='%d%%'
                    ) / 100

        with st.expander('Umgebungsbedingungen (Exergie)'):
            params['ambient']['T'] = st.slider(
                'Temperatur', min_value=1, max_value=45, step=1,
                value=params['ambient']['T'], format='%d°C', key='T_env'
                )
            params['ambient']['p'] = st.number_input(
                'Druck in bar', value=float(params['ambient']['p']), step=0.01,
                format='%.4f', key='p_env'
                )
        
        with st.expander('Parameter zur Kostenkalkulation'):
            costcalcparams = {}

            cepcipath = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 'models', 'input', 'CEPCI.json'
                ))
            with open(cepcipath, 'r', encoding='utf-8') as file:
                cepci = json.load(file)

            costcalcparams['cost_method'] = st.selectbox(
                'Kostenmethodik',
                options=list(COST_METHOD_OPTIONS.keys()),
                format_func=lambda x: COST_METHOD_OPTIONS[x],
                key='cost_method',
                **_session_index_kwargs(
                    'cost_method',
                    list(COST_METHOD_OPTIONS.keys()).index(
                        ss.get('cost_method', 'standard')
                        if ss.get('cost_method', 'standard') in COST_METHOD_OPTIONS
                        else 'standard'
                    )
                )
            )

            repo_cost_mode = costcalcparams['cost_method'] == 'repo_hthp'
            if repo_cost_mode:
                pass
            else:
                st.caption(
                    'Das CEPCI-Referenzjahr ist fest auf 2015 gesetzt. '
                    + 'Im Dashboard wird nur das aktuelle Kostenjahr ausgewählt.'
                )

            costcalcparams['current_year'] = st.selectbox(
                'CEPCI-Kostenindexjahr' if repo_cost_mode else 'Jahr der Kostenkalkulation',
                options=sorted(list(cepci.keys()), reverse=True),
                key='current_year',
                **_session_index_kwargs(
                    'current_year',
                    0 if '2025' not in cepci else sorted(list(cepci.keys()), reverse=True).index('2025')
                )
            )

            if repo_cost_mode:
                costcalcparams['analysis_year'] = st.number_input(
                    'Analysejahr',
                    min_value=1980, max_value=2100, step=1,
                    key='analysis_year',
                    **_session_value_kwargs('analysis_year', 2026)
                )
            else:
                costcalcparams['analysis_year'] = int(costcalcparams['current_year'])

            costcalcparams['elec_price_cent_kWh'] = st.number_input(
                'Strompreis [ct/kWh]',
                min_value=0.0, max_value=200.0, step=1.0,
                key='elec_price_cent_kWh',
                **_session_value_kwargs('elec_price_cent_kWh', 40.0)
            )

            costcalcparams['b1_cost_eur_per_GJ'] = st.number_input(
                'Kosten Feedwater B1 [EUR/GJ]',
                min_value=0.0, max_value=10000.0, step=0.1,
                format='%.2f',
                key='b1_cost_eur_per_GJ',
                **_session_value_kwargs('b1_cost_eur_per_GJ', 0.0)
            )

            costcalcparams['tau_h_per_year'] = st.number_input(
                'Volllaststunden [h/a]',
                min_value=0.0, max_value=9000.0, step=100.0,
                key='tau_h_per_year',
                **_session_value_kwargs('tau_h_per_year', 5500.0)
            )

            costcalcparams['usd_to_eur'] = st.number_input(
                'Währungsumrechnung USD → EUR [-]',
                min_value=0.1, max_value=5.0, step=0.01,
                format='%.2f',
                key='usd_to_eur',
                **_session_value_kwargs('usd_to_eur', 0.93)
            )

            costcalcparams['hex_cost_model'] = st.selectbox(
                'PEC Wärmeübertrager',
                options=list(PEC_HEX_OPTIONS.keys()),
                format_func=lambda x: PEC_HEX_OPTIONS[x]['label'],
                key='hex_cost_model',
                **_session_index_kwargs(
                    'hex_cost_model',
                    list(PEC_HEX_OPTIONS.keys()).index(
                        ss.get('hex_cost_model', 'ommen')
                        if ss.get('hex_cost_model', 'ommen') in PEC_HEX_OPTIONS
                        else 'ommen'
                    )
                )
            )

            costcalcparams['compressor_cost_model'] = st.selectbox(
                'PEC Verdichter',
                options=list(PEC_COMP_OPTIONS.keys()),
                format_func=lambda x: PEC_COMP_OPTIONS[x]['label'],
                key='compressor_cost_model',
                **_session_index_kwargs(
                    'compressor_cost_model',
                    list(PEC_COMP_OPTIONS.keys()).index(
                        ss.get('compressor_cost_model', 'ommen')
                        if ss.get('compressor_cost_model', 'ommen') in PEC_COMP_OPTIONS
                        else 'ommen'
                    )
                )
            )

            costcalcparams['flash_cost_model'] = st.selectbox(
                'PEC Flashtank',
                options=list(PEC_FLASH_OPTIONS.keys()),
                format_func=lambda x: PEC_FLASH_OPTIONS[x]['label'],
                key='flash_cost_model',
                **_session_index_kwargs(
                    'flash_cost_model',
                    list(PEC_FLASH_OPTIONS.keys()).index(
                        ss.get('flash_cost_model', 'ommen')
                        if ss.get('flash_cost_model', 'ommen') in PEC_FLASH_OPTIONS
                        else 'ommen'
                    )
                )
            )
            st.caption('Die Pumpenkorrelation ist aktuell fest auf Shamoushaki et al. gesetzt.')

            if repo_cost_mode:
                costcalcparams['include_pumps_in_pec'] = st.checkbox(
                    'Pumpen in TCI berücksichtigen',
                    key='include_pumps_in_pec',
                    **_session_value_kwargs('include_pumps_in_pec', True)
                )
                costcalcparams['hx_area_method'] = st.selectbox(
                    'Repo-HX-Flächenansatz',
                    options=['q_lmtd', 'tespy_ka'],
                    format_func=lambda x: (
                        'Q / (U * LMTD)' if x == 'q_lmtd'
                        else 'TESPy kA / U'
                    ),
                    key='hx_area_method',
                    **_session_index_kwargs(
                        'hx_area_method',
                        1 if ss.get('hx_area_method', 'q_lmtd') == 'tespy_ka' else 0
                    )
                )
                costcalcparams['compressor_eta_vol'] = float(
                    ss.get('compressor_eta_vol', 1.0)
                )
                costcalcparams['compressor_eta_vol_lt'] = st.number_input(
                    'Volumetrischer Wirkungsgrad LT-Verdichter η_vol,LT [-]',
                    min_value=0.1, max_value=1.5, step=0.01,
                    format='%.2f',
                    key='compressor_eta_vol_lt',
                    **_session_value_kwargs(
                        'compressor_eta_vol_lt',
                        float(ss.get('compressor_eta_vol', 1.0))
                    )
                )
                costcalcparams['compressor_eta_vol_ht'] = st.number_input(
                    'Volumetrischer Wirkungsgrad HT-Verdichter η_vol,HT [-]',
                    min_value=0.1, max_value=1.5, step=0.01,
                    format='%.2f',
                    key='compressor_eta_vol_ht',
                    **_session_value_kwargs(
                        'compressor_eta_vol_ht',
                        float(ss.get('compressor_eta_vol', 1.0))
                    )
                )
            else:
                costcalcparams['include_pumps_in_pec'] = True
                costcalcparams['hx_area_method'] = 'q_lmtd'
                costcalcparams['compressor_eta_vol'] = 1.0
                costcalcparams['compressor_eta_vol_lt'] = 1.0
                costcalcparams['compressor_eta_vol_ht'] = 1.0

            costcalcparams['k_evap'] = st.slider(
                'Wärmedurchgangskoeffizient (Verdampfung)',
                min_value=0, max_value=5000, step=10,
                format='%d W/m²K', key='k_evap',
                **_session_value_kwargs('k_evap', 1500)
                )

            costcalcparams['k_cond'] = st.slider(
                'Wärmedurchgangskoeffizient (Verflüssigung)',
                min_value=0, max_value=5000, step=10,
                format='%d W/m²K', key='k_cond',
                **_session_value_kwargs('k_cond', 3500)
                )

            if hp_model['nr_refrigs'] == 2:
                costcalcparams['k_inter'] = st.slider(
                    'Wärmedurchgangskoeffizient (Zwischenwärmeübertrager)',
                    min_value=0, max_value=5000, step=10,
                    format='%d W/m²K', key='k_inter',
                    **_session_value_kwargs('k_inter', 2200)
                )
            else:
                costcalcparams['k_inter'] = int(ss.get('k_inter', 2200))

            if 'ihx' in hp_model_name:
                costcalcparams['k_ihx'] = st.slider(
                    'Wärmedurchgangskoeffizient (Interner Wärmeübertrager, IHX)',
                    min_value=0, max_value=5000, step=10,
                    format='%d W/m²K', key='k_ihx',
                    **_session_value_kwargs('k_ihx', 1500)
                )
            else:
                costcalcparams['k_ihx'] = int(ss.get('k_ihx', 1500))

            if 'trans' in hp_model_name:
                costcalcparams['k_trans'] = st.slider(
                    'Wärmedurchgangskoeffizient (transkritisch)',
                    min_value=0, max_value=1000, step=5,
                    format='%d W/m²K', key='k_trans',
                    **_session_value_kwargs('k_trans', 60)
                    )

            costcalcparams['k_misc'] = st.slider(
                'Wärmedurchgangskoeffizient (Sonstige)',
                min_value=0, max_value=1000, step=5,
                format='%d W/m²K', key='k_misc',
                **_session_value_kwargs('k_misc', 50)
                )

            costcalcparams['residence_time'] = st.slider(
                'Verweildauer Flashtank',
                min_value=0, max_value=60, step=1,
                format='%d s', key='residence_time',
                **_session_value_kwargs('residence_time', 10)
                )

            st.markdown('**Weitere Parameter für die exergoökonomische Analyse**')
            st.number_input(
                'Effektiver Zinssatz i_eff [-]',
                min_value=0.0, max_value=1.0, step=0.005,
                format='%.3f', key='i_eff',
                **_session_value_kwargs('i_eff', 0.08)
            )
            st.number_input(
                'Nutzungsdauer n [a]',
                min_value=1, max_value=100, step=1,
                key='n',
                **_session_value_kwargs('n', 20)
            )
            st.number_input(
                'Relative O&M-Kosten f_O&M [-]',
                min_value=0.0, max_value=1.0, step=0.005,
                format='%.3f', key='omc_rel',
                **_session_value_kwargs('omc_rel', 0.03)
            )
            if repo_cost_mode:
                st.number_input(
                    'Preissteigerungsrate O&M / allgemeine Inflation R_N_OM [-]',
                    min_value=0.0, max_value=1.0, step=0.005,
                    format='%.3f', key='r_n_om',
                    **_session_value_kwargs('r_n_om', 0.02)
                )
                st.number_input(
                    'Preissteigerungsrate Strom R_N_EL [-]',
                    min_value=0.0, max_value=1.0, step=0.005,
                    format='%.3f', key='r_n_el',
                    **_session_value_kwargs('r_n_el', 0.02)
                )
                st.number_input(
                    'Installationsfaktor F_install [-]',
                    min_value=0.0, max_value=50.0, step=0.1,
                    format='%.2f', key='install_factor',
                    **_session_value_kwargs('install_factor', 4.16)
                )
            else:
                st.number_input(
                    'Preissteigerungsrate r_n [-]',
                    min_value=0.0, max_value=1.0, step=0.005,
                    format='%.3f', key='r_n',
                    **_session_value_kwargs('r_n', 0.02)
                )
                st.number_input(
                    'TCI-Faktor [-]',
                    min_value=0.0, max_value=50.0, step=0.1,
                    format='%.2f', key='tci_factor',
                    **_session_value_kwargs('tci_factor', 6.32)
                )

        kosmadakis_params = {}
        with st.expander('Parameter für Projektkostenabschätzung nach Kosmadakis et al. (2020)'):
            kosmadakis_params['gas_price_cent_kWh'] = st.number_input(
                'Gaspreis [ct/kWh]',
                min_value=0.0, max_value=200.0, step=0.5,
                key='gas_price_cent_kWh',
                **_session_value_kwargs('gas_price_cent_kWh', 8.0)
            )

            kosmadakis_params['gas_boiler_efficiency'] = st.number_input(
                'Wirkungsgrad Ersatz-Gaskessel [-]',
                min_value=0.1, max_value=1.0, step=0.01,
                format='%.2f',
                key='gas_boiler_efficiency',
                **_session_value_kwargs('gas_boiler_efficiency', 0.90)
            )

            kosmadakis_params['refrigerant_charge_kg'] = st.number_input(
                'Kältemittelfüllmenge gesamt [kg]',
                min_value=0.0, max_value=100000.0, step=1.0,
                key='refrigerant_charge_kg',
                **_session_value_kwargs('refrigerant_charge_kg', 0.0)
            )

            kosmadakis_params['refrigerant_price_eur_kg'] = st.number_input(
                'Kältemittelpreis [€/kg]',
                min_value=0.0, max_value=10000.0, step=1.0,
                key='refrigerant_price_eur_kg',
                **_session_value_kwargs('refrigerant_price_eur_kg', 50.0)
            )

            kosmadakis_params['kos_piping_factor'] = st.number_input(
                'Faktor Rohrleitungen C_p-t / PEC [-]',
                min_value=0.0, max_value=10.0, step=0.01,
                format='%.2f',
                key='kos_piping_factor',
                **_session_value_kwargs('kos_piping_factor', 0.10)
            )

            kosmadakis_params['kos_electrical_factor'] = st.number_input(
                'Faktor elektrische Installation C_el^CI / PEC [-]',
                min_value=0.0, max_value=10.0, step=0.01,
                format='%.2f',
                key='kos_electrical_factor',
                **_session_value_kwargs('kos_electrical_factor', 0.10)
            )

            kosmadakis_params['kos_project_factor'] = st.number_input(
                'Projektkostenfaktor [-]',
                min_value=0.0, max_value=50.0, step=0.1,
                format='%.2f',
                key='kos_project_factor',
                **_session_value_kwargs('kos_project_factor', 4.16)
            )

            kosmadakis_params['kos_om_factor'] = st.number_input(
                'O&M-Faktor Projektkosten [-]',
                min_value=0.0, max_value=1.0, step=0.005,
                format='%.3f',
                key='kos_om_factor',
                **_session_value_kwargs('kos_om_factor', 0.02)
            )

            kosmadakis_params['kos_discount_rate'] = st.number_input(
                'Diskontsatz Amortisationszeit [-]',
                min_value=0.0, max_value=1.0, step=0.005,
                format='%.3f',
                key='kos_discount_rate',
                **_session_value_kwargs('kos_discount_rate', 0.05)
            )

        ss.costcalcparams = dict(costcalcparams)
        ss.kosmadakis_params = dict(kosmadakis_params)
        ss.econ_ui_params = {
            'i_eff': float(ss.get('i_eff', 0.08)),
            'r_n': float(ss.get('r_n', 0.02)),
            'r_n_om': float(ss.get('r_n_om', ss.get('r_n', 0.02))),
            'r_n_el': float(ss.get('r_n_el', ss.get('r_n', 0.02))),
            'n': int(ss.get('n', 20)),
            'omc_rel': float(ss.get('omc_rel', 0.03)),
            'tci_factor': float(ss.get('tci_factor', 6.32)),
            'install_factor': float(ss.get('install_factor', 4.16)),
        }
        ss.hp_params = params

        run_sim = st.button('🧮 Auslegung ausführen')
        # run_sim = True
    # autorun = st.checkbox('AutoRun Simulation', value=True)

    # %% MARK: Offdesign
    if mode == 'Teillast' and 'hp' in ss:
        params = ss.hp_params
        st.header('Teillastsimulation der Wärmepumpe')

        with st.expander('Teillast'):
            (params['offdesign']['partload_min'],
             params['offdesign']['partload_max']) = st.slider(
                'Bezogen auf Nennmassenstrom',
                min_value=0, max_value=120, step=5,
                value=(30, 100), format='%d%%', key='pl_slider'
                )

            params['offdesign']['partload_min'] /= 100
            params['offdesign']['partload_max'] /= 100

            params['offdesign']['partload_steps'] = int(np.ceil(
                    (params['offdesign']['partload_max']
                     - params['offdesign']['partload_min'])
                    / 0.1
                    ) + 1)

        with st.expander('Wärmequelle'):
            type_hs = st.radio(
                'Wärmequelle', ('Konstant', 'Variabel'), index=1,
                horizontal=True, key='temp_hs', label_visibility='hidden'
                )
            if type_hs == 'Konstant':
                params['offdesign']['T_hs_ff_start'] = (
                    ss.hp.params['B1']['T']
                    )
                params['offdesign']['T_hs_ff_end'] = (
                    params['offdesign']['T_hs_ff_start'] + 1
                    )
                params['offdesign']['T_hs_ff_steps'] = 1

                text = (
                    f'Temperatur <p style="color:{var.st_color_hex}">'
                    + f'{params["offdesign"]["T_hs_ff_start"]} °C'
                    + r'</p>'
                    )
                st.markdown(text, unsafe_allow_html=True)

            elif type_hs == 'Variabel':
                params['offdesign']['T_hs_ff_start'] = st.slider(
                    'Starttemperatur',
                    min_value=0, max_value=ss.T_crit, step=1,
                    value=int(
                        ss.hp.params['B1']['T']
                        - 5
                        ),
                    format='%d°C', key='T_hs_ff_start_slider'
                    )
                params['offdesign']['T_hs_ff_end'] = st.slider(
                    'Endtemperatur',
                    min_value=0, max_value=ss.T_crit, step=1,
                    value=int(
                        ss.hp.params['B1']['T']
                        + 5
                        ),
                    format='%d°C', key='T_hs_ff_end_slider'
                    )
                params['offdesign']['T_hs_ff_steps'] = int(np.ceil(
                    (params['offdesign']['T_hs_ff_end']
                     - params['offdesign']['T_hs_ff_start'])
                    / 3
                    ) + 1)

        with st.expander('Wärmesenke'):
            type_cons = st.radio(
                'Wärmesenke', ('Konstant', 'Variabel'), index=1,
                horizontal=True, key='temp_cons', label_visibility='hidden'
                )
            if type_cons == 'Konstant':
                params['offdesign']['T_cons_ff_start'] = (
                    ss.hp.params['C3']['T']
                    )
                params['offdesign']['T_cons_ff_end'] = (
                    params['offdesign']['T_cons_ff_start'] + 1
                    )
                params['offdesign']['T_cons_ff_steps'] = 1

                text = (
                    f'Temperatur <p style="color:{var.st_color_hex}">'
                    + f'{params["offdesign"]["T_cons_ff_start"]} °C'
                    + r'</p>'
                    )
                st.markdown(text, unsafe_allow_html=True)

            elif type_cons == 'Variabel':
                params['offdesign']['T_cons_ff_start'] = st.slider(
                    'Starttemperatur',
                    min_value=0, max_value=ss.T_crit, step=1,
                    value=int(
                        ss.hp.params['C3']['T']
                        - 10
                        ),
                    format='%d°C', key='T_cons_ff_start_slider'
                    )
                params['offdesign']['T_cons_ff_end'] = st.slider(
                    'Endtemperatur',
                    min_value=0, max_value=ss.T_crit, step=1,
                    value=int(
                        ss.hp.params['C3']['T']
                        + 10
                        ),
                    format='%d°C', key='T_cons_ff_end_slider'
                    )
                params['offdesign']['T_cons_ff_steps'] = int(np.ceil(
                    (params['offdesign']['T_cons_ff_end']
                     - params['offdesign']['T_cons_ff_start'])
                    / 1
                    ) + 1)

        ss.hp_params = params
        run_pl_sim = st.button('🧮 Teillast simulieren')

# %% MARK: Main Content
st.title('*heatpumps*')
st.caption('Exergoeconomic analysis with ExerPy')

if mode == 'Start':
    # %% MARK: Landing Page
    st.markdown(
        """
        <style>
        .hp-hero {
            padding: 1.4rem 1.5rem;
            border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(16, 88, 136, 0.16), rgba(17, 138, 178, 0.08)),
                linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
            border: 1px solid rgba(120, 150, 170, 0.25);
            margin-bottom: 1rem;
        }
        .hp-hero h2 {
            margin: 0 0 0.4rem 0;
            font-size: 1.8rem;
            line-height: 1.2;
        }
        .hp-hero p {
            margin: 0.45rem 0;
            font-size: 1rem;
        }
        .hp-card {
            padding: 1rem 1rem 0.85rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(120, 150, 170, 0.22);
            background: rgba(127, 127, 127, 0.06);
            min-height: 215px;
        }
        .hp-card h4 {
            margin: 0 0 0.45rem 0;
            font-size: 1.05rem;
        }
        .hp-card p {
            margin: 0.35rem 0;
            font-size: 0.96rem;
        }
        .hp-card ul {
            margin: 0.45rem 0 0 1rem;
            padding: 0;
        }
        .hp-card li {
            margin: 0.2rem 0;
        }
        .hp-band {
            padding: 1rem 1.1rem;
            border-left: 4px solid rgba(16, 88, 136, 0.7);
            background: rgba(16, 88, 136, 0.08);
            border-radius: 10px;
            margin: 1rem 0 1.25rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hp-hero">
            <h2>Thermodynamische Simulation und Exergoökonomie</h2>
            <p>
                Der Wärmepumpensimulator <strong>heatpumps</strong> ist eine
                leistungsfähige Simulationssoftware zur Analyse und Bewertung
                von Wärmepumpen.
            </p>
            <p>
                Mit diesem Dashboard lassen sich komplexe thermodynamische
                Anlagenmodelle über eine einfache Oberfläche steuern. Neben der
                Auslegung und stationären Teillastsimulation werden auch
                Zustandsgrößen, COP, Komponentenaufwand und wirtschaftliche
                Kenngrößen der Wärmepumpe transparent ausgewertet.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="hp-card">
                <h4>Simulation</h4>
                <p>Numerische Auslegung und stationäre Teillastanalyse für eine
                breite Auswahl gängiger Wärmepumpentopologien.</p>
                <ul>
                    <li>Sub- und transkritische Prozesse</li>
                    <li>Kaskaden, Economizer, Flash-Tank, IHX und mehr</li>
                    <li>Vergleich verschiedener Randbedingungen und Arbeitsmedien</li>
                    <li>Wärmesenke als expliziter Wasserstrang statt vereinfachter Verbrauchergrenze</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="hp-card">
                <h4>Exergie und Exergoökonomie</h4>
                <p>Zusätzlich zur thermodynamischen Auslegung werden
                Ineffizienzen, Exergieverluste und Kostenstrukturen auf
                Komponentenebene sichtbar gemacht.</p>
                <ul>
                    <li>Exergieanalyse mit ExerPy für Gesamtanlage und Komponenten</li>
                    <li>Exergoökonomische Bewertung mit ExerPy</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="hp-band">
            <strong>ExerPy im Dashboard</strong><br>
            Dieses Dashboard nutzt
            <a href="https://exerpy.readthedocs.io/en/latest/">ExerPy</a>
            für die Exergie- und Exergoökonomieauswertung. ExerPy ist eine
            Python-Bibliothek zur automatisierten Exergieanalyse von
            Energieumwandlungssystemen und ergänzt den TESPy-basierten
            Modellierungsansatz um konsistente Exergiebilanzen und
            kostenbezogene Auswertungen. In dieser Implementierung wurden
            dafür die Topologien gegenüber dem ursprünglichen Stand gezielt
            erweitert: elektrische Leistungsströme werden über
            <strong>PowerBus</strong> und <strong>PowerConnection</strong>
            der aktuellen TESPy-Version abgebildet, die Wärmesenke ist als
            expliziter Verbraucherstrang modelliert, und darauf aufbauend
            sind ExerPy-gestützte Exergie- und exergoökonomische Analysen
            in das Dashboard integriert.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Key Features

        - Stationäre Auslegungs- und Teillastsimulation basierend auf
          [TESPy](https://github.com/oemof/tespy)
        - Parametrisierung und Ergebnisvisualisierung über ein
          [Streamlit](https://github.com/streamlit/streamlit) Dashboard
        - In Industrie, Forschung und Entwicklung gängige Schaltungstopologien
        - Sub- und transkritische Prozesse
        - Große Auswahl an Arbeitsmedien durch die Integration von
          [CoolProp](https://github.com/CoolProp/CoolProp)
        - Integration von `PowerBus` und `PowerConnection` aus aktuellen
          TESPy-Versionen
        - Explizite Modellierung der Wärmesenke als Wasserstrang
        - Exergie- und exergoökonomische Bewertung mit
          [ExerPy](https://exerpy.readthedocs.io/en/latest/)
        """
    )

    st.button('Auslegung starten', on_click=switch2design)

    st.divider()

    with st.expander('Verwendete Software'):
        st.info(
            """
            #### Verwendete Software:

            Zur Modellerstellung und Berechnung der Simulationen wird die
            Open Source Software TESPy verwendet. Des Weiteren werden
            eine Reihe weiterer Pythonpakete zur Datenverarbeitung,
            -aufbereitung und -visualisierung genutzt.

            ---

            #### TESPy:

            TESPy (Thermal Engineering Systems in Python) ist ein
            leistungsfähiges Simulationswerkzeug für thermische
            Verfahrenstechnik, zum Beispiel für Kraftwerke,
            Fernwärmesysteme oder Wärmepumpen. Mit dem TESPy-Paket ist es
            möglich, Anlagen auszulegen und den stationären Betrieb zu
            simulieren. Danach kann das Teillastverhalten anhand der
            zugrundeliegenden Charakteristiken für jede Komponente der
            Anlage ermittelt werden. Die komponentenbasierte Struktur in
            Kombination mit der Lösungsmethode bieten eine sehr hohe
            Flexibilität hinsichtlich der Anlagentopologie und der
            Parametrisierung. Weitere Informationen zu TESPy sind in dessen
            [Onlinedokumentation](https://tespy.readthedocs.io) in
            englischer Sprache zu finden.

            #### Weitere Pakete:

            - [Streamlit](https://docs.streamlit.io) (Graphische Oberfläche)
            - [NumPy](https://numpy.org) (Datenverarbeitung)
            - [pandas](https://pandas.pydata.org) (Datenverarbeitung)
            - [SciPy](https://scipy.org/) (Interpolation)
            - [scikit-learn](https://scikit-learn.org) (Regression)
            - [Matplotlib](https://matplotlib.org) (Datenvisualisierung)
            - [FluProDia](https://fluprodia.readthedocs.io)
            (Datenvisualisierung)
            - [CoolProp](http://www.coolprop.org) (Stoffdaten)
            - [ExerPy](https://exerpy.readthedocs.io/en/latest/)
            (Exergie- und Exergoökonomieanalyse)
            """
            )

    with st.expander('Disclaimer'):
        st.warning(
            """
            #### Simulationsergebnisse:

            Numerische Simulationen sind Berechnungen mittels geeigneter
            Iterationsverfahren in Bezug auf die vorgegebenen und gesetzten
            Randbedingungen und Parameter. Eine Berücksichtigung aller
            möglichen Einflüsse ist in Einzelfällen nicht möglich, so dass
            Abweichungen zu Erfahrungswerten aus Praxisanwendungen
            entstehen können und bei der Bewertung berücksichtigt werden
            müssen. Die Ergebnisse geben hinreichenden bis genauen
            Aufschluss über das prinzipielle Verhalten, den COP und
            Zustandsgrößen in den einzelnen Komponenten der Wärmepumpe.
            Dennoch sind alle Angaben und Ergebnisse ohne Gewähr.
            """
            )

    with st.expander('Copyright'):

        st.success(
            """
            #### Softwarelizenz
            MIT License

            Copyright © 2023 Jonas Freißmann and Malte Fritz

            Permission is hereby granted, free of charge, to any person
            obtaining a copy of this software and associated documentation
            files (the "Software"), to deal in the Software without
            restriction, including without limitation the rights to use, copy,
            modify, merge, publish, distribute, sublicense, and/or sell copies
            of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be
            included in all copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
            EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
            MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
            NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
            BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
            ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
            CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            """
        )

if mode == 'Auslegung':
    # %% MARK: Design Simulation
    if not run_sim:
        # %% Topology & Refrigerant
        col_left, col_right = st.columns([1, 4])

        with col_left:
            st.subheader('Topologie')
            top_file = resolve_topology_svg_path(
                src_path,
                hp_model_name_topology,
                is_dark=is_dark,
            )
            if top_file is not None:
                st.image(top_file)
            else:
                st.info('Keine statische Topologie-SVG fuer dieses Modell vorhanden.')

        with col_right:
            st.subheader('Kältemittel')

            if hp_model['nr_refrigs'] == 1:
                st.dataframe(df_refrig, use_container_width=True)
            elif hp_model['nr_refrigs'] == 2:
                st.markdown('#### Hochtemperaturkreis')
                st.dataframe(df_refrig2, use_container_width=True)
                st.markdown('#### Niedertemperaturkreis')
                st.dataframe(df_refrig1, use_container_width=True)

            st.write(
                """
                Alle Stoffdaten und Klassifikationen aus
                [CoolProp](http://www.coolprop.org) oder
                [Arpagaus et al. (2018)](https://doi.org/10.1016/j.energy.2018.03.166)
                """
                )

        with st.expander('Anleitung'):
            st.info(
                """
                #### Anleitung

                Sie befinden sich auf der Oberfläche zur Auslegungssimulation
                Ihrer Wärmepumpe. Dazu sind links in der Sidebar neben der
                Dimensionierung und der Wahl des zu verwendenden Kältemittels
                verschiedene zentrale Parameter des Kreisprozesse vorzugeben.

                Dies sind zum Beispiel die Temperaturen der Wärmequelle und
                -senke, aber auch die dazugehörigen Netzdrücke. Darüber hinaus
                kann optional ein interner Wärmeübertrager hinzugefügt werden.
                Dazu ist weiterhin die resultierende Überhitzung des
                verdampften Kältemittels vorzugeben.

                Ist die Auslegungssimulation erfolgreich abgeschlossen, werden
                die generierten Ergebnisse graphisch in Zustandsdiagrammen
                aufgearbeitet und quantifiziert. Die zentralen Größen wie die
                Leistungszahl (COP) sowie die relevanten Wärmeströme und
                Leistung werden aufgeführt. Darüber hinaus werden die
                thermodynamischen Zustandsgrößen in allen Prozessschritten
                tabellarisch aufgelistet.

                Im Anschluss an die Auslegungsimulation erscheint ein Knopf zum
                Wechseln in die Teillastoberfläche. Dies kann ebenfalls über
                das Dropdownmenü in der Sidebar erfolgen. Informationen zur
                Durchführung der Teillastsimulationen befindet sich auf der
                Startseite dieser Oberfläche.
                """
                )

    if run_sim or 'hp' in ss:
        # %% Run Design Simulation
        if run_sim:
            with st.spinner('Simulation wird durchgeführt...'):
                try:
                    ss.hp = run_design(hp_model_name, params)
                    sim_succeded = True
                    st.success(
                        'Die Simulation der Wärmepumpenauslegung war erfolgreich.'
                        )
                except (ValueError, RuntimeError) as e:
                    sim_succeded = False
                    print(f'ValueError: {e}')
                    st.error(
                        'Bei der Simulation der Wärmepumpe ist der nachfolgende '
                        + 'Fehler aufgetreten. Bitte korrigieren Sie die '
                        + f'Eingangsparameter und versuchen es erneut.\n\n"{e}"'
                        )
        else:
            sim_succeded = 'hp' in ss

        # %% MARK: Results
        if sim_succeded:
            with st.spinner('Ergebnisse werden visualisiert...'):

                stateconfigpath = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), 'models', 'input',
                    'state_diagram_config.json'
                    ))
                with open(stateconfigpath, 'r', encoding='utf-8') as file:
                    config = json.load(file)
                if hp_model['nr_refrigs'] == 1:
                    if ss.hp.params['setup']['refrig'] in config:
                        state_props = config[
                            ss.hp.params['setup']['refrig']
                            ]
                    else:
                        state_props = config['MISC']
                if hp_model['nr_refrigs'] == 2:
                    if ss.hp.params['setup']['refrig1'] in config:
                        state_props1 = config[
                            ss.hp.params['setup']['refrig1']
                            ]
                    else:
                        state_props1 = config['MISC']
                    if ss.hp.params['setup']['refrig2'] in config:
                        state_props2 = config[
                            ss.hp.params['setup']['refrig2']
                            ]
                    else:
                        state_props2 = config['MISC']

                st.header('Ergebnisse der Auslegung')

                df_selected_params = build_selected_params_df(
                    params=params,
                    hp_model=hp_model,
                    base_topology=base_topology,
                    model_name=model_name,
                    process_type=process_type
                    )
                if not df_selected_params.empty:
                    with st.expander('Gewählte Eingangsparameter', expanded=True):
                        st.dataframe(
                            df_selected_params,
                            use_container_width=True,
                            hide_index=True
                            )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric('COP', round(ss.hp.cop, 2))
                # Thermal heat delivered to consumer (W → MW)
                Q_out_W = getattr(ss.hp, 'Q_out', None)
                if Q_out_W is None or (isinstance(Q_out_W, float) and np.isnan(Q_out_W)):
                    if hasattr(ss.hp, '_get_heat_output_W'):
                        Q_out_W = ss.hp._get_heat_output_W()
                    else:
                        Q_out_W = ss.hp.comps['cons'].Q.val
                Q_dot_ab = abs(Q_out_W) / 1e6
                col2.metric('Q̇_ab (thermisch)', f"{Q_dot_ab:.2f} MW")
                # Electrical input from grid (E0) (W → MW)
                P_zu = ss.hp.conns['E0'].E.val / 1e6
                col3.metric('P_zu (elektrisch)', f"{P_zu:.2f} MW")

                # Heat extracted at evaporator (W → MW)
                Q_dot_zu = abs(ss.hp.comps['evap'].Q.val) / 1e6
                col4.metric('Q̇_zu (thermisch)', f"{Q_dot_zu:.2f} MW")

                def _fmt_metric(value, unit='', scale=1.0, digits=2, percent=False):
                    try:
                        value = float(value)
                    except Exception:
                        return '—'
                    if not np.isfinite(value):
                        return '—'
                    if percent:
                        return f"{value * 100:.{digits}f} %"
                    return f"{value / scale:.{digits}f} {unit}".strip()

                show_extra_cop_definitions = hp_model_name in {
                    'cascade', 'cascade_2ihx', 'cascade_mbhx'
                }
                if show_extra_cop_definitions and hasattr(ss.hp, 'get_power_cop_metrics'):
                    power_metrics = ss.hp.get_power_cop_metrics()
                    with st.expander('Zusätzliche COP-Definitionen', expanded=False):
                        cop_row = st.columns(4)
                        cop_row[0].metric(
                            'COP gesamt',
                            _fmt_metric(power_metrics.get('cop_total'))
                        )
                        cop_row[1].metric(
                            'COP nur Verdichter elektrisch',
                            _fmt_metric(power_metrics.get('cop_comp_el'))
                        )
                        cop_row[2].metric(
                            'COP nur Verdichter mechanisch',
                            _fmt_metric(power_metrics.get('cop_comp_mech'))
                        )
                        cop_row[3].metric(
                            'P_aux elektrisch',
                            _fmt_metric(
                                power_metrics.get('P_aux_el_W'),
                                unit='kW', scale=1e3
                            )
                        )

                        st.caption(
                            'Der Standard-COP des Dashboards nutzt die gesamte '
                            + 'elektrische Aufnahme. Die zusätzlichen Kennzahlen '
                            + 'trennen Verdichterleistung und Hilfsantriebe für '
                            + 'einen besseren Literaturvergleich.'
                        )

                calibration_result = getattr(ss.hp, 'calibration_result', None)
                if calibration_result and calibration_result.get('mode') == 'a_target':
                    with st.expander('Literatur-Kalibrierung', expanded=False):
                        cal_row = st.columns(4)
                        cal_row[0].metric(
                            'A_target',
                            _fmt_metric(calibration_result.get('target_A'), digits=3)
                        )
                        cal_row[1].metric(
                            'A erreicht',
                            _fmt_metric(calibration_result.get('achieved_A'), digits=3)
                        )
                        cal_row[2].metric(
                            'RIP gefunden',
                            _fmt_metric(calibration_result.get('rip_factor'), digits=3)
                        )

                        delta_a = np.nan
                        target_a = calibration_result.get('target_A')
                        achieved_a = calibration_result.get('achieved_A')
                        if target_a is not None and achieved_a is not None:
                            delta_a = float(achieved_a) - float(target_a)
                        cal_row[3].metric('ΔA', _fmt_metric(delta_a, digits=3))

                        if calibration_result.get('matched'):
                            st.caption(
                                'Der Zielwert für den Einspritzmassenanteil wurde '
                                + 'innerhalb der Kalibrierungstoleranz getroffen.'
                            )
                        else:
                            st.warning(
                                'A_target konnte nur näherungsweise erreicht '
                                + 'werden. Das Modell verwendet den besten stabilen '
                                + 'RIP-Wert aus der Kalibrierung.'
                            )

                if hasattr(ss.hp, 'get_injection_metrics'):
                    injection = ss.hp.get_injection_metrics()
                    if np.isfinite(injection.get('A_inj', np.nan)):
                        with st.expander('Injektionskennzahlen', expanded=False):
                            inj_row = st.columns(4)
                            inj_row[0].metric(
                                'A_inj',
                                _fmt_metric(injection.get('A_inj'), digits=3)
                            )
                            inj_row[1].metric(
                                'ṁ_main',
                                _fmt_metric(
                                    injection.get('m_main_kg_s'), unit='kg/s'
                                )
                            )
                            inj_row[2].metric(
                                'ṁ_inj',
                                _fmt_metric(
                                    injection.get('m_inj_kg_s'), unit='kg/s'
                                )
                            )
                            inj_row[3].metric(
                                'ΔT_sup,inj',
                                _fmt_metric(
                                    injection.get('dT_sup_inj_K'), unit='K'
                                )
                            )

                            st.caption(
                                f"{injection.get('branch_type', 'Injektionszweig')}: "
                                + 'Der Einspritzmassenanteil A_inj wird relativ zum '
                                + 'Hauptmassenstrom des Niederdruckverdichters angegeben.'
                            )

                if hp_model['nr_refrigs'] == 2 and hasattr(ss.hp, 'get_cycle_split_metrics'):
                    split = ss.hp.get_cycle_split_metrics()

                    with st.expander('Kaskadenaufteilung', expanded=True):
                        split_row1 = st.columns(4)
                        split_row1[0].metric(
                            'T_mid',
                            _fmt_metric(split.get('T_mid_C'), unit='°C')
                        )
                        split_row1[1].metric(
                            'Splitfaktor α',
                            _fmt_metric(split.get('t_mid_fraction'), digits=3)
                        )
                        split_row1[2].metric(
                            'ṁ_NTK',
                            _fmt_metric(split.get('m_lt_kg_s'), unit='kg/s')
                        )
                        split_row1[3].metric(
                            'ṁ_HTK',
                            _fmt_metric(split.get('m_ht_kg_s'), unit='kg/s')
                        )

                        split_row2 = st.columns(4)
                        split_row2[0].metric(
                            'P_NTK',
                            _fmt_metric(split.get('P_lt_W'), unit='MW', scale=1e6)
                        )
                        split_row2[1].metric(
                            'P_HTK',
                            _fmt_metric(split.get('P_ht_W'), unit='MW', scale=1e6)
                        )
                        split_row2[2].metric(
                            'NTK-Anteil Verdichterleistung',
                            _fmt_metric(
                                split.get('power_share_lt'), percent=True, digits=1
                            )
                        )
                        split_row2[3].metric(
                            'HTK-Anteil Verdichterleistung',
                            _fmt_metric(
                                split.get('power_share_ht'), percent=True, digits=1
                            )
                        )

                        st.caption(
                            'Die Kreisaufteilung wird aus den gelösten '
                            + 'Zirkulationsmassenströmen und '
                            + 'Verdichterleistungen berechnet. '
                            + f"ṁ_HTK / ṁ_NTK = {_fmt_metric(split.get('m_ht_to_lt'), digits=3)}."
                        )

                        split_row3 = st.columns(4)
                        split_row3[0].metric(
                            'Q̇_NTK->ZWÜ',
                            _fmt_metric(split.get('Q_lt_W'), unit='MW', scale=1e6)
                        )
                        split_row3[1].metric(
                            'Q̇_HTK,ab',
                            _fmt_metric(split.get('Q_ht_W'), unit='MW', scale=1e6)
                        )
                        split_row3[2].metric(
                            'NTK-Kennzahl*',
                            _fmt_metric(split.get('cop_lt'), digits=2)
                        )
                        split_row3[3].metric(
                            'HTK-Kennzahl*',
                            _fmt_metric(split.get('cop_ht'), digits=2)
                        )

                        st.caption(
                            '* Teilprozess-Kennzahlen, keine separaten '
                            + 'Gesamt-COPs: NTK* = Q̇_NTK->ZWÜ / P_NTK und '
                            + 'HTK* = Q̇_HTK,ab / P_HTK. Der HTK-Wert kann '
                            + 'vergleichsweise groß werden, weil seine '
                            + 'Nutzwärme bereits die aus dem NTK '
                            + 'übertragene Wärme enthält. Die Werte sind '
                            + 'nicht direkt zum Gesamt-COP addierbar.'
                        )

                with st.expander('Topologie & Kältemittel'):
                    # %% Topology & Refrigerant
                    col_left, col_right = st.columns([1, 4])

                    with col_left:
                        st.subheader("Topologie")
                        top_file = resolve_topology_svg_path(
                            src_path,
                            hp_model_name_topology,
                            is_dark=is_dark,
                            labeled=True,
                        )

                        topo_col_left, topo_col_right = st.columns(2)
                        theme = "dark" if is_dark else "light"

                        with topo_col_left:
                            st.markdown("**Generierte Topologie**")
                            try:
                                dot = build_graph_from_hp(ss.hp, theme=theme)
                                st.graphviz_chart(dot.source, use_container_width=True)
                                st.caption("Erzeugt mit Graphviz")
                            except Exception as e:
                                st.warning("⚠️ Generierte Topologie aktuell nicht verfügbar.")
                                st.text(f"Fehler beim Erzeugen des Diagramms: {e}")

                        with topo_col_right:
                            st.markdown("**original**")
                            if top_file is not None:
                                st.image(top_file)
                                st.caption("Vorlage aus `img/topologies`")
                            else:
                                st.info("Keine statische Vorlage fuer dieses Modell vorhanden.")











                    with col_right:
                        st.subheader('Kältemittel')

                        if hp_model['nr_refrigs'] == 1:
                            st.dataframe(df_refrig, use_container_width=True)
                        elif hp_model['nr_refrigs'] == 2:
                            st.markdown('#### Hochtemperaturkreis')
                            st.dataframe(df_refrig2, use_container_width=True)
                            st.markdown('#### Niedertemperaturkreis')
                            st.dataframe(df_refrig1, use_container_width=True)

                        st.write(
                            """
                            Alle Stoffdaten und Klassifikationen aus
                            [CoolProp](http://www.coolprop.org) oder
                            [Arpagaus et al. (2018)](https://doi.org/10.1016/j.energy.2018.03.166)
                            """
                            )

                with st.expander('Zustandsdiagramme'):
                    # %% State Diagrams
                    col_left, _, col_right = st.columns([0.495, 0.01, 0.495])
                    _, slider_left, _, slider_right, _ = (
                        st.columns([0.5, 8, 1, 8, 0.5])
                        )

                    if is_dark:
                        state_diagram_style = 'dark'
                    else:
                        state_diagram_style = 'light'

                    with col_left:
                        # %% Log(p)-h-Diagram
                        st.subheader('Log(p)-h-Diagramm')
                        if hp_model['nr_refrigs'] == 1:
                            xmin, xmax = calc_limits(
                                wf=ss.hp.wf, prop='h', padding_rel=0.35
                                )
                            ymin, ymax = calc_limits(
                                wf=ss.hp.wf, prop='p', padding_rel=0.25,
                                scale='log'
                                )

                            diagram = ss.hp.generate_state_diagram(
                                diagram_type='logph',
                                figsize=(12, 7.5),
                                xlims=(xmin, xmax), ylims=(ymin, ymax),
                                style=state_diagram_style,
                                return_diagram=True, display_info=False,
                                open_file=False, savefig=False
                                )
                            st.pyplot(diagram.fig)

                        elif hp_model['nr_refrigs'] == 2:
                            xmin1, xmax1 = calc_limits(
                                wf=ss.hp.wf1, prop='h', padding_rel=0.35
                                )
                            ymin1, ymax1 = calc_limits(
                                wf=ss.hp.wf1, prop='p', padding_rel=0.25,
                                scale='log'
                                )

                            xmin2, xmax2 = calc_limits(
                                wf=ss.hp.wf2, prop='h', padding_rel=0.35
                                )
                            ymin2, ymax2 = calc_limits(
                                wf=ss.hp.wf2, prop='p', padding_rel=0.25,
                                scale='log'
                                )

                            diagram1, diagram2 = ss.hp.generate_state_diagram(
                                diagram_type='logph',
                                figsize=(12, 7.5),
                                xlims=((xmin1, xmax1), (xmin2, xmax2)),
                                ylims=((ymin1, ymax1), (ymin2, ymax2)),
                                style=state_diagram_style,
                                return_diagram=True, display_info=False,
                                savefig=False, open_file=False
                                )
                            st.pyplot(diagram1.fig)
                            st.pyplot(diagram2.fig)

                    with col_right:
                        # %% T-s-Diagram
                        st.subheader('T-s-Diagramm')
                        if hp_model['nr_refrigs'] == 1:
                            xmin, xmax = calc_limits(
                                wf=ss.hp.wf, prop='s', padding_rel=0.35
                                )
                            ymin, ymax = calc_limits(
                                wf=ss.hp.wf, prop='T', padding_rel=0.25
                                )

                            diagram = ss.hp.generate_state_diagram(
                                diagram_type='Ts',
                                figsize=(12, 7.5),
                                xlims=(xmin, xmax), ylims=(ymin, ymax),
                                style=state_diagram_style,
                                return_diagram=True, display_info=False,
                                open_file=False, savefig=False
                                )
                            st.pyplot(diagram.fig)

                        elif hp_model['nr_refrigs'] == 2:
                            xmin1, xmax1 = calc_limits(
                                wf=ss.hp.wf1, prop='s', padding_rel=0.35
                                )
                            ymin1, ymax1 = calc_limits(
                                wf=ss.hp.wf1, prop='T', padding_rel=0.25
                                )

                            xmin2, xmax2 = calc_limits(
                                wf=ss.hp.wf2, prop='s', padding_rel=0.35
                                )
                            ymin2, ymax2 = calc_limits(
                                wf=ss.hp.wf2, prop='T', padding_rel=0.25
                                )

                            diagram1, diagram2 = ss.hp.generate_state_diagram(
                                diagram_type='Ts',
                                figsize=(12, 7.5),
                                xlims=((xmin1, xmax1), (xmin2, xmax2)),
                                ylims=((ymin1, ymax1), (ymin2, ymax2)),
                                style=state_diagram_style,
                                return_diagram=True, display_info=False,
                                savefig=False, open_file=False
                                )
                            st.pyplot(diagram1.fig)
                            st.pyplot(diagram2.fig)

                with st.expander("Zustandsgrößen", expanded=False):
                    from CoolProp.CoolProp import PhaseSI  # <- wichtig

                    df = ss.hp.nw.results["Connection"].copy()

                    # Einheiten-Spalten entfernen
                    df = df.loc[:, ~df.columns.str.contains("_unit", case=False, regex=False)]

                    # --- Fluid-Spalten: bool (wie im Original) ---
                    if "water" in df.columns:
                        df["water"] = (df["water"] == 1.0)
                    elif "H2O" in df.columns:
                        df["H2O"] = (df["H2O"] == 1.0)

                    wf_cols = []
                    if hp_model["nr_refrigs"] == 1:
                        wf = ss.hp.params["setup"]["refrig"]
                        if wf in df.columns:
                            df[wf] = (df[wf] == 1.0)
                            wf_cols.append(wf)
                    else:
                        wf1 = ss.hp.params["setup"]["refrig1"]
                        wf2 = ss.hp.params["setup"]["refrig2"]
                        if wf1 in df.columns:
                            df[wf1] = (df[wf1] == 1.0)
                            wf_cols.append(wf1)
                        if wf2 in df.columns:
                            df[wf2] = (df[wf2] == 1.0)
                            wf_cols.append(wf2)

                    # Störspalte entfernen
                    if "Td_bp" in df.columns:
                        df = df.drop(columns=["Td_bp"])

                    # ==========================
                    # 1) Phase bestimmen (NEU)
                    # ==========================
                    # Wir nehmen als "Working Fluid" das Kältemittel (wf / wf1 / wf2).
                    # Falls das aus irgendeinem Grund nicht gesetzt ist, fällt es auf water/H2O zurück.
                    def _pick_fluid(row):
                        for c in wf_cols:
                            if c in row and row[c] is True:
                                return c
                        if "water" in row and row["water"] is True:
                            return "water"
                        if "H2O" in row and row["H2O"] is True:
                            return "Water"
                        return None

                    def _phase_from_pT(p_bar, T_C, fluid):
                        if fluid is None:
                            return "-"
                        try:
                            p_Pa = float(p_bar) * 1e5
                            T_K = float(T_C) + 273.15
                            # CoolProp PhaseSI liefert z.B. 'gas', 'liquid', 'twophase', 'supercritical_gas', ...
                            ph = PhaseSI("T", T_K, "P", p_Pa, fluid)
                            return str(ph)
                        except Exception:
                            return "?"

                    # neue Spalte "Phase"
                    df.insert(0, "Phase", "-")
                    if ("p" in df.columns) and ("T" in df.columns):
                        for idx, row in df.iterrows():
                            fluid = _pick_fluid(row)
                            df.at[idx, "Phase"] = _phase_from_pT(row["p"], row["T"], fluid)

                    # ==========================
                    # 2) x (Quality) wie Original
                    # ==========================
                    if "x" in df.columns:
                        # x < 0 bedeutet "einphasig" -> "-" (string, Arrow-safe)
                        def _fmt_quality(v):
                            try:
                                fv = float(v)
                            except Exception:
                                return "-"
                            if fv != fv:  # NaN
                                return "-"
                            if fv < 0:
                                return "-"
                            return f"{fv:.5}"

                        df["x"] = df["x"].apply(_fmt_quality)

                    # ==========================
                    # 3) Formatierung wie Original
                    # ==========================
                    for col in df.columns:
                        if df[col].dtype == np.float64:
                            df[col] = df[col].apply(lambda x: f"{x:.5}")

                    df.rename(
                        columns={
                            "m": "m in kg/s",
                            "p": "p in bar",
                            "h": "h in kJ/kg",
                            "T": "T in °C",
                            "v": "v in m³/kg",
                            "vol": "vol in m³/s",
                            "s": "s in kJ/(kgK)",
                            "x": "x [-]",
                        },
                        inplace=True,
                    )

                    st.markdown("**Materialströme (Zustände)** – `x` nur im Zweiphasengebiet; einphasig → `x = -` und Phase zeigt Gas/Flüssig.")
                    st.dataframe(df, use_container_width=True)


                    # -----------------------------
                    # 2) PowerConnections separat anzeigen (E0/E1/E2/E3, e1/e2/e3)
                    # -----------------------------
                    power_rows = []
                    for name, conn in ss.hp.conns.items():
                        # PowerConnections sind nicht im Connection-Result-DF
                        if name.startswith(("E", "e")):
                            P_kW = None
                            try:
                                # je nach TESPy-Version: conn.E.val oder conn.P.val etc.
                                if hasattr(conn, "E") and conn.E.val is not None:
                                    P_kW = float(conn.E.val) / 1e3
                                elif hasattr(conn, "P") and conn.P.val is not None:
                                    P_kW = float(conn.P.val) / 1e3
                            except Exception:
                                P_kW = None

                            power_rows.append({
                                "Label": name,
                                "Power [kW]": "-" if P_kW is None else f"{P_kW:.2f}"
                            })

                    if power_rows:
                        st.markdown("**Nicht-materielle Ströme (elektrische Leistung)**")
                        st.dataframe(pd.DataFrame(power_rows), use_container_width=True, hide_index=True)


                exergy_container = st.container()
                exergoecon_container = st.container()

                with exergoecon_container.expander('Ökonomische / Exergoökonomische Bewertung', expanded=True):
                    # Use the live widget values from this run as the source of
                    # truth. Re-reading the cached dict from session state can
                    # lag behind preset changes and overwrite the current UI
                    # values with stale data.
                    costcalcparams = dict(costcalcparams)
                    elec_price_cent_kWh = float(costcalcparams.get('elec_price_cent_kWh', 40.0))
                    tau_h_per_year = float(costcalcparams.get('tau_h_per_year', 5500.0))

                    # ===============================
                    # CEPCI
                    # ===============================
                    cepcipath = os.path.abspath(os.path.join(
                        os.path.dirname(__file__), 'models', 'input', 'CEPCI.json'
                    ))
                    with open(cepcipath, 'r', encoding='utf-8') as f:
                        _cepci = json.load(f)

                    ref_year = "2015" if "2015" in _cepci else min(_cepci.keys())
                    CEPCI_cur = float(_cepci[str(costcalcparams['current_year'])])
                    CEPCI_ref = float(_cepci[str(ref_year)])
                    cost_method = str(costcalcparams.get('cost_method', 'standard'))
                    repo_cost_mode = cost_method == 'repo_hthp'

                    # ===============================
                    # CAPEX / OPEX (NO exergy here)
                    # ===============================
                    PEC, TCI, Z, cost_diag = build_costs(
                        None, ss.hp,
                        cost_method=cost_method,
                        return_diagnostics=True,
                        hx_area_method=str(costcalcparams.get('hx_area_method', 'q_lmtd')),
                        compressor_eta_vol=float(costcalcparams.get('compressor_eta_vol', 1.0)),
                        compressor_eta_vol_lt=float(costcalcparams.get('compressor_eta_vol_lt', costcalcparams.get('compressor_eta_vol', 1.0))),
                        compressor_eta_vol_ht=float(costcalcparams.get('compressor_eta_vol_ht', costcalcparams.get('compressor_eta_vol', 1.0))),
                        CEPCI_cur=CEPCI_cur,
                        CEPCI_ref=CEPCI_ref,
                        k_evap=float(costcalcparams.get('k_evap', 1500.0)),
                        k_cond=float(costcalcparams.get('k_cond', 3500.0)),
                        k_inter=float(costcalcparams.get('k_inter', 2200.0)),
                        k_trans=float(costcalcparams.get('k_trans', 60.0)) if 'trans' in hp_model_name else 60.0,
                        k_econ=1500.0,
                        k_misc=float(costcalcparams.get('k_misc', 50.0)),
                        usd_to_eur=float(costcalcparams.get('usd_to_eur', 0.93)),
                        hex_cost_model=costcalcparams.get('hex_cost_model', 'ommen'),
                        compressor_cost_model=costcalcparams.get('compressor_cost_model', 'ommen'),
                        flash_cost_model=costcalcparams.get('flash_cost_model', 'ommen'),
                        flash_residence_time_s=float(costcalcparams.get('residence_time', 10.0)),
                        cost_index_year=int(costcalcparams.get('current_year', 2025)),
                        analysis_year=int(costcalcparams.get('analysis_year', costcalcparams.get('current_year', 2025))),
                        install_factor=float(ss.get('install_factor', 4.16)),
                        include_pumps_in_pec=bool(costcalcparams.get('include_pumps_in_pec', True)),
                        tci_factor=float(ss.get('tci_factor', 6.32)),
                        omc_rel=float(ss.get('omc_rel', 0.03)),
                        i_eff=float(ss.get('i_eff', 0.08)),
                        r_n=float(ss.get('r_n', 0.02)),
                        r_n_om=float(ss.get('r_n_om', ss.get('r_n', 0.02))),
                        n=int(ss.get('n', 20)),
                        tau_h_per_year=float(tau_h_per_year)
                    )

                    pec_total = float(sum(PEC.values()))
                    capex_total = float(sum(TCI.values()))
                    col1, col2, col3 = st.columns(3)
                    col1.metric('PEC', f"{pec_total:,.0f} €")
                    col2.metric('TCI', f"{capex_total:,.0f} €")
                    try:
                        Q_out_W = getattr(ss.hp, 'Q_out', None)
                        if Q_out_W is None or (isinstance(Q_out_W, float) and np.isnan(Q_out_W)):
                            if hasattr(ss.hp, '_get_heat_output_W'):
                                Q_out_W = ss.hp._get_heat_output_W()
                            else:
                                Q_out_W = ss.hp.comps['cons'].Q.val
                        inv_spec = capex_total / abs(Q_out_W / 1e6)
                        col3.metric('Spez. Investitionskosten', f"{inv_spec:,.0f} €/MW")
                    except Exception:
                        pass

                    st.markdown("**Ausgewählte PEC-Korrelationen**")
                    st.dataframe(
                        _selected_pec_summary(costcalcparams, CEPCI_cur),
                        use_container_width=True,
                        hide_index=True
                    )
                    if repo_cost_mode:
                        st.caption(
                            f"Mit F_install = {float(ss.get('install_factor', 4.16)):.2f}, "
                            + f"CEPCI-Jahr {costcalcparams['current_year']} → Analysejahr "
                            + f"{int(costcalcparams.get('analysis_year', costcalcparams['current_year']))}, "
                            + f"R_N_OM = {float(ss.get('r_n_om', ss.get('r_n', 0.02))) * 100:.1f} %, "
                            + f"R_N_EL = {float(ss.get('r_n_el', ss.get('r_n', 0.02))) * 100:.1f} %, "
                            + f"HX-Ansatz = {('TESPy kA / U' if costcalcparams.get('hx_area_method') == 'tespy_ka' else 'Q / (U * LMTD)')}, "
                            + f"η_vol,LT = {float(costcalcparams.get('compressor_eta_vol_lt', costcalcparams.get('compressor_eta_vol', 1.0))):.2f}, "
                            + f"η_vol,HT = {float(costcalcparams.get('compressor_eta_vol_ht', costcalcparams.get('compressor_eta_vol', 1.0))):.2f}, "
                            + f"i_eff = {float(ss.get('i_eff', 0.08)) * 100:.1f} %, "
                            + f"n = {int(ss.get('n', 20))} Jahren, "
                            + f"f_O&M = {float(ss.get('omc_rel', 0.03)) * 100:.1f} % "
                            + "und τ_h = Volllaststunden pro Jahr."
                        )
                    else:
                        st.caption(
                            f"Mit i_eff = {float(ss.get('i_eff', 0.08)) * 100:.1f} %, "
                            + f"r_n = {float(ss.get('r_n', 0.02)) * 100:.1f} %, "
                            + f"n = {int(ss.get('n', 20))} Jahren, "
                            + f"f_O&M = {float(ss.get('omc_rel', 0.03)) * 100:.1f} % "
                            + "und τ_h = Volllaststunden pro Jahr."
                        )

                    st.markdown("**Kostenaufschlüsselung (PEC)**")
                    st.dataframe(
                        pd.DataFrame({"Component": list(PEC.keys()), "PEC [EUR]": list(PEC.values())})
                        .sort_values("PEC [EUR]", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("**Kostenaufschlüsselung (TCI)**")
                    st.dataframe(
                        pd.DataFrame({"Component": list(TCI.keys()), "TCI [EUR]": list(TCI.values())})
                        .sort_values("TCI [EUR]", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("**Betriebskostenraten (Z)**")
                    st.dataframe(
                        pd.DataFrame({"Component": list(Z.keys()), "Z [EUR/h]": list(Z.values())})
                        .sort_values("Z [EUR/h]", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )

                    if repo_cost_mode and cost_diag:
                        cost_diag_df = pd.DataFrame(cost_diag)

                    st.markdown("### Projektkostenabschätzung nach Kosmadakis et al. (2020)")
                    kos_rows = build_kosmadakis_project_cost_df(
                        hp=ss.hp,
                        PEC=PEC,
                        kosmadakis_params=ss.get('kosmadakis_params', {}),
                        elec_price_cent_kWh=elec_price_cent_kWh,
                        tau_h_per_year=tau_h_per_year,
                    )
                    kos_values = kos_rows.set_index("Größe")["Wert"]
                    C_project = float(kos_values.get("C_project", np.nan))
                    E_s = float(kos_values.get("E_s", np.nan))
                    PBP = float(kos_values.get("PBP", np.nan))

                    kc1, kc2, kc3 = st.columns(3)
                    kc1.metric(
                        "C_project",
                        "—" if np.isnan(C_project) else f"{C_project:,.0f} €"
                    )
                    kc2.metric(
                        "E_s",
                        "—" if np.isnan(E_s) else f"{E_s:,.0f} €/a"
                    )
                    kc3.metric(
                        "PBP",
                        "—" if np.isnan(PBP) else f"{PBP:,.2f} a"
                    )

                    st.dataframe(
                        kos_rows.style.format({"Wert": "{:,.2f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption(
                        "Dieser Block ist eine eigenständige Projektkostenabschätzung."
                    )

                    # ===============================
                    # HARD reuse exergy boundaries
                    # ===============================
                    boundaries = getattr(ss.hp, "exergy_boundaries", None)
                    if boundaries is None:
                        st.error(
                            "❌ Exergy boundaries fehlen.\n\n"
                            "Fix in HeatPumpBase.perform_exergy_analysis:\n"
                            "self.exergy_boundaries = {'fuel': fuel, 'product': product, 'loss': loss}"
                        )
                        st.stop()

                    # Optional debug toggle (NOT an expander!)
                    show_bounds = st.checkbox("Randbedingungen anzeigen (Debug)", value=False)
                    if show_bounds:
                        st.json(boundaries)

                    # Debug: Valve 2 outlet state (A2) for ExerPy support (print to terminal)
                    for lbl in ["A1", "A2"]:
                        c = ss.hp.conns.get(lbl)
                        if c is None:
                            print(f"[DBG] {lbl}: connection not found")
                            continue
                        x_val = getattr(c, "x", None)
                        x_val = getattr(x_val, "val", None) if x_val is not None else None
                        print(
                            f"[DBG] {lbl}: p={getattr(c.p,'val',None)} bar, "
                            f"T={getattr(c.T,'val',None)} °C, "
                            f"h={getattr(c.h,'val',None)} kJ/kg, "
                            f"x={x_val}"
                        )

                    # ===============================
                    # Ambient
                    # ===============================
                    Tamb_K = float(params["ambient"]["T"]) + 273.15
                    pamb_Pa = float(params["ambient"]["p"]) * 1e5

                    econ_params = dict(
                        i_eff=float(ss.get('i_eff', 0.08)),
                        r_n=float(ss.get('r_n', 0.02)),
                        r_n_om=float(ss.get('r_n_om', ss.get('r_n', 0.02))),
                        r_n_el=float(ss.get('r_n_el', ss.get('r_n', 0.02))),
                        n=int(ss.get('n', 20)),
                        omc_rel=float(ss.get('omc_rel', 0.03)),
                        tci_factor=float(ss.get('tci_factor', 6.32)),
                        install_factor=float(ss.get('install_factor', 4.16)),
                    )

                    # ===============================
                    # Run exergoeconomic analysis
                    # ===============================
                    try:
                        df_execo_comp, df_mat1, df_mat2, df_non_mat, df_execo_eval, ean, Exe_Eco_Costs = run_exergoeconomic_from_hp(
                            hp=ss.hp,
                            Tamb_K=Tamb_K,
                            pamb_Pa=pamb_Pa,
                            boundaries=boundaries,   # ← NO UI, NO GUESSING
                            elec_price_cent_kWh=float(elec_price_cent_kWh),
                            costcalcparams=costcalcparams,
                            CEPCI_cur=CEPCI_cur,
                            CEPCI_ref=CEPCI_ref,
                            tau_h_per_year=float(tau_h_per_year),
                            econ_params=econ_params,
                            print_results=False,
                            debug=True
                        )

                        #print to terminal 
                        if df_execo_comp is not None:
                            print("\n" + "="*80)
                            print("[EXERGOECONOMIC ANALYSIS – COMPONENT TABLE]")
                            print("="*80)
                            print(df_execo_comp.to_string())
                            print("="*80 + "\n")
                        st.success("✅ Exergoökonomische Analyse erfolgreich.")
                        if getattr(ean, "used_allow_singular_exergoeconomics", False):
                            st.info(
                                "Hinweis: Die exergoökonomische Gleichungsmatrix "
                                + "war singulär. ExerPy wurde daher mit "
                                + "`allow_singular=True` im Least-Squares-Modus "
                                + "fortgesetzt."
                            )
                    except Exception as exc:
                        import traceback
                        st.error(f"❌ Exergoökonomische Analyse fehlgeschlagen:\n{exc}")
                        st.code(traceback.format_exc())  # <-- FULL traceback in UI
                        df_execo_comp = df_mat1 = df_mat2 = df_non_mat = df_execo_eval = None
                        Exe_Eco_Costs = None


                    # ===============================
                    # Results 
                    # ===============================
                    if df_execo_comp is not None:
                        if "Component" not in df_execo_comp.columns:
                            df_execo_comp = df_execo_comp.reset_index().rename(columns={"index": "Component"})

                        st.markdown("### Exergoökonomische Ergebnisse")
                        st.markdown("**Komponenten**")
                        st.dataframe(st_safe_df(df_execo_comp), use_container_width=True, hide_index=True)

                        if df_execo_eval is not None:
                            if "Component" not in df_execo_eval.columns:
                                df_execo_eval = df_execo_eval.reset_index().rename(columns={"index": "Component"})
                            st.markdown("**Komponentenranking (`evaluate_results`)**")
                            st.dataframe(st_safe_df(df_execo_eval), use_container_width=True, hide_index=True)

                        summary_df = pd.DataFrame([{
                            "COP": getattr(ss.hp, "cop", np.nan),
                            "epsilon": getattr(ss.hp, "epsilon", np.nan),
                            "E_F [MW]": float(getattr(ss.hp, "E_F", np.nan)) / 1e6,
                            "E_P [MW]": float(getattr(ss.hp, "E_P", np.nan)) / 1e6,
                            "E_D [MW]": float(getattr(ss.hp, "E_D", np.nan)) / 1e6,
                            "E_L [MW]": float(getattr(ss.hp, "E_L", np.nan)) / 1e6,
                            "PEC_total [EUR]": pec_total,
                            "TCI_total [EUR]": capex_total,
                            "Z_total [EUR/h]": float(sum(Z.values())),
                        }])
                        pec_df = pd.DataFrame({
                            "Component": list(PEC.keys()),
                            "PEC [EUR]": list(PEC.values())
                        }).sort_values("PEC [EUR]", ascending=False)
                        tci_df = pd.DataFrame({
                            "Component": list(TCI.keys()),
                            "TCI [EUR]": list(TCI.values())
                        }).sort_values("TCI [EUR]", ascending=False)
                        z_df = pd.DataFrame({
                            "Component": list(Z.keys()),
                            "Z [EUR/h]": list(Z.values())
                        }).sort_values("Z [EUR/h]", ascending=False)
                        cost_diag_df = pd.DataFrame(cost_diag) if cost_diag else pd.DataFrame()
                        exergy_export_df = getattr(ss.hp, "component_exergy_df", None)
                        if exergy_export_df is None and ean is not None:
                            try:
                                exergy_export_df, _, _ = ean.exergy_results(print_results=False)
                            except Exception:
                                exergy_export_df = pd.DataFrame()
                        elif exergy_export_df is None:
                            exergy_export_df = pd.DataFrame()
                        selected_params_export_df = build_selected_params_df(
                            params=params,
                            hp_model=hp_model,
                            base_topology=base_topology,
                            model_name=model_name,
                            process_type=process_type
                        )
                        literature_metrics_export_df = build_literature_metrics_export_df(
                            hp=ss.hp,
                            hp_model_name=hp_model_name
                        )
                        reference_internal_export_df = build_reference_internal_states_export_df(
                            hp=ss.hp,
                            params=params
                        )
                        export_sheets = {
                            "selected_inputs": selected_params_export_df,
                            "summary": summary_df,
                            "exergy_components": exergy_export_df,
                            "exergoeconomic_components": df_execo_comp,
                            "evaluate_results": df_execo_eval if df_execo_eval is not None else pd.DataFrame(),
                            "PEC": pec_df,
                            "TCI": tci_df,
                            "Z": z_df,
                            "Kosmadakis_project_cost": kos_rows,
                        }
                        if not cost_diag_df.empty:
                            export_sheets["cost_diagnostics"] = cost_diag_df
                        if not literature_metrics_export_df.empty:
                            export_sheets["Topologiespezifische Kennzahl"] = (
                                literature_metrics_export_df
                            )
                        if not reference_internal_export_df.empty:
                            export_sheets["Referenzvergleich intern"] = (
                                reference_internal_export_df
                            )
                        export_bytes = _build_excel_xml_workbook(export_sheets)
                        st.download_button(
                            "Analyse-Export herunterladen",
                            data=export_bytes,
                            file_name="exergoeconomic_analysis_export.xls",
                            mime="application/vnd.ms-excel"
                        )


                with exergy_container.expander('Exergiebewertung', expanded=True):
                    # --- Guard ---
                    if not hasattr(ss.hp, 'ean') or ss.hp.ean is None:
                        st.error("Exergieanalyse wurde nicht durchgeführt.")
                        st.stop()

                    ean = ss.hp.ean

                    # ===== 2) Tabellarische Ergebnisse direkt aus ExerPy =====
                    # ExerPy returns (df_components, df_material_connections, df_nonmaterial_connections)
                    res = ean.exergy_results(print_results=False)

                    if not (isinstance(res, tuple) and len(res) >= 3):
                        st.error("Unerwartetes Rückgabeformat von ean.exergy_results().")
                        st.stop()

                    df_comp, df_mat, df_nonmat = res[:3]


                    # --- Clean up / Arrow-friendly types ---
                    import pandas as pd, numpy as np
                    for df in (df_comp, df_mat, df_nonmat):
                        if isinstance(df, pd.DataFrame):
                            for c in df.columns:
                                # only convert numeric-looking strings
                                try:
                                    df[c] = pd.to_numeric(df[c], errors='ignore')
                                except Exception:
                                    pass

                    # ===== 1) Top-level KPIs (robust, with fallback to TOT row) =====
                    def _fmt(x, den=1e3, unit='kW'):
                        try:
                            return f"{float(x)/den:,.2f} {unit}"
                        except Exception:
                            return "—"

                    # Try system attributes first
                    EF_sys = getattr(ean, 'E_F', None)
                    EP_sys = getattr(ean, 'E_P', None)
                    ED_sys = getattr(ean, 'E_D', None)
                    EL_sys = getattr(ean, 'E_L', None)
                    eps_sys = getattr(ean, 'epsilon', None)

                    # Fallback from TOT row if any missing
                    try:
                        if isinstance(df_comp, pd.DataFrame) and not df_comp.empty:
                            if "Component" in df_comp.columns:
                                tot_row = df_comp.loc[df_comp["Component"].astype(str) == "TOT"]
                            else:
                                tot_row = pd.DataFrame()
                            if not tot_row.empty:
                                def _get_tot(colname, factor=1.0):
                                    if colname in tot_row.columns:
                                        v = pd.to_numeric(tot_row[colname], errors='coerce').iloc[0]
                                        return None if pd.isna(v) else float(v) * factor
                                    return None
                                # tables are in kW → convert to W with *1e3 for consistency
                                EF_sys = EF_sys if EF_sys is not None else _get_tot("E_F [kW]", 1e3)
                                EP_sys = EP_sys if EP_sys is not None else _get_tot("E_P [kW]", 1e3)
                                ED_sys = ED_sys if ED_sys is not None else _get_tot("E_D [kW]", 1e3)
                                EL_sys = EL_sys if EL_sys is not None else _get_tot("E_L [kW]", 1e3)
                                if eps_sys is None and (EF_sys not in (None, 0)) and (EP_sys is not None):
                                    eps_sys = float(EP_sys) / float(EF_sys)
                    except Exception:
                        pass

                    col1, col2, col3, col4, col5 = st.columns(5)
                    try:
                        col1.metric('ε (gesamt)', f"{(float(eps_sys) if eps_sys is not None else 0.0)*100:,.2f} %")
                    except Exception:
                        col1.metric('ε (gesamt)', "—")
                    col2.metric('E_F', _fmt(EF_sys, 1e6, 'MW'))
                    col3.metric('E_P', _fmt(EP_sys, 1e6, 'MW'))
                    col4.metric('E_D', _fmt(ED_sys, 1e6, 'MW'))
                    col5.metric('E_L', _fmt(EL_sys, 1e3, 'kW'))

                    st.caption("Hinweis: ε = E_P / E_F; E_D = E_F − E_P − E_L.")

                    # ===== 2a) Komponenten =====
                    st.subheader("Komponenten (Exergie)")
                    # Bring "TOT" to bottom if present
                    if "Component" in df_comp.columns:
                        tot_mask = df_comp["Component"].astype(str).eq("TOT")
                        df_comp = pd.concat([df_comp.loc[~tot_mask], df_comp.loc[tot_mask]], ignore_index=True)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)

                    boundary_info = getattr(ss.hp, "exergy_boundary_info", {}) or {}
                    active_case = boundary_info.get("scenario", "fallback")
                    if boundary_info.get("return_below_ambient", False):
                        st.warning("Return is below ambient.")

                    case_labels = {
                        "case_a_environmental_source": "Case A",
                        "case_b_waste_heat": "Case B",
                        "case_c_waste_heat_further_usage": "Case C",
                        "fallback": "Fallback",
                    }
                    st.caption(f"Aktueller Systemgrenzenfall: {case_labels.get(active_case, active_case)}")

                    st.markdown("**Definition von Fuel, Product und Loss**")
                    st.code(
                        'Case A — environmental source, cooled below ambient\n\n'
                        'fuel = {"inputs": ["E0"], "outputs": []}\n'
                        'product = {"inputs": ["C3"], "outputs": ["C1"]}\n'
                        'loss = {"inputs": ["B3"], "outputs": ["B1"]}',
                        language="python"
                    )
                    st.code(
                        'Case B — Standardvariante: Abwärme, die abgekühlt wird. Die Austrittstemperatur kann über oder gleich der Umgebungstemperatur werden.\n\n'
                        'fuel = {"inputs": ["E0", "B1"], "outputs": []}\n'
                        'product = {"inputs": ["C3"], "outputs": ["C1"]}\n'
                        'loss = {"inputs": ["B3"], "outputs": []}',
                        language="python"
                    )
                    st.code(
                        'Case C — Abwärme, die abgekühlt wird. Die Austrittstemperatur ist über Umgebungstemperatur und kann in weiteren Anlagen als Abwärme verwertet werden.\n\n'
                        'fuel = {"inputs": ["E0","B1"], "outputs": []}\n'
                        'product = {"inputs": ["C3"], "outputs": ["C1"]}\n'
                        'loss = {"inputs": [""], "outputs": [""]}',
                        language="python"
                    )

                    # ===== 2b) Material-Verbindungen =====
                    st.subheader("Materialströme (Verbindungen)")
                    st.dataframe(df_mat, use_container_width=True, hide_index=True)

                    # ===== 2c) Nicht-materielle Verbindungen (Leistung/Wärme) =====
                    st.subheader("Nicht-materielle Ströme (Leistung/Wärme)")
                    st.dataframe(df_nonmat, use_container_width=True, hide_index=True)

                    # ===== 3) Konsistenzprüfungen / Hinweise (NO nested expander) =====
                    with st.container():
                        st.markdown("### Konsistenz / Debug")
                        try:
                            # Try component sum in kW → convert to W for comparison
                            comp_ED_kw = pd.to_numeric(df_comp.get("E_D [kW]"), errors="coerce")
                            ED_sum_W = float(comp_ED_kw.dropna().sum()) * 1e3 if isinstance(comp_ED_kw, pd.Series) else np.nan

                            st.write(
                                f"E_F: {_fmt(EF_sys,1,'W')},  "
                                f"E_P: {_fmt(EP_sys,1,'W')},  "
                                f"E_L: {_fmt(EL_sys,1,'W')}"
                            )
                            st.write(f"E_D (System): {_fmt(ED_sys,1,'W')}")
                            if (ED_sys is not None) and (ED_sum_W is not None) and not (np.isnan(ED_sys) or np.isnan(ED_sum_W)):
                                delta = ED_sum_W - ED_sys
                                if abs(delta) < max(1e-6 * max(abs(ED_sys), 1.0), 5.0):
                                    st.success("Exergiebilanz passt (innerhalb Toleranz).")
                                else:
                                    st.info(
                                        "Abweichungen können auftreten (Rundungen, Definition der Loss-Ströme, "
                                        "Komponenten ohne vollständige Bilanzgleichungen)."
                                    )
                        except Exception as e:
                            st.info(f"Konsistenzprüfung übersprungen: {e}")

                    # ===== 4) Downloads =====
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    col_dl1.download_button(
                        "Komponenten als CSV", df_comp.to_csv(index=False).encode("utf-8"),
                        file_name="exergy_components.csv", mime="text/csv"
                    )
                    col_dl2.download_button(
                        "Material-Verbindungen als CSV", df_mat.to_csv(index=False).encode("utf-8"),
                        file_name="exergy_material_connections.csv", mime="text/csv"
                    )
                    col_dl3.download_button(
                        "Nicht-materielle Verbindungen als CSV", df_nonmat.to_csv(index=False).encode("utf-8"),
                        file_name="exergy_nonmaterial_connections.csv", mime="text/csv"
                    )

                    # ===== 5) Plots =====
                    col6, col7 = st.columns(2)
                    with col6:
                        st.subheader('Grassmann Diagramm')
                        diagram_placeholder_sankey = st.empty()
                        try:
                            diagram_sankey = ss.hp.generate_sankey_diagram()
                            diagram_placeholder_sankey.plotly_chart(
                                diagram_sankey, use_container_width=True
                            )
                        except Exception as e:
                            st.info(f"Sankey ausgelassen: {e}")

                    with col7:
                        st.subheader('Wasserfall Diagramm')
                        diagram_placeholder_waterfall = st.empty()
                        try:
                            dia_wf_fig, _ = ss.hp.generate_waterfall_diagram(return_fig_ax=True)
                            diagram_placeholder_waterfall.pyplot(
                                dia_wf_fig, use_container_width=True
                            )
                        except Exception as e:
                            st.info(f"Wasserfall ausgelassen: {e}")

                partload_reason = None
                if hasattr(ss.hp, 'get_partload_mode_reason'):
                    partload_reason = ss.hp.get_partload_mode_reason()
                if partload_reason:
                    st.info(partload_reason)
                else:
                    st.info('Um die Teillast zu berechnen, drücke auf "Teillast simulieren".')
                    st.button('Teillast simulieren', on_click=switch2partload)

if mode == 'Teillast':
    # %% MARK: Offdesign Simulation
    st.header('Betriebscharakteristik')

    if 'hp' not in ss:
        st.warning(
            '''
            Um eine Teillastsimulation durchzuführen, muss zunächst eine 
            Wärmepumpe ausgelegt werden. Wechseln Sie bitte zunächst in den 
            Modus "Auslegung".
            '''
        )
    else:
        if (
            hasattr(ss.hp, 'supports_partload_boundary_modes')
            and not ss.hp.supports_partload_boundary_modes()
        ):
            st.warning(ss.hp.get_partload_mode_reason())
            st.stop()

        if not run_pl_sim and 'partload_char' not in ss:
            # %% Landing Page
            st.write(
                '''
                Parametrisierung der Teillastberechnung:
                + Prozentualer Anteil Teillast
                + Bereich der Quelltemperatur
                + Bereich der Senkentemperatur
                '''
                )

        if run_pl_sim:
            # %% Run Offdesign Simulation
            with st.spinner(
                    'Teillastsimulation wird durchgeführt... Dies kann eine '
                    + 'Weile dauern.'
                    ):
                ss.hp, ss.partload_char = (
                    run_partload(ss.hp)
                    )
                # ss.partload_char = pd.read_csv(
                #     'partload_char.csv', index_col=[0, 1, 2], sep=';'
                #     )
                st.success(
                    'Die Simulation der Wärmepumpencharakteristika war '
                    + 'erfolgreich.'
                    )

        if run_pl_sim or 'partload_char' in ss:
            # %% Results
            with st.spinner('Ergebnisse werden visualisiert...'):

                with st.expander('Diagramme', expanded=True):
                    col_left, col_right = st.columns(2)

                    with col_left:
                        figs, axes = ss.hp.plot_partload_char(
                            ss.partload_char, cmap_type='COP',
                            cmap='plasma', return_fig_ax=True
                            )
                        pl_cop_placeholder = st.empty()

                        if type_hs == 'Konstant':
                            T_select_cop = (
                                ss.hp.params['offdesign']['T_hs_ff_start']
                                )
                        elif type_hs == 'Variabel':
                            T_hs_min = (
                                ss.hp.params['offdesign']['T_hs_ff_start']
                                )
                            T_hs_max = (
                                ss.hp.params['offdesign']['T_hs_ff_end']
                                )
                            T_select_cop = st.slider(
                                'Quellentemperatur',
                                min_value=T_hs_min,
                                max_value=T_hs_max,
                                value=int((T_hs_max+T_hs_min)/2),
                                format='%d °C',
                                key='pl_cop_slider'
                                )

                        pl_cop_placeholder.pyplot(figs[T_select_cop])

                    with col_right:
                        figs, axes = ss.hp.plot_partload_char(
                            ss.partload_char, cmap_type='T_cons_ff',
                            cmap='plasma', return_fig_ax=True
                            )
                        pl_T_cons_ff_placeholder = st.empty()

                        if type_hs == 'Konstant':
                            T_select_T_cons_ff = (
                                ss.hp.params['offdesign']['T_hs_ff_start']
                                )
                        elif type_hs == 'Variabel':
                            T_select_T_cons_ff = st.slider(
                                'Quellentemperatur',
                                min_value=T_hs_min,
                                max_value=T_hs_max,
                                value=int((T_hs_max+T_hs_min)/2),
                                format='%d °C',
                                key='pl_T_cons_ff_slider'
                                )
                        pl_T_cons_ff_placeholder.pyplot(
                            figs[T_select_T_cons_ff]
                            )

                with st.expander('Exergieanalyse Teillast', expanded=True):

                    col_left_1, col_right_1 = st.columns(2)

                    with col_left_1:
                        figs, axes = ss.hp.plot_partload_char(
                            ss.partload_char, cmap_type='epsilon',
                            cmap='plasma', return_fig_ax=True
                        )
                        pl_epsilon_placeholder = st.empty()

                        if type_hs == 'Konstant':
                            T_select_epsilon = (
                                ss.hp.params['offdesign']['T_hs_ff_start']
                            )
                        elif type_hs == 'Variabel':
                            T_hs_min = (
                                ss.hp.params['offdesign']['T_hs_ff_start']
                                )
                            T_hs_max = (
                                ss.hp.params['offdesign']['T_hs_ff_end']
                                )
                            T_select_epsilon = st.slider(
                                'Quellentemperatur',
                                min_value=T_hs_min,
                                max_value=T_hs_max,
                                value=int((T_hs_max + T_hs_min) / 2),
                                format='%d °C',
                                key='pl_epsilon_slider'
                            )

                        pl_epsilon_placeholder.pyplot(figs[T_select_epsilon])

                st.button('Neue Wärmepumpe auslegen', on_click=reset2design)

# %% MARK: Footer
st.markdown("<br><br>", unsafe_allow_html=True)

pad_left, col_bot, pad_right = st.columns(3)

mail_path = os.path.join(icon_path, 'mail_icon_bw.svg')
orcid_path = os.path.join(icon_path, 'orcid_icon_bw.svg')
github_path = os.path.join(icon_path, 'github_icon_bw.svg')
linkedin_path = os.path.join(icon_path, 'linkedin_icon_bw.svg')

mail64 = img_to_base64(mail_path)
orcid64 = img_to_base64(orcid_path)
github64 = img_to_base64(github_path)
linkedin64 = img_to_base64(linkedin_path)

if col_bot.button(
    '© Jonas Freißmann & Malte Fritz :material/open_in_new:', type='tertiary',
    use_container_width=True):
    footer()
