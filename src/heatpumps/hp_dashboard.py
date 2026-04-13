import base64
import json
import os

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
    _append_param_row(rows, 'Wärmequelle', 'Vorlauf', f"{source_ff.get('T')} °C")
    _append_param_row(rows, 'Wärmequelle', 'Rücklauf', f"{source_bf.get('T')} °C")
    if 'p' in source_ff:
        _append_param_row(
            rows, 'Wärmequelle', 'Eintrittsdruck',
            f"{source_ff.get('p')} bar"
            )

    sink_rf = params.get('C1', {})
    sink_ff = params.get('C3', {})
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

            # Hard-reset cached model state when topology/model changes
            model_signature = f"{base_topology}|{model_name}|{process_type}|{hp_model_name}"
            if ss.get("hp_model_signature") != model_signature:
                _hard_reset_model_state()
                ss.hp_model_signature = model_signature

            parampath = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 'models', 'input',
                f'params_hp_{hp_model_name}.json'
                ))
            with open(parampath, 'r', encoding='utf-8') as file:
                params = json.load(file)
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
            params['B1']['T'] = st.slider(
                'Temperatur Vorlauf', min_value=0, max_value=T_crit,
                value=params['B1']['T'], format='%d°C', key='T_heatsource_ff'
                )
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

        # TODO: Aktuell wird T_mid im Modell als Mittelwert zwischen von Ver-
        #       dampfungs- und Kondensationstemperatur gebildet. An sich wäre
        #       es analytisch sicher interessant den Wert selbst festlegen zu
        #       können.
        # if hp_model['nr_refrigs'] == 2:
        #     with st.expander('Zwischenwärmeübertrager'):
        #         param['design']['T_mid'] = st.slider(
        #             'Mittlere Temperatur', min_value=0, max_value=T_crit,
        #             value=40, format='%d°C', key='T_mid'
        #             )

        with st.expander('Wärmesenke'):
            T_max_sink = T_crit
            if 'trans' in hp_model_name:
                T_max_sink = 200  # °C -- Ad hoc value, maybe find better one

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

            invalid_temp_diff = params['C1']['T'] >= params['C3']['T']
            if invalid_temp_diff:
                st.error(
                    'Die Rücklauftemperatur muss niedriger sein, als die '
                    + 'Vorlauftemperatur.'
                )
            invalid_temp_diff = params['C1']['T'] <= params['B1']['T']
            if invalid_temp_diff:
                st.error(
                    'Die Temperatur der Wärmesenke muss höher sein, als die '
                    + 'der Wärmequelle.'
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

            st.caption(
                'Das CEPCI-Referenzjahr ist fest auf 2015 gesetzt. '
                'Im Dashboard wird nur das aktuelle Kostenjahr ausgewählt.'
            )

            costcalcparams['current_year'] = st.selectbox(
                'Jahr der Kostenkalkulation',
                options=sorted(list(cepci.keys()), reverse=True),
                key='current_year'
            )

            costcalcparams['k_evap'] = st.slider(
                'Wärmedurchgangskoeffizient (Verdampfung)',
                min_value=0, max_value=5000, step=10,
                value=1500, format='%d W/m²K', key='k_evap'
                )

            costcalcparams['k_cond'] = st.slider(
                'Wärmedurchgangskoeffizient (Verflüssigung)',
                min_value=0, max_value=5000, step=10,
                value=3500, format='%d W/m²K', key='k_cond'
                )

            if 'trans' in hp_model_name:
                costcalcparams['k_trans'] = st.slider(
                    'Wärmedurchgangskoeffizient (transkritisch)',
                    min_value=0, max_value=1000, step=5,
                    value=60, format='%d W/m²K', key='k_trans'
                    )

            costcalcparams['k_misc'] = st.slider(
                'Wärmedurchgangskoeffizient (Sonstige)',
                min_value=0, max_value=1000, step=5,
                value=50, format='%d W/m²K', key='k_misc'
                )

            costcalcparams['residence_time'] = st.slider(
                'Verweildauer Flashtank',
                min_value=0, max_value=60, step=1,
                value=10, format='%d s', key='residence_time'
                )

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

            if is_dark:
                try:
                    top_file = os.path.join(
                        src_path, 'img', 'topologies',
                        f'hp_{hp_model_name_topology}_dark.svg'
                        )
                    st.image(top_file)
                except:
                    top_file = os.path.join(
                        src_path, 'img', 'topologies',
                        f'hp_{hp_model_name_topology}.svg'
                        )
                    st.image(top_file)

            else:
                top_file = os.path.join(
                    src_path, 'img', 'topologies',
                    f'hp_{hp_model_name_topology}.svg'
                    )
                st.image(top_file)

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

    if run_sim:
        # %% Run Design Simulation
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
                with st.expander('Topologie & Kältemittel'):
                    # %% Topology & Refrigerant
                    col_left, col_right = st.columns([1, 4])

                    with col_left:
                        st.subheader("Topologie")
                        top_file = os.path.join(
                            src_path, "img", "topologies",
                            f"hp_{hp_model_name_topology}_label.svg"
                        )
                        if is_dark:
                            top_file_dark = os.path.join(
                                src_path, "img", "topologies",
                                f"hp_{hp_model_name_topology}_label_dark.svg"
                            )
                            if os.path.exists(top_file_dark):
                                top_file = top_file_dark

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
                            st.image(top_file)
                            st.caption("Vorlage aus `img/topologies`")











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

                    # ===============================
                    # User inputs (ONLY economics)
                    # ===============================
                    colp, colt = st.columns(2)
                    elec_price_cent_kWh = colp.number_input(
                        "Strompreis [ct/kWh]", 0.0, 200.0, 40.0, step=1.0
                    )
                    tau_h_per_year = colt.number_input(
                        "Volllaststunden [h/a]", 0.0, 9000.0, 5500.0, step=100.0
                    )

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
                    cepci_factor = CEPCI_cur / CEPCI_ref

                    # ===============================
                    # CAPEX / OPEX (NO exergy here)
                    # ===============================
                    PEC, TCI, Z = build_costs(
                        None, ss.hp,
                        CEPCI_cur=CEPCI_cur,
                        CEPCI_ref=CEPCI_ref,
                        k_evap=float(costcalcparams.get('k_evap', 1500.0)),
                        k_cond=float(costcalcparams.get('k_cond', 3500.0)),
                        k_inter=2200.0,
                        k_trans=float(costcalcparams.get('k_trans', 60.0)) if 'trans' in hp_model_name else 60.0,
                        k_econ=1500.0,
                        k_misc=float(costcalcparams.get('k_misc', 50.0)),
                        flash_residence_time_s=float(costcalcparams.get('residence_time', 10.0)),
                        tci_factor=6.32,
                        omc_rel=0.03,
                        i_eff=0.08,
                        r_n=0.02,
                        n=20,
                        tau_h_per_year=float(tau_h_per_year)
                    )

                    pec_total = float(sum(PEC.values()))
                    capex_total = float(sum(TCI.values()))
                    col1, col2, col3 = st.columns(3)
                    col1.metric('Komponentenkosten (PEC)', f"{pec_total:,.0f} €")
                    col2.metric('Gesamtinvestitionskosten (TCI)', f"{capex_total:,.0f} €")
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

                    st.markdown("### Verwendete Kostengleichungen")
                    st.markdown(
                        f"""
                        Die Komponentenpreise werden in dieser Oberfläche mit den im Paket
                        implementierten Skalierungsgleichungen berechnet. Anschließend
                        wird die Preisbasis mit dem CEPCI-Faktor korrigiert.

                        Aktuell gilt:
                        - `CEPCI-Referenzjahr = {ref_year}`
                        - `CEPCI-Faktor = CEPCI_aktuell / CEPCI_ref = {CEPCI_cur:.1f} / {CEPCI_ref:.1f} = {cepci_factor:.3f}`
                        - Strompreis für die Exergoökonomie: `c_el = (ct/kWh / 100) * 277.78`
                        """
                    )

                    st.latex(r"PEC_{\mathrm{comp}} = 19{,}850 \left(\frac{\dot{V}_{in}}{279.8}\right)^{0.73} \cdot \frac{\mathrm{CEPCI}_{cur}}{\mathrm{CEPCI}_{ref}}")
                    st.caption("Verdichterkosten aus dem Eintritts-Volumenstrom \\(\\dot{V}_{in}\\) in m³/h.")

                    st.latex(r"PEC_{\mathrm{pump}} = \left(\log_{10}(\dot{W}_{P}) - 0.03195 \cdot \dot{W}_{P}^{2} + 467.2 \cdot \dot{W}_{P} + 20480\right) \cdot \frac{\mathrm{CEPCI}_{cur}}{\mathrm{CEPCI}_{ref}}")
                    st.caption("Pumpenkosten aus der Pumpenleistung \\(\\dot{W}_{P}\\) in kW.")

                    st.latex(r"\mathrm{LMTD} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}, \qquad A = \frac{\dot{Q}}{k \cdot \mathrm{LMTD}}")
                    st.latex(r"PEC_{\mathrm{HEX}} = 15{,}526 \left(\frac{A}{42}\right)^{0.8} \cdot \frac{\mathrm{CEPCI}_{cur}}{\mathrm{CEPCI}_{ref}}")
                    st.caption("Wärmeübertragerkosten aus Fläche \\(A\\); der U-Wert \\(k\\) wird je nach Bauteiltyp gesetzt.")

                    st.latex(r"PEC_{\mathrm{flash}} = 1{,}444 \left(\frac{V}{0.089}\right)^{0.63} \cdot \frac{\mathrm{CEPCI}_{cur}}{\mathrm{CEPCI}_{ref}}")
                    st.caption("Flashtankkosten aus dem abgeschätzten Behältervolumen \\(V\\) in m³.")
                    st.markdown(
                        """
                        `Z_k` ist die stündliche Kostenrate eines Bauteils. Dafür
                        werden zuerst aus `TCI` die Kapitalrückzahlung und die
                        jährlichen Betriebs- und Wartungskosten bestimmt. Danach
                        werden diese Gesamtkosten proportional zum PEC-Anteil des
                        jeweiligen Bauteils aufgeteilt und auf Vollbenutzungsstunden
                        bezogen.
                        """
                    )
                    st.caption("For more detailed information, please refer to the documentation.")
                    st.latex(r"a = \frac{i_{eff} - r_n}{1 - \left(\frac{1+r_n}{1+i_{eff}}\right)^n}")
                    st.latex(r"CCL = a \cdot \sum TCI_k, \qquad OMCL = f_{\mathrm{O\&M}} \cdot \sum TCI_k")
                    st.latex(r"Z_k = \frac{(CCL + OMCL)\left(\frac{PEC_k}{\sum PEC_k}\right)}{\tau_h}")
                    st.caption(
                        "Mit i_eff = 8 %, r_n = 2 %, n = 20 Jahren, "
                        "f_O&M = 3 % und τ_h = Volllaststunden pro Jahr."
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
                        i_eff=0.08,
                        r_n=0.02,
                        n=20,
                        omc_rel=0.03,
                        tci_factor=6.32
                    )

                    # ===============================
                    # Run exergoeconomic analysis
                    # ===============================
                    try:
                        df_execo_comp, df_mat1, df_mat2, df_non_mat, ean, Exe_Eco_Costs = run_exergoeconomic_from_hp(
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
                    except Exception as exc:
                        import traceback
                        st.error(f"❌ Exergoökonomische Analyse fehlgeschlagen:\n{exc}")
                        st.code(traceback.format_exc())  # <-- FULL traceback in UI
                        df_execo_comp = df_mat1 = df_mat2 = df_non_mat = None
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
                        'Case B — waste heat, return still above or is the same as ambient.\n\n'
                        'fuel = {"inputs": ["E0", "B1"], "outputs": []}\n'
                        'product = {"inputs": ["C3"], "outputs": ["C1"]}\n'
                        'loss = {"inputs": ["B3"], "outputs": []}',
                        language="python"
                    )
                    st.code(
                        'Case C — Spezieller Fall, waste heat, return still above ambient\n\n'
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
                        st.caption('!!Diese Darstellung ist aktuell noch in Bearbeitung')
                        try:
                            diagram_sankey = ss.hp.generate_sankey_diagram()
                            st.plotly_chart(diagram_sankey, use_container_width=True)
                        except Exception as e:
                            st.info(f"Sankey ausgelassen: {e}")

                    with col7:
                        st.subheader('Wasserfall Diagramm')
                        st.caption('!!Diese Darstellung ist aktuell noch in Bearbeitung')
                        try:
                            dia_wf_fig, _ = ss.hp.generate_waterfall_diagram(return_fig_ax=True)
                            st.pyplot(dia_wf_fig, use_container_width=True)
                        except Exception as e:
                            st.info(f"Wasserfall ausgelassen: {e}")

                st.write(
                    """
                    Definitionen und Methodik der Exergieanalyse basierend auf
                    [Morosuk und Tsatsaronis (2019)](https://doi.org/10.1016/j.energy.2018.10.090),
                    dessen Implementation in TESPy beschrieben in [Witte und Hofmann et al. (2022)](https://doi.org/10.3390/en15114087)
                    und didaktisch aufbereitet in [Witte, Freißmann und Fritz (2023)](https://fwitte.github.io/TESPy_teaching_exergy/).
                    """
                )

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
