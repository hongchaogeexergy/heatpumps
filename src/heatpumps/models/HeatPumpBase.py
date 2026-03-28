import json
import os
from datetime import datetime
from time import time
import tempfile
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from CoolProp.CoolProp import PropsSI as PSI
from fluprodia import FluidPropertyDiagram
from scipy.interpolate import interpn
from sklearn.linear_model import LinearRegression
from tespy.networks import Network
from tespy.tools.characteristics import CharLine
from tespy.tools.characteristics import load_default_char as ldc
from exerpy import ExergyAnalysis
from tespy.components import Compressor, Pump, Valve, HeatExchanger, CycleCloser, Source, Sink, PowerSource, PowerBus, Motor
from tespy.connections import Connection, PowerConnection, Ref
from ..economics.econ_params import EconParams, econ_defaults
from ..economics.economic_utils import run_full_economic_pipeline
from ..economics.exerpy_costing import determine_exergy_boundaries


class HeatPumpBase:
    """Super class of all concrete heat pump models."""

    def __init__(self, params):
        """Initialize model and set necessary attributes."""
        self.params = params

        self.nw = Network(
            T_unit='C', p_unit='bar', h_unit='kJ / kg', m_unit='kg / s'
        )

        self._init_fluids()

        self.comps = dict()
        self.conns = dict()

        self.cop = np.nan
        self.cop_lorenz = np.nan
        self.eta_lorenz = np.nan
        self.cop_carnot = np.nan
        self.eta_carnot = np.nan
        self.epsilon = np.nan
        self.solved_design = False

        # placeholders for economics/exergoeconomics
        self.econ = {}
        self.exergoecon = {}

        self._init_vals = {
            'm_dot_rel_econ_closed': 0.9,
            'dh_rel_comp': 1.15
        }

        self._init_dir_paths()

    @staticmethod
    def _normalize_eta(value: float) -> float:
        if value is None:
            raise ValueError("eta_s is None")
        v = float(value)
        if v > 1:   # assume percent
            v /= 100.0
        if not (0 < v <= 1):
            raise ValueError(f"eta_s out of range: {v}")
        print(f"[DEBUG] Normalizing eta_s: raw={value} → normalized={v}")
        return v

    def _init_fluids(self):
        """Initialize fluid attributes."""
        self.wf = self.params['fluids']['wf']
        self.si = self.params['fluids']['si']
        self.so = self.params['fluids']['so']

    def generate_components(self):
        """Initialize components of heat pump."""
        raise NotImplementedError

    def generate_connections(self):
        """Initialize and add connections and buses to network."""
        raise NotImplementedError

    def init_simulation(self, **kwargs):
        """Perform initial parametrization with starting values."""
        raise NotImplementedError

    def design_simulation(self, **kwargs):
        """Perform final parametrization and design simulation."""
        raise NotImplementedError

    def _solve_model(self, **kwargs):
        """Solve the model in design mode."""
        if 'iterinfo' in kwargs:
            self.nw.set_attr(iterinfo=kwargs['iterinfo'])
        self.solved_design = False
        self.nw.solve('design')

        if 'print_results' in kwargs:
            if kwargs['print_results']:
                self.nw.print_results()
        residual = self.nw.residual[-1] if len(self.nw.residual) else np.inf
        if np.isfinite(residual) and abs(residual) < 1e-3:
            self.solved_design = True
            self.nw.save(self.design_path)

    def _get_heat_output_W(self):
        """Return heat output in W with fallbacks for models without a consumer HEX."""
        cons = self.comps.get('cons')
        if cons is not None and hasattr(cons, 'Q'):
            Q_val = getattr(cons.Q, 'val', None)
            if Q_val is not None:
                return Q_val

        cond = self.comps.get('cond')
        if cond is not None and hasattr(cond, 'Q'):
            Q_val = getattr(cond.Q, 'val', None)
            if Q_val is not None:
                return Q_val

        trans = self.comps.get('trans')
        if trans is not None and hasattr(trans, 'Q'):
            Q_val = getattr(trans.Q, 'val', None)
            if Q_val is not None:
                return Q_val

        if 'C1' in self.conns and 'C2' in self.conns:
            m_sink = self.conns['C2'].m.val or self.conns['C1'].m.val
            h_in = self.conns['C1'].h.val
            h_out = self.conns['C2'].h.val
            if m_sink is not None and h_in is not None and h_out is not None:
                return m_sink * (h_out - h_in) * 1e3

        if 'C2' in self.conns and 'C3' in self.conns:
            m_sink = self.conns['C3'].m.val or self.conns['C2'].m.val
            h_in = self.conns['C2'].h.val
            h_out = self.conns['C3'].h.val
            if m_sink is not None and h_in is not None and h_out is not None:
                return m_sink * (h_out - h_in) * 1e3

        return np.nan

    def calc_efficiencies(self):
        """Calculate ideal and simulated cycle efficiencies."""
        # Simulated COP = heat output / power input
        Q_out = self._get_heat_output_W()
        self.Q_out = Q_out

        W_in = np.nan
        if 'E0' in self.conns:
            W_in = self.conns['E0'].E.val

        if W_in != 0 and not np.isnan(W_in):
            self.cop = abs(Q_out / W_in)
        else:
            self.cop = np.nan

        # Lorenz COP
        T_ln_source = (
            (self.params['B2']['T'] - self.params['B1']['T']) /
            (np.log(self.params['B2']['T'] + 273.15) - np.log(self.params['B1']['T'] + 273.15))
        )
        t_sink_hot = self.params.get('C2', self.params.get('C3'))['T']
        T_ln_sink = (
            (t_sink_hot - self.params['C1']['T']) /
            (np.log(t_sink_hot + 273.15) - np.log(self.params['C1']['T'] + 273.15))
        )
        self.cop_lorenz = T_ln_sink / (T_ln_sink - T_ln_source)
        if self.cop_lorenz != 0:
            self.eta_lorenz = self.cop / self.cop_lorenz

        # Carnot COP
        if 'cond' in self.params.keys():
            T_cond = t_sink_hot + self.params['cond']['ttd_u'] + 273.15
            T_evap = self.params['B2']['T'] - self.params['evap']['ttd_l'] + 273.15
            self.cop_carnot = T_cond / (T_cond - T_evap)
            if self.cop_carnot != 0:
                self.eta_carnot = self.cop / self.cop_carnot

    # added economic_analysis & econ_overrides
    def run_model(self, print_cop=False, exergy_analysis=True,
                  economic_analysis=False, econ_overrides=None, **kwargs):
        """Run the initialization and design simulation routine."""
        self.generate_components()
        self.generate_connections()
        self.init_simulation(**kwargs)
        self.design_simulation(**kwargs)
        if not self.solved_design:
            residual = self.nw.residual[-1] if len(self.nw.residual) else np.nan
            raise RuntimeError(
                f'TESPy design solve did not converge for '
                f'{self.params["setup"]["type"]} (residual={residual}).'
            )
        self.check_consistency()
        self.calc_efficiencies()
        if exergy_analysis:
            self.perform_exergy_analysis(**kwargs)

        if economic_analysis:
            self.run_economics(econ_overrides=econ_overrides)

        if print_cop:
            print(f'COP = {self.cop:.3f}')
            print(f'Lorenz COP = {self.cop_lorenz:.3f}')
            print(f'Lorenz \\eta = {self.eta_lorenz:.3f}')
            print(f'Carnot COP = {self.cop_carnot:.3f}')
            print(f'Carnot \\eta = {self.eta_carnot:.3f}')

    def create_ranges(self):
        """Create stable and base ranges for T_hs_ff, T_cons_ff and pl."""
        self.T_hs_ff_range = np.linspace(
            self.params['offdesign']['T_hs_ff_start'],
            self.params['offdesign']['T_hs_ff_end'],
            self.params['offdesign']['T_hs_ff_steps'],
            endpoint=True
        ).round(decimals=3)
        half_len_hs = int(len(self.T_hs_ff_range)/2) - 1
        self.T_hs_ff_stablerange = np.concatenate([
            self.T_hs_ff_range[half_len_hs::-1],
            self.T_hs_ff_range,
            self.T_hs_ff_range[:half_len_hs:-1]
        ])

        self.T_cons_ff_range = np.linspace(
            self.params['offdesign']['T_cons_ff_start'],
            self.params['offdesign']['T_cons_ff_end'],
            self.params['offdesign']['T_cons_ff_steps'],
            endpoint=True
        ).round(decimals=3)
        half_len_cons = int(len(self.T_cons_ff_range)/2) - 1
        self.T_cons_ff_stablerange = np.concatenate([
            self.T_cons_ff_range[half_len_cons::-1],
            self.T_cons_ff_range,
            self.T_cons_ff_range[:half_len_cons:-1]
        ])

        self.pl_range = np.linspace(
            self.params['offdesign']['partload_min'],
            self.params['offdesign']['partload_max'],
            self.params['offdesign']['partload_steps'],
            endpoint=True
        ).round(decimals=3)
        self.pl_stablerange = np.concatenate(
            [self.pl_range[::-1], self.pl_range]
        )

    def df_to_array(self, results_offdesign):
        """Create 3D arrays of heat output, power input and epsilon from DataFrame."""
        self.Q_array = []
        self.P_array = []
        self.epsilon_array = []
        for i, T_hs_ff in enumerate(self.T_hs_ff_range):
            self.Q_array.append([])
            self.P_array.append([])
            self.epsilon_array.append([])
            for T_cons_ff in self.T_cons_ff_range:
                self.Q_array[i].append(
                    results_offdesign.loc[(T_hs_ff, T_cons_ff), 'Q'].tolist()
                )
                self.P_array[i].append(
                    results_offdesign.loc[(T_hs_ff, T_cons_ff), 'P'].tolist()
                )
                self.epsilon_array[i].append(
                    results_offdesign.loc[(T_hs_ff, T_cons_ff), 'epsilon'].tolist()
                )

    def get_pressure_levels(self, T_evap, T_cond, wf=None):
        """Calculate evaporation, condensation and middle pressure in bar."""
        if not wf:
            wf = self.wf
        p_evap = PSI(
            'P', 'Q', 1,
            'T', T_evap - self.params['evap']['ttd_l'] + 273.15,
            wf
        ) * 1e-5
        p_cond = PSI(
            'P', 'Q', 0,
            'T', T_cond + self.params['cond']['ttd_u'] + 273.15,
            wf
        ) * 1e-5
        p_mid = np.sqrt(p_evap * p_cond)

        return p_evap, p_cond, p_mid

    # === DEPRECATED: use self.run_economics(...) instead ===
    def calc_cost(self, *args, **kwargs):
        """
        DEPRECATED. Kept for backward compatibility only.
        Please call `self.run_economics(econ_overrides=...)` instead.
        """
        import warnings
        warnings.warn("HeatPumpBase.calc_cost is deprecated. "
                      "Use HeatPumpBase.run_economics instead.", DeprecationWarning)
        return None

    # === DEPRECATED helper (unused) ===
    def eval_costfunc(self, *args, **kwargs):
        """
        DEPRECATED. Cost functions are implemented centrally in
        `economic_utils.run_full_economic_pipeline`.
        """
        import warnings
        warnings.warn("HeatPumpBase.eval_costfunc is deprecated.", DeprecationWarning)
        return None
    # workaround

    def _exerpy_export_json_dict(self, ean) -> dict:
        """
        Baut ein ExerPy-kompatibles JSON-Dict aus dem ExerPy-Objekt,
        das via from_tespy erzeugt wurde.
        Nutzt interne Datenstrukturen, die bei dir bereits existieren
        (du greifst später auch auf self.ean._component_data zu).
        """
        # ExerPy hält die geparsten Daten typischerweise in privaten Feldern.
        # Wir kopieren sie in ein JSON-fähiges dict.
        data = {}

        # Manche ExerPy-Versionen nutzen diese Felder:
        if hasattr(ean, "_component_data"):
            data["components"] = deepcopy(ean._component_data)
        if hasattr(ean, "_connection_data"):
            data["connections"] = deepcopy(ean._connection_data)

        # Fallback: falls ExerPy andere Namen nutzt, kann man hier später erweitern.
        if "components" not in data or "connections" not in data:
            raise RuntimeError(
                "Kann ExerPy-Daten nicht exportieren: _component_data/_connection_data fehlen. "
                "Bitte zeig mir kurz `dir(ean)` dann passe ich den Export an."
            )

        # Ambient conditions optional (nicht zwingend, aber sauber)
        data["ambient_conditions"] = {"Tamb": float(ean.Tamb), "pamb": float(ean.pamb)}

        return data

    @staticmethod
    def _exerpy_patch_condenser_as_hx(data: dict, condenser_label: str = "Condenser") -> dict:
        """
        Verschiebt EINEN Eintrag aus components['Condenser'][label] nach components['HeatExchanger'][label].
        """
        import copy
        patched = copy.deepcopy(data)

        comps = patched.get("components", {})
        if "Condenser" not in comps:
            return patched  # nichts zu patchen

        cond_group = comps["Condenser"]
        if condenser_label not in cond_group:
            return patched  # Name weicht ab → später ggf. erweitern

        hx_group = comps.setdefault("HeatExchanger", {})
        hx_group[condenser_label] = cond_group[condenser_label]

        del cond_group[condenser_label]
        if len(cond_group) == 0:
            del comps["Condenser"]

        return patched

    def perform_exergy_analysis(self, print_results=False, **kwargs):
        T0 = self.params['ambient']['T'] + 273.15  # K
        p0 = self.params['ambient']['p'] * 1e5     # Pa

        # 1) ExerPy parse aus TESPy (TESPy-Modell bleibt Condenser!)
        ean_raw = ExergyAnalysis.from_tespy(self.nw, T0, p0, split_physical_exergy=True)

        # 2) Export als JSON-dict
        raw_dict = self._exerpy_export_json_dict(ean_raw)

        # 3) Patch: Condenser wird für ExerPy als HeatExchanger behandelt
        patched_dict = self._exerpy_patch_condenser_as_hx(raw_dict, condenser_label="Condenser")

        # 4) Schreibe gepatchtes JSON in eine Temp-Datei und lade via from_json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(patched_dict, f, indent=2)
            patched_path = f.name

        self.ean = ExergyAnalysis.from_json(patched_path, split_physical_exergy=True)

        # ---------------------------
        # Ab hier: STABIL & EINFACH
        # ---------------------------
        boundaries = determine_exergy_boundaries(self)
        fuel = boundaries["fuel"]
        product = boundaries["product"]
        loss = boundaries["loss"]
        self.exergy_boundary_scenario = boundaries.get("scenario")

        self.ean.analyse(E_F=fuel, E_P=product, E_L=loss)

        self.exergy_boundaries = {"fuel": fuel, "product": product, "loss": loss}

        if print_results:
            self.ean.print_results(**kwargs)

        # totals
        self.epsilon = self.ean.epsilon
        self.E_F = self.ean.E_F
        self.E_P = self.ean.E_P
        self.E_L = self.ean.E_L
        self.E_D = self.ean.E_D

        # component table (ExerPy liefert bei dir tuple → so behandeln!)
        self.component_exergy_df = None
        try:
            df_comp, df_mat, df_non_mat = self.ean.exergy_results(print_results=False)
            df_comp = df_comp.copy()

            # optional rounding
            for col in ["E_F [kW]", "E_P [kW]", "E_D [kW]", "E_L [kW]"]:
                if col in df_comp.columns:
                    df_comp[col] = df_comp[col].round(3)

            self.component_exergy_df = df_comp

            # Debug: Condenser-Zeile robust ausgeben
            try:
                if "Condenser" in df_comp.index:
                    print("[DEBUG] Condenser row:", df_comp.loc["Condenser"])
                elif "Component" in df_comp.columns:
                    print(df_comp[df_comp["Component"].astype(str).str.lower() == "condenser"])
            except Exception as e:
                print("[DEBUG] Condenser debug print failed:", e)

        except Exception as e:
            print("Exergy component data not available:", e)
            self.component_exergy_df = None


    # Build EconParams from defaults + model + overrides
    def build_econ_params(self, econ_overrides=None) -> EconParams:
        base = dict(econ_defaults)  # copy defaults
        model_block = self.params.get('economic', {}) or {}
        base.update(model_block)
        if econ_overrides:
            base.update(econ_overrides)

        # ensure strings
        if 'ref_year' in base and isinstance(base['ref_year'], int):
            base['ref_year'] = str(base['ref_year'])
        if 'current_year' in base and isinstance(base['current_year'], int):
            base['current_year'] = str(base['current_year'])

        return EconParams.from_dict(base)

    def run_economics(self, econ_overrides=None):
        if not self.solved_design:
            raise RuntimeError(
                "Economics require a solved design. Run design_simulation first."
            )

        # Build merged EconParams and run the central pipeline
        econ_params = self.build_econ_params(econ_overrides)
        results = run_full_economic_pipeline(hp=self, econ=econ_params)

        # Filter what you display in the dashboard
        actual_labels = {c.label for c in self.comps.values()}
        exclude_helpers = ["CycleCloser", "Source", "Sink", "PowerBus", "PowerSource"]

        def _filter_dict(d: dict):
            return {
                k: v for k, v in d.items()
                if (k in actual_labels) and not any(ex in k for ex in exclude_helpers)
            }

        if results:
            results["capex_breakdown"] = _filter_dict(results.get("capex_breakdown", {}))
            results["opex_breakdown"]  = _filter_dict(results.get("opex_breakdown", {}))
            results["capex_equipment"] = _filter_dict(results.get("capex_equipment", {}))

        # Store for UI
        self.econ = {
            "capex_breakdown": results.get("capex_breakdown", {}),
            "capex":           results.get("capex_breakdown", {}),
            "capex_total":     results.get("capex_total"),
            "capex_equipment": results.get("capex_equipment"),
            "opex_breakdown":  results.get("opex_breakdown", {}),
            "opex":            results.get("opex_breakdown", {}),
            "opex_total":      results.get("opex_total"),
            "totals":          results.get("totals", {}),
            "kpis":            results.get("kpis", {}),
            "assumptions":     results.get("assumptions", {}),
        }

        # Pass-through exergoeconomics from the pipeline (do not run ExerPy again here)
        self.exergoecon = results.get("exergoeconomics", {}) or {}

        return results

    def get_plotting_states(self):
        """Generate data of states to plot in state diagram."""
        return {}

    def generate_state_diagram(self, refrig='', diagram_type='logph',
                               style='light', figsize=(16, 10), fontsize=10,
                               legend=True, legend_loc='upper left',
                               return_diagram=False, savefig=False,
                               open_file=False, filepath=None, **kwargs):
        """
        Generate log(p)-h-diagram of heat pump process.
        """
        if not refrig:
            refrig = self.params['setup']['refrig']
        # axis and isoline variables
        if diagram_type == 'logph':
            var = {'x': 'h', 'y': 'p', 'isolines': ['T', 's']}
        elif diagram_type == 'Ts':
            var = {'x': 's', 'y': 'T', 'isolines': ['h', 'p']}
        else:
            print('Parameter "diagram_type" has to be "logph" or "Ts".')
            return

        # Get plotting state data
        result_dict = self.get_plotting_states(**kwargs)
        if len(result_dict) == 0:
            print("'get_plotting_states' not implemented for this heat pump model.")
            return

        if style == 'light':
            plt.style.use('default')
            isoline_data = None
        elif style == 'dark':
            plt.style.use('dark_background')
            isoline_data = {
                'T': {'style': {'color': 'dimgrey'}},
                'v': {'style': {'color': 'dimgrey'}},
                'Q': {'style': {'color': '#FFFFFF'}},
                'h': {'style': {'color': 'dimgrey'}},
                'p': {'style': {'color': 'dimgrey'}},
                's': {'style': {'color': 'dimgrey'}}
            }

        # Initialize diagram
        fig, ax = plt.subplots(figsize=figsize)

        diagram_data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'input', 'diagrams', f"{refrig}.json"
        ))

        # Generate isolines
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'input', 'state_diagram_config.json'
        ))
        with open(path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        if refrig in config:
            state_props = config[refrig]
        else:
            state_props = config['MISC']

        if os.path.isfile(diagram_data_path):
            diagram = FluidPropertyDiagram.from_json(diagram_data_path)
        else:
            diagram = FluidPropertyDiagram(refrig)
            diagram.set_unit_system(T='°C', p='bar', h='kJ/kg')

            iso1 = np.arange(
                state_props[var['isolines'][0]]['isorange_low'],
                state_props[var['isolines'][0]]['isorange_high'],
                state_props[var['isolines'][0]]['isorange_step']
            )
            iso2 = np.arange(
                state_props[var['isolines'][1]]['isorange_low'],
                state_props[var['isolines'][1]]['isorange_high'],
                state_props[var['isolines'][1]]['isorange_step']
            )

            diagram.set_isolines(**{
                var['isolines'][0]: iso1,
                var['isolines'][1]: iso2
            })
            diagram.calc_isolines()
            diagram.to_json(diagram_data_path)

        # Calculate components process data
        for compdata in result_dict.values():
            compdata['datapoints'] = (
                diagram.calc_individual_isoline(**compdata)
            )
        diagram.fig = fig
        diagram.ax = ax

        # Set axes limits
        if 'xlims' in kwargs:
            xlims = kwargs['xlims']
        else:
            xlims = (state_props[var['x']]['min'], state_props[var['x']]['max'])
        if 'ylims' in kwargs:
            ylims = kwargs['ylims']
        else:
            ylims = (state_props[var['y']]['min'], state_props[var['y']]['max'])

        diagram.draw_isolines(
            diagram_type=diagram_type, fig=fig, ax=ax,
            x_min=xlims[0], x_max=xlims[1], y_min=ylims[0], y_max=ylims[1],
            isoline_data=isoline_data
        )

        # Draw process
        for i, key in enumerate(result_dict.keys()):
            datapoints = result_dict[key]['datapoints']
            has_xvals = len(datapoints[var['x']]) > 0
            has_yvals = len(datapoints[var['y']]) > 0
            if has_xvals and has_yvals:
                ax.plot(
                    datapoints[var['x']][:], datapoints[var['y']][:],
                    color='#EC6707'
                )
                ax.scatter(
                    datapoints[var['x']][0], datapoints[var['y']][0],
                    color='#B54036',
                    label=f'$\\bf{i+1:.0f}$: {key}',
                    s=14*int(fontsize*0.9), alpha=0.5
                )
                ax.annotate(
                    f'{i+1:.0f}',
                    (datapoints[var['x']][0], datapoints[var['y']][0]),
                    ha='center', va='center', color='w',
                    fontsize=int(fontsize*0.9)
                )
            else:
                ax.scatter(0, 0, color='#FFFFFF', s=0, alpha=1.0,
                           label=f'$\\bf{i+1:.0f}$: {key}')
                ax.annotate(
                    'Error\nMissing Plotting Data', (0.5, 0.5),
                    xycoords='axes fraction', ha='center', va='center',
                    fontsize=60, color='#B54036'
                )

        # Labels
        ax.set_title(refrig, fontsize=int(fontsize*1.2))
        if diagram_type == 'logph':
            ax.set_xlabel('Spezifische Enthalpie in $kJ/kg$', fontsize=fontsize)
            ax.set_ylabel('Druck in $bar$', fontsize=fontsize)
        elif diagram_type == 'Ts':
            ax.set_xlabel('Spezifische Entropie in $kJ/(kg \\cdot K)$', fontsize=fontsize)
            ax.set_ylabel('Temperatur in $°C$', fontsize=fontsize)

        ax.tick_params(axis='both', labelsize=int(fontsize*0.9))

        if legend:
            ax.legend(
                loc=legend_loc,
                prop={'size': fontsize * (1 - 0.02 * len(result_dict))},
                markerscale=(1 - 0.02 * len(result_dict))
            )

        if savefig:
            if filepath is None:
                filename = f'logph_{self.params["setup"]["type"]}_{refrig}.pdf'
                filepath = os.path.abspath(os.path.join(os.getcwd(), filename))

            plt.tight_layout()
            plt.savefig(filepath, dpi=300)

            if open_file:
                os.startfile(filepath)

        if return_diagram:
            return diagram

    def generate_sankey_diagram(self, width=None, height=None):
        """Custom Sankey Diagram of Heat Pump model based on ExergyAnalysis results."""
        import plotly.graph_objects as go

        nodes = [
            'Electrical Input (E_F)',  # 0
            'Exergy Loss (E_L)',       # 1
            'Exergy Destruction (E_D)',# 2
            'Useful Output (E_P)'      # 3
        ]

        colors = {
            'E_F': '#00395B',
            'E_P': '#B54036',
            'E_L': '#EC6707',
            'E_D': '#EC6707'
        }

        E_F = self.ean.E_F
        E_P = self.ean.E_P
        E_L = self.ean.E_L
        E_D = self.ean.E_D

        links = dict(
            source=[0, 0, 0],
            target=[1, 2, 3],
            value=[E_L, E_D, E_P],
            label=['E_L', 'E_D', 'E_P'],
            color=[colors['E_L'], colors['E_D'], colors['E_P']]
        )

        fig = go.Figure(
            go.Sankey(
                arrangement='snap',
                node={'label': nodes, 'pad': 15, 'color': '#EC6707'},
                link=dict(
                    source=links['source'],
                    target=links['target'],
                    value=links['value'],
                    label=links['label'],
                    color=links['color']
                )
            )
        )

        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)

        fig.update_layout(title_text="Exergy Flow Sankey Diagram", font_size=12)
        return fig

    def generate_waterfall_diagram(self, figsize=(16, 10), legend=True,
                                   return_fig_ax=False, show_epsilon=True):
        """Generates waterfall diagram of exergy analysis (manual version)."""
        comps = ['Fuel Exergy']
        E_F_total = self.ean.E_F
        E_D = [0]
        E_P = [E_F_total]

        comp_data = self.ean._component_data.get('component_exergy', None)
        if comp_data is not None:
            comp_data = comp_data.dropna(subset=['E_D'])
            for comp in comp_data.sort_values(by='E_D', ascending=False).index:
                if comp_data.at[comp, 'E_D'] > 1:
                    comps.append(comp)
                    E_D.append(comp_data.at[comp, 'E_D'])
                    E_F_total -= comp_data.at[comp, 'E_D']
                    E_P.append(E_F_total)

        comps.append('Product Exergy')
        E_D.append(0)
        E_P.append(E_F_total)

        # Convert to kW
        E_D = [e * 1e-3 for e in E_D]
        E_P = [e * 1e-3 for e in E_P]

        colors_E_P = ['#74ADC0'] * len(comps)
        colors_E_P[0] = '#00395B'
        colors_E_P[-1] = '#B54036'

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(np.arange(len(comps)), E_P, align='center', color=colors_E_P)
        ax.barh(np.arange(len(comps)), E_D, align='center', left=E_P, label='E_D', color='#EC6707')

        if legend:
            ax.legend()

        if show_epsilon:
            epsilon_val = getattr(self.ean, 'epsilon', None)
            if epsilon_val is not None:
                ax.annotate(
                    rf'$\epsilon_{{tot}} = ${epsilon_val:.3f}',
                    (0.96, 0.06), xycoords='axes fraction',
                    ha='right', va='center', color='k',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white')
                )

        ax.set_xlabel('Exergie in kW')
        ax.set_yticks(np.arange(len(comps)))
        ax.set_yticklabels(comps)
        ax.set_xlim([0, max(E_P) + 1000])
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        ax.invert_yaxis()

        if return_fig_ax:
            return fig, ax

    def calc_partload_char(self, **kwargs):
        """
        Interpolate data points of heat output, power input and epsilon.
        """
        necessary_params = [
            'Q_array', 'P_array', 'epsilon_array', 'pl_range', 'T_hs_ff_range',
            'T_cons_ff_range'
        ]
        if len(kwargs):
            for nec_param in necessary_params:
                if nec_param not in kwargs:
                    raise KeyError(
                        f'Necessary parameter {nec_param} not '
                        + 'in kwargs. The necessary parameters'
                        + f' are: {necessary_params}'
                    )
            Q_array = np.asarray(kwargs['Q_array'])
            P_array = np.asarray(kwargs['P_array'])
            epsilon_array = np.asarray(kwargs['epsilon_array'])
            pl_range = kwargs['pl_range']
            T_hs_ff_range = kwargs['T_hs_ff_range']
            T_cons_ff_range = kwargs['T_cons_ff_range']
        else:
            for nec_param in necessary_params:
                if nec_param not in self.__dict__:
                    raise AttributeError(
                        f'Necessary parameter {nec_param} can '
                        + 'not be found in the instances '
                        + 'attributes. Please make sure to '
                        + 'perform the offdesign_simulation '
                        + 'method or provide the necessary '
                        + 'parameters as kwargs. These are: '
                        + f'{necessary_params}'
                    )
            Q_array = np.asarray(self.Q_array)
            P_array = np.asarray(self.P_array)
            epsilon_array = np.asarray(self.epsilon_array)
            pl_range = self.pl_range
            T_hs_ff_range = self.T_hs_ff_range
            T_cons_ff_range = self.T_cons_ff_range

        pl_step = 0.01
        T_hs_ff_step = 1
        T_cons_ff_step = 1

        pl_fullrange = np.arange(
            pl_range[0],
            pl_range[-1]+pl_step,
            pl_step
        )
        T_hs_ff_fullrange = np.arange(
            T_hs_ff_range[0], T_hs_ff_range[-1]+T_hs_ff_step, T_hs_ff_step
        )
        T_cons_ff_fullrange = np.arange(
            T_cons_ff_range[0], T_cons_ff_range[-1]+T_cons_ff_step,
            T_cons_ff_step
        )

        multiindex = pd.MultiIndex.from_product(
            [T_hs_ff_fullrange, T_cons_ff_fullrange, pl_fullrange],
            names=['T_hs_ff', 'T_cons_ff', 'pl']
        )

        partload_char = pd.DataFrame(
            index=multiindex, columns=['Q', 'P', 'COP', 'epsilon']
        )

        for T_hs_ff in T_hs_ff_fullrange:
            for T_cons_ff in T_cons_ff_fullrange:
                for pl in pl_fullrange:
                    partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'Q'] = abs(
                        interpn(
                            (T_hs_ff_range, T_cons_ff_range, pl_range),
                            Q_array,
                            (round(T_hs_ff, 3), round(T_cons_ff, 3),
                             round(pl, 3)),
                            bounds_error=False
                        )[0]
                    )
                    partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'P'] = interpn(
                        (T_hs_ff_range, T_cons_ff_range, pl_range),
                        P_array,
                        (round(T_hs_ff, 3), round(T_cons_ff, 3), round(pl, 3)),
                        bounds_error=False
                    )[0]
                    partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'COP'] = (
                        partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'Q']
                        / partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'P']
                    )
                    partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'epsilon'] = interpn(
                        (T_hs_ff_range, T_cons_ff_range, pl_range),
                        epsilon_array,
                        (round(T_hs_ff, 3), round(T_cons_ff, 3), round(pl, 3)),
                        bounds_error=False
                    )[0]

        return partload_char

    def linearize_partload_char(self, partload_char, variable='P',
                                line_type='offset', regression_type='OLS',
                                normalize=None):
        """
        Linearize partload characteristic for usage in MILP problems.
        """
        cols = [f'{variable}_max', f'{variable}_min']
        if line_type == 'origin':
            cols += ['COP']
        elif line_type == 'offset':
            cols += ['c_1', 'c_0']

        T_hs_ff_range = set(
            partload_char.index.get_level_values('T_hs_ff')
        )
        T_cons_ff_range = set(
            partload_char.index.get_level_values('T_cons_ff')
        )

        multiindex = pd.MultiIndex.from_product(
            [T_hs_ff_range, T_cons_ff_range],
            names=['T_hs_ff', 'T_cons_ff']
        )
        linear_model = pd.DataFrame(index=multiindex, columns=cols)

        if variable == 'P':
            resp_variable = 'Q'
        elif variable == 'Q':
            resp_variable = 'P'
        else:
            raise ValueError(
                f"Argument {variable} for parameter 'variable' is not valid."
                + "Choose either 'P' or 'Q'."
            )

        for T_hs_ff in T_hs_ff_range:
            for T_cons_ff in T_cons_ff_range:
                idx = (T_hs_ff, T_cons_ff)
                linear_model.loc[idx, f'{variable}_max'] = (
                    partload_char.loc[idx, variable].max()
                )
                linear_model.loc[idx, f'{variable}_min'] = (
                    partload_char.loc[idx, variable].min()
                )
                if regression_type == 'MinMax':
                    if line_type == 'origin':
                        linear_model.loc[idx, 'COP'] = (
                            partload_char.loc[idx, 'Q'].max()
                            / partload_char.loc[idx, 'P'].max()
                        )
                    elif line_type == 'offset':
                        linear_model.loc[idx, 'c_1'] = (
                            (partload_char.loc[idx, 'Q'].max()
                             - partload_char.loc[idx, 'Q'].min())
                            / (partload_char.loc[idx, 'P'].max()
                               - partload_char.loc[idx, 'P'].min())
                        )
                        linear_model.loc[idx, 'c_0'] = (
                            partload_char.loc[idx, 'Q'].max()
                            - partload_char.loc[idx, 'P'].max()
                            * linear_model.loc[idx, 'c_1']
                        )
                elif regression_type == 'OLS':
                    regressor = partload_char.loc[idx, variable].to_numpy().reshape(-1, 1)
                    response = partload_char.loc[idx, resp_variable].to_numpy()
                    if line_type == 'origin':
                        LinReg = LinearRegression(fit_intercept=False).fit(regressor, response)
                        linear_model.loc[idx, 'COP'] = LinReg.coef_[0]
                    elif line_type == 'offset':
                        LinReg = LinearRegression().fit(regressor, response)
                        linear_model.loc[idx, 'c_1'] = LinReg.coef_[0]
                        linear_model.loc[idx, 'c_0'] = LinReg.intercept_

        if normalize:
            variable_nom = partload_char.loc[
                (np.round(normalize['T_hs_ff'], 3),
                 np.round(normalize['T_cons_ff'], 3)),
                variable
            ].max()

            linear_model[f'{variable}_max'] /= variable_nom
            linear_model[f'{variable}_min'] /= variable_nom
            if line_type == 'offset':
                linear_model['c_0'] /= variable_nom
                linear_model['c_1'] /= variable_nom

        return linear_model

    def arrange_char_timeseries(self, linear_model, temp_ts):
        """
        Arrange a timeseries of the characteristics based on temperature data.
        """
        char_ts = pd.DataFrame(index=temp_ts.index, columns=linear_model.columns)
        for i in temp_ts.index:
            try:
                char_ts.loc[i, :] = linear_model.loc[
                    (temp_ts.loc[i, 'T_hs_ff'], temp_ts.loc[i, 'T_cons_ff']), :
                ]
            except KeyError:
                print(temp_ts.loc[i, 'T_cons_ff'], 'not in linear_model.')
                T_cons_ff_range = linear_model.index.get_level_values('T_cons_ff')
                if temp_ts.loc[i, 'T_cons_ff'] < min(T_cons_ff_range):
                    multi_idx = (temp_ts.loc[i, 'T_hs_ff'], min(T_cons_ff_range))
                elif temp_ts.loc[i, 'T_cons_ff'] > max(T_cons_ff_range):
                    multi_idx = (temp_ts.loc[i, 'T_hs_ff'], max(T_cons_ff_range))
                char_ts.loc[i, :] = linear_model.loc[multi_idx, :]

        return char_ts

    def plot_partload_char(self, partload_char, cmap_type='', cmap='viridis',
                           return_fig_ax=False, savefig=False, open_file=False):
        """
        Plot the partload characteristic of the heat pump.
        """
        if not cmap_type:
            print('Please provide a cmap_type of "T_cons_ff" or "COP" or "epsilon".')
            return

        colormap = plt.get_cmap(cmap)
        T_hs_ff_range = set(partload_char.index.get_level_values('T_hs_ff'))

        if cmap_type == 'T_cons_ff':
            colors = colormap(
                np.linspace(
                    0, 1,
                    len(set(partload_char.index.get_level_values('T_cons_ff')))
                )
            )
            figs = {}
            axes = {}
            for T_hs_ff in T_hs_ff_range:
                fig, ax = plt.subplots(figsize=(9.5, 6))

                T_cons_ff_range = set(partload_char.index.get_level_values('T_cons_ff'))
                for i, T_cons_ff in enumerate(T_cons_ff_range):
                    ax.plot(
                        partload_char.loc[(T_hs_ff, T_cons_ff), 'P'],
                        partload_char.loc[(T_hs_ff, T_cons_ff), 'Q'],
                        color=colors[i]
                    )

                ax.grid()
                sm = plt.cm.ScalarMappable(
                    cmap=colormap, norm=plt.Normalize(
                        vmin=np.min(partload_char.index.get_level_values('T_cons_ff')),
                        vmax=np.max(partload_char.index.get_level_values('T_cons_ff'))
                    )
                )
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Senkentemperatur in $°C$')
                ax.set_xlim(0, partload_char['P'].max() * 1.05)
                ax.set_ylim(0, partload_char['Q'].max() * 1.05)
                ax.set_xlabel('Elektrische Leistung $P$ in $MW$')
                ax.set_ylabel('Wärmestrom $\\dot{{Q}}$ in $MW$')
                ax.set_title(f'Quellentemperatur: {T_hs_ff:.0f} °C')
                figs[T_hs_ff] = fig
                axes[T_hs_ff] = ax

        if cmap_type == 'COP':
            figs = {}
            axes = {}
            for T_hs_ff in T_hs_ff_range:
                fig, ax = plt.subplots(figsize=(9.5, 6))

                scatterplot = ax.scatter(
                    partload_char.loc[(T_hs_ff), 'P'],
                    partload_char.loc[(T_hs_ff), 'Q'],
                    c=partload_char.loc[(T_hs_ff), 'COP'],
                    cmap=colormap,
                    vmin=(partload_char['COP'].min() - partload_char['COP'].max() * 0.05),
                    vmax=(partload_char['COP'].max() + partload_char['COP'].max() * 0.05)
                )

                cbar = plt.colorbar(scatterplot, ax=ax)
                cbar.set_label('Leistungszahl $COP$')

                ax.grid()
                ax.set_xlim(0, partload_char['P'].max() * 1.05)
                ax.set_ylim(0, partload_char['Q'].max() * 1.05)
                ax.set_xlabel('Elektrische Leistung $P$ in $MW$')
                ax.set_ylabel('Wärmestrom $\\dot{{Q}}$ in $MW$')
                ax.set_title(f'Quellentemperatur: {T_hs_ff:.0f} °C')
                figs[T_hs_ff] = fig
                axes[T_hs_ff] = ax

        if cmap_type == 'epsilon':
            figs = {}
            axes = {}
            for T_hs_ff in T_hs_ff_range:
                fig, ax = plt.subplots(figsize=(9.5, 6))
                scatterplot = ax.scatter(
                    partload_char.loc[T_hs_ff, 'P'],
                    partload_char.loc[T_hs_ff, 'Q'],
                    c=partload_char.loc[T_hs_ff, 'epsilon'],
                    cmap=colormap,
                    vmin=(partload_char['epsilon'].min() - partload_char['epsilon'].max() * 0.05),
                    vmax=(partload_char['epsilon'].max() + partload_char['epsilon'].max() * 0.05)
                )

                cbar = plt.colorbar(scatterplot, ax=ax)
                cbar.set_label('Exergetische Effizienz $ε$')

                ax.grid()
                ax.set_xlim(0, partload_char['P'].max() * 1.05)
                ax.set_ylim(0, partload_char['Q'].max() * 1.05)
                ax.set_xlabel('Elektrische Leistung $P$ in $MW$')
                ax.set_ylabel('Wärmestrom $\\dot{{Q}}$ in $MW$')
                ax.set_title(f'Quellentemperatur: {T_hs_ff:.0f} °C')
                figs[T_hs_ff] = fig
                axes[T_hs_ff] = ax

        if savefig:
            try:
                filename = (
                    f'partload_{cmap_type}_{self.params["setup"]["type"]}_'
                    + f'{self.params["setup"]["refrig"]}.pdf'
                )
            except KeyError:
                filename = (
                    f'partload_{cmap_type}_{self.params["setup"]["type"]}_'
                    + f'{self.params["setup"]["refrig1"]}_and_'
                    + f'{self.params["setup"]["refrig2"]}.pdf'
                )
            filepath = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 'output', filename
            ))
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)

            if open_file:
                os.startfile(filepath)

        elif return_fig_ax:
            return figs, axes
        else:
            plt.show()

    def _init_dir_paths(self):
        """Initialize paths and directories."""
        self.subdirname = (
            f"{self.params['setup']['type']}_"
            + f"{self.params['setup']['refrig'].replace('::', '_')}"
        )
        self.design_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'stable', f'{self.subdirname}_design.json'
        ))
        self.validate_dir()

    def validate_dir(self):
        """Check for a 'stable' directory and create it if necessary."""
        stablepath = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'stable'
        ))
        if not os.path.exists(stablepath):
            os.mkdir(stablepath)
        outputpath = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'output'
        ))
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)

    def check_consistency(self):
        """Perform all necessary checks to protect consistency of parameters."""
        self.check_thermodynamic_results()

    def check_thermodynamic_results(self):
        """Perform thermodynamic checks of the main cycle components."""
        user_help_prompt = (
            'Please check the heat pump parameters and model for thermodynamic'
            + ' plausibility.'
        )

        mask_neg_m_dot = self.nw.results['Connection']['m'] < 0
        if any(mask_neg_m_dot):
            conns_neg_m_dot = [
                idx for idx
                in self.nw.results['Connection'].loc[mask_neg_m_dot, 'm'].index
            ]
            raise ValueError(
                f'Mass flow in connection(s) {conns_neg_m_dot} is negative. '
                + user_help_prompt
            )

        if 'HeatExchanger' in self.nw.results:
            mask_heatex_pos_Q_dot = self.nw.results['HeatExchanger']['Q'] > 0
            if any(mask_heatex_pos_Q_dot):
                heatex_pos_Q_dot = [
                    idx for idx
                    in self.nw.results['HeatExchanger'].loc[
                        mask_heatex_pos_Q_dot, 'Q'
                    ].index
                ]
                raise ValueError(
                    f'Heat flow in HeatExchanger(s) {heatex_pos_Q_dot} is '
                    + f'positive, indicating flow form cold to hot side. '
                    + user_help_prompt
                )

            mask_heatex_neg_ttd_u = (
                self.nw.results['HeatExchanger']['ttd_u'] <= 0
            )
            if any(mask_heatex_neg_ttd_u):
                heatex_neg_ttd_u = [
                    idx for idx
                    in self.nw.results['HeatExchanger'].loc[
                        mask_heatex_neg_ttd_u, 'ttd_u'
                    ].index
                ]
                raise ValueError(
                    'Upper terminal temperature difference in HeatExchanger(s)'
                    + f' {heatex_neg_ttd_u} is not positive. '
                    + user_help_prompt
                )

            mask_heatex_neg_ttd_l = (
                self.nw.results['HeatExchanger']['ttd_l'] <= 0
            )
            if any(mask_heatex_neg_ttd_u):
                heatex_neg_ttd_l = [
                    idx for idx
                    in self.nw.results['HeatExchanger'].loc[
                        mask_heatex_neg_ttd_l, 'ttd_l'
                    ].index
                ]
                raise ValueError(
                    'Lower terminal temperature difference in HeatExchanger(s)'
                    + f' {heatex_neg_ttd_l} is not positive. '
                    + user_help_prompt
                )

        if 'Condenser' in self.nw.results:
            mask_cond_pos_Q_dot = self.nw.results['Condenser']['Q'] > 0
            if any(mask_cond_pos_Q_dot):
                cond_pos_Q_dot = [
                    idx for idx
                    in self.nw.results['Condenser'].loc[
                        mask_cond_pos_Q_dot, 'Q'
                    ].index
                ]
                raise ValueError(
                    f'Heat flow in Condenser(s) {cond_pos_Q_dot} is '
                    + 'positive, indicating flow form cold to hot side. '
                    + user_help_prompt
                )

            mask_cond_neg_ttd_u = (
                self.nw.results['Condenser']['ttd_u'] <= 0
            )
            if any(mask_cond_neg_ttd_u):
                cond_neg_ttd_u = [
                    idx for idx
                    in self.nw.results['Condenser'].loc[
                        mask_cond_neg_ttd_u, 'ttd_u'
                    ].index
                ]
                raise ValueError(
                    'Upper terminal temperature difference in Condenser(s)'
                    + f' {cond_neg_ttd_u} is not positive. {user_help_prompt}'
                )

            mask_cond_neg_ttd_l = (
                self.nw.results['Condenser']['ttd_l'] <= 0
            )
            if any(mask_cond_neg_ttd_u):
                cond_neg_ttd_l = [
                    idx for idx
                    in self.nw.results['Condenser'].loc[
                        mask_cond_neg_ttd_l, 'ttd_l'
                    ].index
                ]
                raise ValueError(
                    'Lower terminal temperature difference in Condenser(s)'
                    + f' {cond_neg_ttd_l} is not positive. {user_help_prompt}'
                )

        if 'Compressor' in self.nw.results:
            mask_comp_neg_P = self.nw.results['Compressor']['P'] < 0
            if any(mask_comp_neg_P):
                comp_neg_P = [
                    idx for idx
                    in self.nw.results['Compressor'].loc[
                        mask_comp_neg_P, 'P'
                    ].index
                ]
                raise ValueError(
                    f'Power input in Compressor(s) {comp_neg_P} is negative. '
                    + user_help_prompt
                )

            mask_comp_neg_pr = (
                self.nw.results['Compressor']['pr'] <= 0
            )
            if any(mask_comp_neg_pr):
                comp_neg_pr = [
                    idx for idx
                    in self.nw.results['Compressor'].loc[
                        mask_comp_neg_pr, 'pr'
                    ].index
                ]
                raise ValueError(
                    f'Pressure ratio in Compressor(s) {comp_neg_pr} is '
                    + f'not positive. {user_help_prompt}'
                )

    def offdesign_simulation(self, log_simulations=False):
        """Perform offdesign parametrization and simulation."""
        if not self.solved_design:
            raise RuntimeError(
                'Heat pump has not been designed via the "design_simulation" '
                + 'method. Therefore the offdesign simulation will fail.'
            )

        # Parametrization
        kA_char1_default = ldc('heat exchanger', 'kA_char1', 'DEFAULT', CharLine)
        kA_char1_cond = ldc('heat exchanger', 'kA_char1', 'CONDENSING FLUID', CharLine)
        kA_char2_evap = ldc('heat exchanger', 'kA_char2', 'EVAPORATING FLUID', CharLine)
        kA_char2_default = ldc('heat exchanger', 'kA_char2', 'DEFAULT', CharLine)

        tespy_components = ['Condenser', 'HeatExchanger', 'Compressor', 'Pump', 'SimpleHeatExchanger']

        for comp in tespy_components:
            df = self.nw.comps
            labels = df[df['comp_type'] == comp].index.tolist()
            for label in labels:
                object = self.nw.get_comp(label)

                if comp == 'Compressor':
                    object.set_attr(design=['eta_s'], offdesign=['eta_s_char'])
                elif comp == 'Pump':
                    object.set_attr(design=['eta_s'], offdesign=['eta_s_char'])
                elif comp == 'HeatExchanger':
                    if 'Internal Heat Exchanger' in label:
                        object.set_attr(
                            kA_char1=kA_char1_default, kA_char2=kA_char2_default,
                            design=['pr1', 'pr2'], offdesign=['zeta1', 'zeta2']
                        )
                    elif 'Transcritical' in label:
                        object.set_attr(
                            kA_char1=kA_char1_default, kA_char2=kA_char2_default,
                            design=['pr2', 'ttd_l'], offdesign=['zeta2', 'kA_char']
                        )
                    elif 'Intermediate Heat Exchanger' in label:
                        object.set_attr(
                            kA_char1=kA_char1_cond, kA_char2=kA_char2_evap,
                            design=['pr1', 'ttd_u'], offdesign=['zeta1', 'kA_char']
                        )
                    else:
                        object.set_attr(
                            kA_char1=kA_char1_default, kA_char2=kA_char2_evap,
                            design=['pr1', 'ttd_l'], offdesign=['zeta1', 'kA_char']
                        )
                elif comp == 'Condenser':
                    object.set_attr(
                        kA_char1=kA_char1_cond, kA_char2=kA_char2_default,
                        design=['pr2', 'ttd_u'], offdesign=['zeta2', 'kA_char']
                    )
                elif comp == 'SimpleHeatExchanger':
                    object.set_attr(design=['pr'], offdesign=['zeta'])
                else:
                    raise ValueError(
                        f'Check whether offdesign parametrization is given to the component {comp}'
                        + f' in the heat pump base class.'
                    )

        self.conns['B1'].set_attr(offdesign=['v'])
        self.conns['B2'].set_attr(design=['T'])

        # Simulation
        print('Using improved offdesign simulation method.')
        self.create_ranges()

        deltaT_hs = (self.params['B1']['T'] - self.params['B2']['T'])

        multiindex = pd.MultiIndex.from_product(
            [self.T_hs_ff_range, self.T_cons_ff_range, self.pl_range],
            names=['T_hs_ff', 'T_cons_ff', 'pl']
        )

        results_offdesign = pd.DataFrame(
            index=multiindex, columns=['Q', 'P', 'COP', 'epsilon', 'residual']
        )

        for T_hs_ff in self.T_hs_ff_stablerange:
            self.conns['B1'].set_attr(T=T_hs_ff)
            if T_hs_ff <= 7:
                self.conns['B2'].set_attr(T=2)
            else:
                self.conns['B2'].set_attr(T=T_hs_ff - deltaT_hs)

            for T_cons_ff in self.T_cons_ff_stablerange:
                self.conns['C3'].set_attr(T=T_cons_ff)

                self.intermediate_states_offdesign(T_hs_ff, T_cons_ff, deltaT_hs)

                for pl in self.pl_stablerange[::-1]:
                    print(
                        f'### Temp. HS = {T_hs_ff} °C, Temp. Cons = '
                        + f'{T_cons_ff} °C, Partload = {pl * 100} % ###'
                    )
                    self.init_path = None
                    no_init_path = (
                        (T_cons_ff != self.T_cons_ff_range[0])
                        and (pl == self.pl_range[-1])
                    )
                    if no_init_path:
                        self.init_path = os.path.abspath(os.path.join(
                            os.path.dirname(__file__), 'stable',
                            f'{self.subdirname}_init.json'
                        ))

                    if 'cons' in self.comps:
                        self.comps['cons'].set_attr(Q=None)
                    self.conns['A0'].set_attr(m=pl * self.m_design)

                    try:
                        self.nw.solve('offdesign', design_path=self.design_path)
                        self.perform_exergy_analysis()
                        failed = False
                    except ValueError:
                        failed = True

                    # Logging simulation
                    if log_simulations:
                        logdirpath = os.path.abspath(os.path.join(
                            os.path.dirname(__file__), 'output', 'logging'
                        ))
                        if not os.path.exists(logdirpath):
                            os.mkdir(logdirpath)
                        logpath = os.path.join(
                            logdirpath, f'{self.subdirname}_offdesign_log.csv'
                        )
                        timestamp = datetime.fromtimestamp(time()).strftime('%H:%M:%S')
                        log_entry = (
                            f'{timestamp};{(self.nw.residual[-1] < 1e-3)};'
                            + f'{T_hs_ff:.2f};{T_cons_ff:.2f};{pl:.1f};'
                            + f'{self.nw.residual[-1]:.2e}\n'
                        )
                        if not os.path.exists(logpath):
                            with open(logpath, 'w', encoding='utf-8') as file:
                                file.write('Time;converged;Temp HS;Temp Cons;Partload;Residual\n')
                                file.write(log_entry)
                        else:
                            with open(logpath, 'a', encoding='utf-8') as file:
                                file.write(log_entry)

                    if pl == self.pl_range[-1] and self.nw.residual[-1] < 1e-3:
                        self.nw.save(os.path.abspath(os.path.join(
                            os.path.dirname(__file__), 'stable',
                            f'{self.subdirname}_init.json'
                        )))

                    inranges = (
                        (T_hs_ff in self.T_hs_ff_range)
                        & (T_cons_ff in self.T_cons_ff_range)
                        & (pl in self.pl_range)
                    )
                    idx = (T_hs_ff, T_cons_ff, pl)
                    if inranges:
                        empty_or_worse = (
                            pd.isnull(results_offdesign.loc[idx, 'Q'])
                            or (self.nw.residual[-1] < results_offdesign.loc[idx, 'residual'])
                        )
                        if empty_or_worse:
                            if failed:
                                results_offdesign.loc[idx, 'Q'] = np.nan
                                results_offdesign.loc[idx, 'P'] = np.nan
                                results_offdesign.loc[idx, 'epsilon'] = np.nan
                            else:
                                # thermal heat to sink (W → MW)
                                Q_out_W = self._get_heat_output_W()
                                results_offdesign.loc[idx, 'Q'] = abs(Q_out_W) / 1e6  # MW

                                # electrical input from grid (W → MW)
                                P_in_W = self.conns['E0'].E.val
                                results_offdesign.loc[idx, 'P'] = P_in_W / 1e6  # MW

                                # exergetic efficiency (dimensionless)
                                results_offdesign.loc[idx, 'epsilon'] = round(self.ean.epsilon, 3)

                            results_offdesign.loc[idx, 'COP'] = (
                                results_offdesign.loc[idx, 'Q'] / results_offdesign.loc[idx, 'P']
                            )
                            results_offdesign.loc[idx, 'residual'] = (self.nw.residual[-1])

        if self.params['offdesign']['save_results']:
            resultpath = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 'output',
                f'{self.subdirname}_partload.csv'
            ))
            results_offdesign.to_csv(resultpath, sep=';')

        self.df_to_array(results_offdesign)

    def intermediate_states_offdesign(self, T_hs_ff, T_cons_ff, deltaT_hs):
        """Calculates intermediate states during part-load simulation"""
        pass

    def get_compressor_results(self):
        """Return key results for each compressor used in the heat pump."""
        results = {}
        for c in self.comps.values():
            if 'Compressor' in c.label:
                comp = c.label
                results[comp] = {}

                results[comp]['V_dot'] = c.inl[0].vol.val_SI * 3600
                results[comp]['p_in'] = c.inl[0].p.val
                results[comp]['p_out'] = c.outl[0].p.val
                results[comp]['PI'] = c.outl[0].p.val / c.inl[0].p.val
                results[comp]['T_in'] = c.inl[0].T.val
                results[comp]['T_out'] = c.outl[0].T.val

        return results
