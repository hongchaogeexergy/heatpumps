import os

import numpy as np
from CoolProp.CoolProp import PropsSI as PSI

if __name__ == '__main__':
    from HeatPumpBase import HeatPumpBase
else:
    from .HeatPumpBase import HeatPumpBase

class HeatPumpCascadeBase(HeatPumpBase):
    """Super class of all concrete two stage heat pump models."""

    _OMMEN_P_MAX_BAR = {
        'R290': 28.0,
        'R1270': 28.0,
        'R600a': 28.0,
        'R600': 28.0,
        'R717': 50.0,
    }
    _OMMEN_P_TOL = 1.10
    _OMMEN_FLUID_ALIASES = {
        'NH3': 'R717',
        'R717': 'R717',
        'Propane': 'R290',
        'R290': 'R290',
        'Propylene': 'R1270',
        'R1270': 'R1270',
        'IsoButane': 'R600a',
        'R600a': 'R600a',
        'n-Butane': 'R600',
        'Butane': 'R600',
        'R600': 'R600',
    }

    def _init_fluids(self):
        """Initialize fluid attributes."""
        self.wf1 = self.params['fluids']['wf1']
        self.wf2 = self.params['fluids']['wf2']
        self.si = self.params['fluids']['si']
        self.so = self.params['fluids']['so']

    def _init_dir_paths(self):
        """Initialize paths and directories."""
        self.subdirname = (
            f"{self.params['setup']['type']}_"
            + f"{self.params['setup']['refrig1'].replace('::', '_')}_"
            + f"{self.params['setup']['refrig2'].replace('::', '_')}"
            )
        self.design_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'stable', f'{self.subdirname}_design.json'
            ))
        self.validate_dir()

    def get_design_t_mid(self, t_source_cold, t_sink_hot):
        """
        Return the design intermediate temperature for cascade models.

        ``setup.cascade_split_mode="t_mid"`` keeps the historical behaviour
        with ``setup.t_mid`` or ``setup.t_mid_fraction``.
        ``setup.cascade_split_mode="lift_share"`` maps the literature lift
        split onto the upper-cycle evaporation temperature and then to T_mid.
        """
        setup = self.params.setdefault('setup', {})
        t_source_cold = float(t_source_cold)
        t_sink_hot = float(t_sink_hot)
        source_hot = self.get_source_inlet_temperature()
        inter_ttd = float(self.params.get('inter', {}).get('ttd_u', 0.0))

        if t_sink_hot <= t_source_cold:
            raise ValueError(
                'Intermediate temperature requires sink hot temperature to be '
                + 'above source cold temperature.'
            )

        split_mode = str(
            setup.get(
                'cascade_split_mode',
                'lift_share' if 'lift_share' in setup else 't_mid'
            )
        )
        if split_mode not in {'t_mid', 'lift_share'}:
            raise ValueError(
                f"Unsupported cascade_split_mode '{split_mode}'. "
                + "Supported values are 't_mid' and 'lift_share'."
            )

        if split_mode == 'lift_share':
            lift_share = float(setup.get('lift_share', 0.5))
            t34 = source_hot + lift_share * (t_sink_hot - source_hot)
            t_mid = t34 + inter_ttd / 2.0
            fraction = (
                (t_mid - t_source_cold) / (t_sink_hot - t_source_cold)
            )
        elif 't_mid_fraction' in setup:
            fraction = float(setup['t_mid_fraction'])
            t_mid = t_source_cold + fraction * (t_sink_hot - t_source_cold)
            lift_share = (
                (t_mid - inter_ttd / 2.0 - source_hot)
                / max(t_sink_hot - source_hot, 1e-9)
            )
            t34 = t_mid - inter_ttd / 2.0
        elif 't_mid' in setup:
            t_mid = float(setup['t_mid'])
            fraction = (
                (t_mid - t_source_cold) / (t_sink_hot - t_source_cold)
            )
            lift_share = (
                (t_mid - inter_ttd / 2.0 - source_hot)
                / max(t_sink_hot - source_hot, 1e-9)
            )
            t34 = t_mid - inter_ttd / 2.0
        else:
            fraction = 0.5
            t_mid = t_source_cold + fraction * (t_sink_hot - t_source_cold)
            lift_share = (
                (t_mid - inter_ttd / 2.0 - source_hot)
                / max(t_sink_hot - source_hot, 1e-9)
            )
            t34 = t_mid - inter_ttd / 2.0

        if not np.isfinite(t_mid):
            raise ValueError('Intermediate temperature T_mid must be finite.')
        if not (t_source_cold < t_mid < t_sink_hot):
            raise ValueError(
                f'Intermediate temperature T_mid={t_mid:.2f} °C must lie '
                + f'between source cold temperature {t_source_cold:.2f} °C '
                + f'and sink hot temperature {t_sink_hot:.2f} °C.'
            )

        setup['t_mid'] = t_mid
        setup['t_mid_fraction'] = fraction
        setup['cascade_split_mode'] = split_mode
        setup['lift_share'] = lift_share
        setup['T34'] = t34
        self.T_mid = t_mid
        self.t_mid_fraction = fraction
        self.lift_share = lift_share
        return t_mid

    def get_cycle_split_metrics(self):
        """Return LT/HT cycle split KPIs for cascade heat pumps."""
        metrics = {
            'T_mid_C': getattr(self, 'T_mid', np.nan),
            't_mid_fraction': getattr(self, 't_mid_fraction', np.nan),
            'lift_share': getattr(self, 'lift_share', np.nan),
            'm_ht_kg_s': np.nan,
            'm_lt_kg_s': np.nan,
            'm_ht_to_lt': np.nan,
            'Q_ht_W': np.nan,
            'Q_lt_W': np.nan,
            'P_ht_W': 0.0,
            'P_lt_W': 0.0,
            'P_total_W': 0.0,
            'power_share_ht': np.nan,
            'power_share_lt': np.nan,
            'cop_ht': np.nan,
            'cop_lt': np.nan,
        }

        for key, field in (('A0', 'm_ht_kg_s'), ('D0', 'm_lt_kg_s')):
            conn = self.conns.get(key)
            if conn is None:
                continue
            try:
                value = float(conn.m.val)
                if np.isfinite(value):
                    metrics[field] = value
            except Exception:
                continue

        if (
            np.isfinite(metrics['m_ht_kg_s'])
            and np.isfinite(metrics['m_lt_kg_s'])
            and metrics['m_lt_kg_s'] != 0
        ):
            metrics['m_ht_to_lt'] = (
                metrics['m_ht_kg_s'] / metrics['m_lt_kg_s']
            )

        for comp in self.comps.values():
            label = getattr(comp, 'label', '').lower()
            if 'temperature compressor' not in label or 'motor' in label:
                continue
            try:
                power = abs(float(comp.P.val))
            except Exception:
                power = np.nan
            if not np.isfinite(power):
                continue
            if 'high' in label:
                metrics['P_ht_W'] += power
            elif 'low' in label:
                metrics['P_lt_W'] += power

        lt_product = self.comps.get('inter')
        if lt_product is not None and hasattr(lt_product, 'Q'):
            try:
                q_lt = abs(float(lt_product.Q.val))
                if np.isfinite(q_lt):
                    metrics['Q_lt_W'] = q_lt
            except Exception:
                pass

        ht_product = self.comps.get('cond', self.comps.get('trans'))
        if ht_product is not None and hasattr(ht_product, 'Q'):
            try:
                q_ht = abs(float(ht_product.Q.val))
                if np.isfinite(q_ht):
                    metrics['Q_ht_W'] = q_ht
            except Exception:
                pass

        metrics['P_total_W'] = metrics['P_ht_W'] + metrics['P_lt_W']
        if metrics['P_total_W'] > 0:
            metrics['power_share_ht'] = (
                metrics['P_ht_W'] / metrics['P_total_W']
            )
            metrics['power_share_lt'] = (
                metrics['P_lt_W'] / metrics['P_total_W']
            )

        if metrics['P_lt_W'] > 0 and np.isfinite(metrics['Q_lt_W']):
            metrics['cop_lt'] = metrics['Q_lt_W'] / metrics['P_lt_W']
        if metrics['P_ht_W'] > 0 and np.isfinite(metrics['Q_ht_W']):
            metrics['cop_ht'] = metrics['Q_ht_W'] / metrics['P_ht_W']

        return metrics

    def get_reference_case_metrics(self):
        """Return literature-comparison KPIs for cascade heat pumps."""
        metrics = {
            'COP': float(getattr(self, 'cop', np.nan)),
            'eta_Lorenz': float(getattr(self, 'eta_lorenz', np.nan)),
            'epsilon': float(getattr(self, 'epsilon', np.nan)),
            'Q_H_kW': np.nan,
            'W_el_kW': np.nan,
            'T_source_out_C': np.nan,
            'm_source_kg_s': np.nan,
            'm_sink_kg_s': np.nan,
            'm_cycle1_kg_s': np.nan,
            'm_cycle2_kg_s': np.nan,
            'V_dot_c1_m3_h': np.nan,
            'V_dot_c2_m3_h': np.nan,
            'p_low_c1_bar': np.nan,
            'p_high_c1_bar': np.nan,
            'T_disch_c1_C': np.nan,
            'p_low_c2_bar': np.nan,
            'p_high_c2_bar': np.nan,
            'T_disch_c2_C': np.nan,
            'lift_share': np.nan,
            'T34_C': np.nan,
            'T_mid_C': np.nan,
        }

        try:
            q_out_w = float(self._get_heat_output_W())
            if np.isfinite(q_out_w):
                metrics['Q_H_kW'] = abs(q_out_w) / 1e3
        except Exception:
            pass

        try:
            w_el_w = float(self.conns['E0'].E.val)
            if np.isfinite(w_el_w):
                metrics['W_el_kW'] = abs(w_el_w) / 1e3
        except Exception:
            pass

        source_out_label = self.get_source_outlet_connection_label()
        if source_out_label in self.conns:
            try:
                t_source_out = float(self.conns[source_out_label].T.val)
                if np.isfinite(t_source_out):
                    metrics['T_source_out_C'] = t_source_out
            except Exception:
                pass
            try:
                m_source = float(self.conns[source_out_label].m.val)
                if np.isfinite(m_source):
                    metrics['m_source_kg_s'] = m_source
            except Exception:
                pass

        if 'C1' in self.conns:
            try:
                m_sink = float(self.conns['C1'].m.val)
                if np.isfinite(m_sink):
                    metrics['m_sink_kg_s'] = m_sink
            except Exception:
                pass

        split_metrics = self.get_cycle_split_metrics()
        for src_key, dst_key in (
            ('m_lt_kg_s', 'm_cycle1_kg_s'),
            ('m_ht_kg_s', 'm_cycle2_kg_s'),
            ('lift_share', 'lift_share'),
            ('T_mid_C', 'T_mid_C'),
        ):
            value = split_metrics.get(src_key, np.nan)
            try:
                value = float(value)
            except Exception:
                value = np.nan
            if np.isfinite(value):
                metrics[dst_key] = value

        try:
            t34 = float(self.params.get('setup', {}).get('T34', np.nan))
            if np.isfinite(t34):
                metrics['T34_C'] = t34
        except Exception:
            pass

        comp_results = self.get_compressor_results()
        for label, values in comp_results.items():
            label_lower = str(label).lower()
            if 'low' in label_lower:
                suffix = 'c1'
            elif 'high' in label_lower:
                suffix = 'c2'
            else:
                continue

            for src_key, dst_key in (
                ('V_dot', f'V_dot_{suffix}_m3_h'),
                ('p_in', f'p_low_{suffix}_bar'),
                ('p_out', f'p_high_{suffix}_bar'),
                ('T_out', f'T_disch_{suffix}_C'),
            ):
                try:
                    value = float(values.get(src_key, np.nan))
                except Exception:
                    value = np.nan
                if np.isfinite(value):
                    metrics[dst_key] = value

        return metrics

    def _resolve_ommen_fluid(self, fluid):
        """Map local fluid names onto Ommen-table refrigerant labels."""
        return self._OMMEN_FLUID_ALIASES.get(fluid, fluid)

    def _check_ommen_cycle_pressure(self, fluid, p_high, cycle_label):
        """Raise if the cycle pressure exceeds the Ommen limit."""
        if bool(self.params.get('setup', {}).get('skip_ommen_check', False)):
            return
        fluid_key = self._resolve_ommen_fluid(fluid)
        p_limit = self._OMMEN_P_MAX_BAR.get(fluid_key)
        if p_limit is None:
            return
        p_limit_eff = p_limit * self._OMMEN_P_TOL
        if p_high > p_limit_eff:
            raise ValueError(
                f'High-side pressure of {cycle_label} with {fluid_key} reaches '
                + f'{p_high:.2f} bar and exceeds the Ommen limit '
                + f'({p_limit_eff:.2f} bar incl. tolerance).'
            )

    def check_cascade_operating_limits(self):
        """Check cycle pressure limits against the Ommen compressor envelope."""
        if all(label in self.conns for label in ('D4', 'D0', 'A4', 'A0')):
            p_high_lt = max(
                float(self.conns['D4'].p.val), float(self.conns['D0'].p.val)
            )
            p_high_ht = max(
                float(self.conns['A4'].p.val), float(self.conns['A0'].p.val)
            )
        elif all(label in self.conns for label in ('D6', 'D0', 'A6', 'A0')):
            p_high_lt = max(
                float(self.conns['D6'].p.val), float(self.conns['D0'].p.val)
            )
            p_high_ht = max(
                float(self.conns['A6'].p.val), float(self.conns['A0'].p.val)
            )
        else:
            return

        self._check_ommen_cycle_pressure(self.wf1, p_high_lt, 'cycle 1')
        self._check_ommen_cycle_pressure(self.wf2, p_high_ht, 'cycle 2')

    def generate_state_diagram(self, refrig='', diagram_type='logph',
                               style='light', figsize=(16, 10), fontsize=10,
                               legend=True, legend_loc='upper left',
                               return_diagram=False, savefig=True,
                               open_file=True, **kwargs):
        """
        Generate log(p)-h-diagram of heat pump process.

        Parameters
        ----------

        refrig : str
            Name of refrigerant to use for plot. Can be left as an empty string
            in single cycle heat pumps.

        diagram_type : str
            Fluid property diagram type. Either 'logph' or 'Ts'. Default is
            'logph'.

        style : str
            Diagram style to chose. Either 'light' or 'dark'. Default is
            'light'.

        figsize : tuple/list of numbers
            Size of matplotlib figure in inches. Default is (16, 10), so the
            figure is 16 inches wide and 10 inches tall.

        fontsize : int/float
            Size of main fonts in points. Title is 20% larger and tick labels
            as well as state annotations are 10% smaller. Default is 10pts.

        legend : bool
            Flag to set if legend should be shown. Default is `True`.

        legend_loc : str
            Location to place legend to. Accepts options as matplotlib allows.
            Default is 'upper left'. Is only used if 'legend' parameter is set
            to `True`.

        return_diagram : bool
            Flag to set if diagram object should be returned by method. Default
            is False.

        savefig : bool
            Flag to set if diagram should be saved to disk. Default is `False`.

        filepath : str
            Path to save the file to. If `None` and `savefig` is `True`, a
            default name is given and saved to the current working directory.
            Default is `None`.

        open_file : bool
            Flag to set if saved file should be opend by the os. Default is
            `False`.

        **kwargs
            Additional keyword arguments to pass through to the
            `get_plotting_states` method of the heat pump class.
        """
        kwargs1 = {}
        kwargs2 = {}
        if 'xlims' in kwargs:
            kwargs1['xlims'] = kwargs['xlims'][0]
            kwargs2['xlims'] = kwargs['xlims'][1]
        if 'ylims' in kwargs:
            kwargs1['ylims'] = kwargs['ylims'][0]
            kwargs2['ylims'] = kwargs['ylims'][1]
        if return_diagram:
            diagram1 = super().generate_state_diagram(
                refrig=self.params['setup']['refrig1'],
                style=style, figsize=figsize, fontsize=fontsize,
                diagram_type=diagram_type, legend=legend,
                legend_loc=legend_loc,
                return_diagram=return_diagram, savefig=savefig,
                open_file=open_file, cycle=1, **kwargs1
            )
            diagram2 = super().generate_state_diagram(
                refrig=self.params['setup']['refrig2'],
                style=style, figsize=figsize, fontsize=fontsize,
                diagram_type=diagram_type, legend=legend,
                legend_loc=legend_loc,
                return_diagram=return_diagram, savefig=savefig,
                open_file=open_file, cycle=2, **kwargs2
            )
            return diagram1, diagram2
        else:
            super().generate_state_diagram(
                refrig=self.params['setup']['refrig1'],
                style=style, figsize=figsize, fontsize=fontsize,
                diagram_type=diagram_type, legend=legend,
                legend_loc=legend_loc,
                return_diagram=return_diagram, savefig=savefig,
                open_file=open_file, cycle=1, **kwargs1
            )
            super().generate_state_diagram(
                refrig=self.params['setup']['refrig2'],
                style=style, figsize=figsize, fontsize=fontsize,
                diagram_type=diagram_type, legend=legend,
                legend_loc=legend_loc,
                return_diagram=return_diagram, savefig=savefig,
                open_file=open_file, cycle=2, **kwargs2
            )

    def check_mid_temperature(self, wf):
        """Check if the intermediate pressure is below the critical pressure."""
        T_crit = PSI('T_critical', wf) - 273.15
        if self.T_mid > T_crit:
            raise ValueError(
                f'Intermediate temperature of {self.T_mid:1f} °C must be below '
                + f'the critical temperature of {wf} of {T_crit:.1f} °C.'
            )
