from tespy.components import MovingBoundaryHeatExchanger
from tespy.connections import Ref

if __name__ == '__main__':
    from HeatPumpCascade import HeatPumpCascade
else:
    from .HeatPumpCascade import HeatPumpCascade


class HeatPumpCascadeMBHX(HeatPumpCascade):
    """
    Experimental cascade heat pump with a moving-boundary interstage HX.

    The model intentionally reuses the current cascade boundary-mode and solve
    logic. Only the intermediate heat exchanger component is swapped from a
    TESPy ``Condenser`` to ``MovingBoundaryHeatExchanger`` so the reference
    case can be tested without touching the existing cascade models.
    """

    def generate_components(self):
        """Initialize components with a moving-boundary interstage HX."""
        super().generate_components()
        self.comps['inter'] = MovingBoundaryHeatExchanger(
            'Intermediate Heat Exchanger'
        )

    def init_simulation(self, **kwargs):
        """
        Perform initial parametrization with one extra MBHX closure.

        The parent cascade relies on the ``Condenser`` hot-side outlet state
        equation during the first initialization solve. Replacing the
        intermediate condenser with a moving-boundary heat exchanger removes
        that built-in closure and the network becomes underdetermined by one
        specification. For the warm-start solve we therefore temporarily impose
        saturated liquid at the LT hot-side outlet of the interstage HX. The
        specification is released afterwards so the MBHX-specific design solve
        can apply its own pinch-based closure.
        """
        source_mode = self.get_source_mode()
        sink_mode = self.get_sink_mode()

        # Components
        self.conns['A4'].set_attr(
            h=Ref(self.conns['A3'], self._init_vals['dh_rel_comp'], 0)
        )
        self.conns['D4'].set_attr(
            h=Ref(self.conns['D3'], self._init_vals['dh_rel_comp'], 0)
        )
        if 'hs_pump' in self.comps:
            self.comps['hs_pump'].set_attr(eta_s=self.params['hs_pump']['eta_s'])
        hsink_params = self.params.get(
            'hsink_pump', self.params.get('cons_pump', {})
        )
        if 'hsink_pump' in self.comps and 'eta_s' in hsink_params:
            self.comps['hsink_pump'].set_attr(eta_s=hsink_params['eta_s'])

        self.comps['evap'].set_attr(
            pr1=self.params['evap']['pr1'], pr2=self.params['evap']['pr2']
        )
        self.comps['inter'].set_attr(
            pr1=self.params['inter']['pr1'], pr2=self.params['inter']['pr2']
        )
        self.comps['cond'].set_attr(
            pr1=self.params['cond']['pr1'], pr2=self.params['cond']['pr2']
        )

        # Connections
        t_cond = self.get_sink_hot_design_temperature()
        t_source_cold = self.get_source_cold_design_temperature()
        self.T_mid = self.get_design_t_mid(t_source_cold, t_cond)

        # Starting values
        p_evap1, p_cond1, p_evap2, p_cond2 = self.get_pressure_levels(
            T_evap=t_source_cold, T_mid=self.T_mid, T_cond=t_cond
        )
        self.p_evap2 = p_evap2
        self.p_evap1 = p_evap1

        # Main cycle
        self.conns['A3'].set_attr(x=self.params['A3']['x'], p=p_evap2)
        self.conns['A0'].set_attr(p=p_cond2, fluid={self.wf2: 1})
        self.conns['D3'].set_attr(x=self.params['D3']['x'], p=p_evap1)
        self.conns['D0'].set_attr(p=p_cond1, fluid={self.wf1: 1}, x=0)

        # Heat source
        m_source = self.params['B1'].get('m', None)
        if source_mode == 'fixed_mass_flow':
            m_source = self.get_source_mass_flow()
            if m_source is None:
                raise ValueError(
                    "source_mode='fixed_mass_flow' requires setup.m_source "
                    + "or B1.m."
                )
        self.conns['B1'].set_attr(
            T=self.params['B1']['T'], p=self.params['B1']['p'],
            m=m_source, fluid={self.so: 1}
        )
        if source_mode == 'fixed_delta_T':
            self.conns['B2'].set_attr(T=t_source_cold)
        if 'B3' in self.conns:
            self.conns['B3'].set_attr(p=self.params['B1']['p'])

        # Heat sink
        if sink_mode == 'steam':
            if self.get_use_sink_pump():
                raise ValueError(
                    "sink_mode='steam' is only supported without a sink pump."
                )
            m_steam = self.get_steam_mass_flow()
            if m_steam is None:
                raise ValueError(
                    "sink_mode='steam' requires setup.m_steam or C1.m."
                )
            c1_p = self.get_sink_pressure_bar()
            self.conns['C1'].set_attr(
                p=c1_p, x=0, m=m_steam, fluid={self.si: 1}
            )
            self.conns['C2'].set_attr(x=1)
        else:
            c1_p = self.get_sink_pressure_bar()
            self.conns['C1'].set_attr(
                T=self.params['C1']['T'], p=c1_p, fluid={self.si: 1}
            )
            self.conns['C2'].set_attr(T=t_cond)
            c3_p = self.params.get('C3', {}).get('p', c1_p)
            if 'C3' in self.conns and c3_p is not None:
                self.conns['C3'].set_attr(p=c3_p)
            c1_m = self.params['C1'].get('m', None)
            if c1_m is None:
                cons = self.params.get('cons', {})
                q_cons = cons.get('Q')
                t_c1 = self.params['C1']['T']
                t_c2 = t_cond
                if q_cons is not None and t_c2 != t_c1:
                    cp_w = 4180.0
                    c1_m = abs(float(q_cons)) / (cp_w * abs(t_c2 - t_c1))
            if c1_m is not None:
                self.conns['C1'].set_attr(m=c1_m)

        # Perform initial simulation and unset temporary starting values.
        self._solve_model(**kwargs)

        self.conns['A0'].set_attr(p=None)
        self.conns['A3'].set_attr(p=None)
        self.conns['D0'].set_attr(p=None, x=None)
        self.conns['D3'].set_attr(p=None)
        self.conns['A4'].set_attr(h=None)
        self.conns['D4'].set_attr(h=None)

    def design_simulation(self, **kwargs):
        """
        Perform an MBHX-specific two-stage design solve.

        Stage 1 reuses the legacy cascade temperature guidance as a warm start.
        Stage 2 releases the hard interstage outlet temperature and lets the
        moving-boundary heat exchanger close the interstage via ``td_pinch``.
        If the pinch-driven stage does not converge, the model falls back to
        the warm-start solution so the experiment branch remains usable.
        """
        inter_ttd = float(self.params['inter']['ttd_u'])
        a3_t_start = self.T_mid - inter_ttd / 2.0

        self.comps['LT_comp'].set_attr(eta_s=self.params['LT_comp']['eta_s'])
        self.comps['HT_comp'].set_attr(eta_s=self.params['HT_comp']['eta_s'])
        self.comps['evap'].set_attr(ttd_l=self.params['evap']['ttd_l'])
        self.comps['cond'].set_attr(ttd_u=self.params['cond']['ttd_u'])

        # Warm start: keep the legacy interstage target to obtain a stable
        # initial design point before relaxing the coupling.
        self.comps['inter'].set_attr(ttd_u=inter_ttd, td_pinch=None)
        self.conns['A3'].set_attr(T=a3_t_start)
        self.conns['D0'].set_attr(x=0)
        self._solve_model(**kwargs)

        if not self.solved_design:
            return

        warm_start_mass_flow = self.conns['A0'].m.val
        warm_start_d0_p = float(self.conns['D0'].p.val)

        # Pinch-driven refinement: keep the initial pressures only as the
        # solved state from stage 1 and let the MBHX determine the HT side of
        # the interstage coupling via its own pinch equation.
        self.comps['inter'].set_attr(ttd_u=None, td_pinch=inter_ttd)
        self.conns['A3'].set_attr(T=None)
        self.conns['D0'].set_attr(p=warm_start_d0_p, x=0)
        for conn_key in ('A0', 'A3', 'D3'):
            self.conns[conn_key].set_attr(p=None)

        self._solve_model(**kwargs)

        if not self.solved_design:
            # Fallback to the warm-start solution if the relaxed pinch-driven
            # stage does not converge for a given parameter set.
            self.comps['inter'].set_attr(td_pinch=None, ttd_u=inter_ttd)
            self.conns['A3'].set_attr(T=a3_t_start)
            self.conns['D0'].set_attr(x=0)
            self._solve_model(**kwargs)
            if self.solved_design:
                self.m_design = warm_start_mass_flow
            return

        self.m_design = self.conns['A0'].m.val
