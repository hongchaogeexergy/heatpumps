# UPDATED: heat sink loop refactor applied
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI as PSI
from tespy.components import (Compressor, Condenser, CycleCloser,
                              HeatExchanger, Pump, Sink,
                              Source, Valve, PowerSource, PowerBus, Motor)
from tespy.connections import Connection, Ref, PowerConnection
from tespy.networks import Network
from tespy.tools.characteristics import CharLine
from tespy.tools.characteristics import load_default_char as ldc

if __name__ == '__main__':
    from HeatPumpCascadeBase import HeatPumpCascadeBase
else:
    from .HeatPumpCascadeBase import HeatPumpCascadeBase


class HeatPumpCascadeICTrans(HeatPumpCascadeBase):
    """Two stage transcritical cascading heat pump with two refrigerants and intercooler."""

    def generate_components(self):
        """Initialize components of heat pump."""
        # Heat source
        self.comps['hs_ff'] = Source('Heat Source Feed Flow')
        self.comps['hs_bf'] = Sink('Heat Source Back Flow')
        self.comps['hs_pump'] = Pump('Heat Source Recirculation Pump')

        # Heat sink
        self.comps['hsink_ff'] = Source('Heat Sink Feed Flow')
        self.comps['hsink_bf'] = Sink('Heat Sink Back Flow')
        self.comps['hsink_pump'] = Pump('Heat Sink Recirculation Pump')

        # Upper cycle
        self.comps['trans'] = HeatExchanger('Transcritical Heat Exchanger')
        self.comps['cc2'] = CycleCloser('Main Cycle Closer 2')
        self.comps['valve2'] = Valve('Valve 2')
        self.comps['inter'] = Condenser('Intermediate Heat Exchanger')
        self.comps['HT_comp1'] = Compressor('High Temperature Compressor 1')
        self.comps['ic2'] = HeatExchanger('Intercooler 2')
        self.comps['HT_comp2'] = Compressor('High Temperature Compressor 2')
        self.comps['ic2_cool_ff'] = Source('Intercooler 2 Cooling In')
        self.comps['ic2_cool_bf'] = Sink('Intercooler 2 Cooling Out')

        # Lower cycle
        self.comps['cc1'] = CycleCloser('Main Cycle Closer 1')
        self.comps['valve1'] = Valve('Valve 1')
        self.comps['evap'] = HeatExchanger('Evaporator')
        self.comps['LT_comp1'] = Compressor('Low Temperature Compressor 1')
        self.comps['ic1'] = HeatExchanger('Intercooler 1')
        self.comps['LT_comp2'] = Compressor('Low Temperature Compressor 2')
        self.comps['ic1_cool_ff'] = Source('Intercooler 1 Cooling In')
        self.comps['ic1_cool_bf'] = Sink('Intercooler 1 Cooling Out')

    def generate_connections(self):
        """Initialize and add connections and buses to network."""
        # Upper Cycle Connections
        self.conns['A0'] = Connection(
            self.comps['trans'], 'out1', self.comps['cc2'], 'in1', 'A0'
        )
        self.conns['A1'] = Connection(
            self.comps['cc2'], 'out1', self.comps['valve2'], 'in1', 'A1'
        )
        self.conns['A2'] = Connection(
            self.comps['valve2'], 'out1', self.comps['inter'], 'in2', 'A2'
        )
        self.conns['A3'] = Connection(
            self.comps['inter'], 'out2', self.comps['HT_comp1'], 'in1', 'A3'
        )
        self.conns['A4'] = Connection(
            self.comps['HT_comp1'], 'out1', self.comps['ic2'], 'in1', 'A4'
        )
        self.conns['A5'] = Connection(
            self.comps['ic2'], 'out1', self.comps['HT_comp2'], 'in1', 'A5'
        )
        self.conns['A6'] = Connection(
            self.comps['HT_comp2'], 'out1', self.comps['trans'], 'in1', 'A6'
        )
        self.conns['F1'] = Connection(
            self.comps['ic2_cool_ff'], 'out1', self.comps['ic2'], 'in2', 'F1'
        )
        self.conns['F2'] = Connection(
            self.comps['ic2'], 'out2', self.comps['ic2_cool_bf'], 'in1', 'F2'
        )

        # Lower cycle connections
        self.conns['D0'] = Connection(
            self.comps['inter'], 'out1', self.comps['cc1'], 'in1', 'D0'
        )
        self.conns['D1'] = Connection(
            self.comps['cc1'], 'out1', self.comps['valve1'], 'in1', 'D1'
        )
        self.conns['D2'] = Connection(
            self.comps['valve1'], 'out1', self.comps['evap'], 'in2', 'D2'
        )
        self.conns['D3'] = Connection(
            self.comps['evap'], 'out2', self.comps['LT_comp1'], 'in1', 'D3'
        )
        self.conns['D4'] = Connection(
            self.comps['LT_comp1'], 'out1', self.comps['ic1'], 'in1', 'D4'
        )
        self.conns['D5'] = Connection(
            self.comps['ic1'], 'out1', self.comps['LT_comp2'], 'in1', 'D5'
        )
        self.conns['D6'] = Connection(
            self.comps['LT_comp2'], 'out1', self.comps['inter'], 'in1', 'D6'
        )
        self.conns['G1'] = Connection(
            self.comps['ic1_cool_ff'], 'out1', self.comps['ic1'], 'in2', 'G1'
        )
        self.conns['G2'] = Connection(
            self.comps['ic1'], 'out2', self.comps['ic1_cool_bf'], 'in1', 'G2'
        )

        self.conns['B1'] = Connection(
            self.comps['hs_ff'], 'out1', self.comps['evap'], 'in1', 'B1'
        )
        self.conns['B2'] = Connection(
            self.comps['evap'], 'out1', self.comps['hs_pump'], 'in1', 'B2'
        )
        self.conns['B3'] = Connection(
            self.comps['hs_pump'], 'out1', self.comps['hs_bf'], 'in1', 'B3'
        )

        self.conns['C1'] = Connection(
            self.comps['hsink_ff'], 'out1', self.comps['trans'], 'in2', 'C1'
            )
        self.conns['C2'] = Connection(
            self.comps['trans'], 'out2', self.comps['hsink_pump'], 'in1', 'C2'
            )
        self.conns['C3'] = Connection(
            self.comps['hsink_pump'], 'out1', self.comps['hsink_bf'], 'in1', 'C3'
            )


        # Buses
        mot_x = np.array([
            0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
            0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15,
            1.2, 10
        ])
        mot_y = (np.array([
            0.01, 0.3148, 0.5346, 0.6843, 0.7835, 0.8477, 0.8885, 0.9145,
            0.9318, 0.9443, 0.9546, 0.9638, 0.9724, 0.9806, 0.9878, 0.9938,
            0.9982, 1.0009, 1.002, 1.0015, 1, 0.9977, 0.9947, 0.9909, 0.9853,
            0.9644
        ]) * 0.98)
        mot = CharLine(x=mot_x, y=mot_y)
        # Create power components
        self.comps['power_source'] = PowerSource('Power Input')
        self.comps['power_bus'] = PowerBus('Power Distribution', num_in=1, num_out=6)

        self.comps['motor_HT1'] = Motor('HT_comp1 motor')
        self.comps['motor_HT2'] = Motor('HT_comp2 motor')
        self.comps['motor_LT1'] = Motor('LT_comp1 motor')
        self.comps['motor_LT2'] = Motor('LT_comp2 motor')
        self.comps['motor_hs'] = Motor('HS_pump motor')
        self.comps['motor_cons'] = Motor('Cons_pump motor')

        # Power connections
        self.conns['E0'] = PowerConnection(self.comps['power_source'], 'power',
                                        self.comps['power_bus'], 'power_in1', label='E0')

        self.conns['E1'] = PowerConnection(self.comps['power_bus'], 'power_out1',
                                        self.comps['motor_HT1'], 'power_in', label='E1')
        self.conns['e1'] = PowerConnection(self.comps['motor_HT1'], 'power_out',
                                        self.comps['HT_comp1'], 'power', label='e1')

        self.conns['E2'] = PowerConnection(self.comps['power_bus'], 'power_out2',
                                        self.comps['motor_HT2'], 'power_in', label='E2')
        self.conns['e2'] = PowerConnection(self.comps['motor_HT2'], 'power_out',
                                        self.comps['HT_comp2'], 'power', label='e2')

        self.conns['E3'] = PowerConnection(self.comps['power_bus'], 'power_out3',
                                        self.comps['motor_LT1'], 'power_in', label='E3')
        self.conns['e3'] = PowerConnection(self.comps['motor_LT1'], 'power_out',
                                        self.comps['LT_comp1'], 'power', label='e3')

        self.conns['E4'] = PowerConnection(self.comps['power_bus'], 'power_out4',
                                        self.comps['motor_LT2'], 'power_in', label='E4')
        self.conns['e4'] = PowerConnection(self.comps['motor_LT2'], 'power_out',
                                        self.comps['LT_comp2'], 'power', label='e4')

        self.conns['E5'] = PowerConnection(self.comps['power_bus'], 'power_out5',
                                        self.comps['motor_hs'], 'power_in', label='E5')
        self.conns['e5'] = PowerConnection(self.comps['motor_hs'], 'power_out',
                                        self.comps['hs_pump'], 'power', label='e5')

        self.conns['E6'] = PowerConnection(self.comps['power_bus'], 'power_out6',
                                        self.comps['motor_cons'], 'power_in', label='E6')
        self.conns['e6'] = PowerConnection(self.comps['motor_cons'], 'power_out',
                                        self.comps['hsink_pump'], 'power', label='e6')

        # Add all connections to network
        self.nw.add_conns(*[conn for conn in self.conns.values()])

        # Set motor efficiency attributes (wie im Beispiel)
        for motor_label in ['motor_HT1', 'motor_HT2', 'motor_LT1', 'motor_LT2', 'motor_hs', 'motor_cons']:
            self.comps[motor_label].set_attr(eta=0.98, eta_char=mot)


    def init_simulation(self, **kwargs):
        """Perform initial parametrization with starting values."""
        # Components
        self.conns['A4'].set_attr(
            h=Ref(self.conns['A3'], self._init_vals['dh_rel_comp'], 0)
            )
        self.conns['A6'].set_attr(
            h=Ref(self.conns['A5'], self._init_vals['dh_rel_comp'], 0)
            )
        self.conns['D4'].set_attr(
            h=Ref(self.conns['D3'], self._init_vals['dh_rel_comp'], 0)
            )
        self.conns['D6'].set_attr(
            h=Ref(self.conns['D5'], self._init_vals['dh_rel_comp'], 0)
            )
        self.comps['hs_pump'].set_attr(eta_s=self.params['hs_pump']['eta_s'])
        hsink_params = self.params.get('hsink_pump', self.params.get('cons_pump', {}))
        if 'eta_s' in hsink_params:
            self.comps['hsink_pump'].set_attr(eta_s=hsink_params['eta_s'])

        self.comps['evap'].set_attr(
            pr1=self.params['evap']['pr1'], pr2=self.params['evap']['pr2']
        )
        self.comps['inter'].set_attr(
            pr1=self.params['inter']['pr1'], pr2=self.params['inter']['pr2']
        )
        self.comps['trans'].set_attr(
            pr1=self.params['trans']['pr1'], pr2=self.params['trans']['pr2']
        )
        ic1_params = self.params.get('ic1', {})
        ic2_params = self.params.get('ic2', {})
        self.comps['ic1'].set_attr(
            pr1=ic1_params.get('pr1', ic1_params.get('pr', 0.98)),
            pr2=ic1_params.get('pr2', 0.98)
        )
        self.comps['ic2'].set_attr(
            pr1=ic2_params.get('pr1', ic2_params.get('pr', 0.98)),
            pr2=ic2_params.get('pr2', 0.98)
        )

        # Connections
        # Starting values
        t_sink_hot = self.params.get('C2', {}).get('T', self.params.get('C3', {}).get('T', self.params['C1']['T']))
        self.T_mid = (self.params['B2']['T'] + t_sink_hot) / 2
        p_evap1, p_cond1, p_mid1, p_evap2, h_trans_out, p_mid2 = self.get_pressure_levels(
            T_evap=self.params['B2']['T'], T_mid=self.T_mid
        )
        self.p_evap1 = p_evap1
        self.p_evap2 = p_evap2
        self.p_mid1 = p_mid1
        self.p_mid2 = p_mid2

        h_s_mid1 = PSI(
            'H', 'P', p_mid1 * 1e5,
            'S', PSI('S', 'Q', 1, 'P', p_evap1 * 1e5, self.wf1),
            self.wf1
        ) * 1e-3
        h_s_mid2 = PSI(
            'H', 'P', p_mid2 * 1e5,
            'S', PSI('S', 'Q', 1, 'P', p_evap2 * 1e5, self.wf2),
            self.wf2
        ) * 1e-3

        # Main cycle
        self.conns['A3'].set_attr(x=self.params['A3']['x'], p=p_evap2)
        self.conns['A0'].set_attr(p=self.params['A0']['p'], h=h_trans_out, fluid={self.wf2: 1})
        self.conns['A5'].set_attr(p=p_mid2, h=h_s_mid2)
        self.conns['D3'].set_attr(x=self.params['D3']['x'], p=p_evap1)
        self.conns['D0'].set_attr(p=p_cond1, fluid={self.wf1: 1})
        self.conns['D5'].set_attr(p=p_mid1, h=h_s_mid1)
        # Heat source
        self.conns['B1'].set_attr(
            T=self.params['B1']['T'], p=self.params['B1']['p'],
            fluid={self.so: 1}
        )
        self.conns['B2'].set_attr(T=self.params['B2']['T'])
        self.conns['B3'].set_attr(p=self.params['B1']['p'])

        # Heat sink
        c1_p = self.params['C1'].get('p', self.params.get('C3', {}).get('p', None))
        self.conns['C1'].set_attr(
            T=self.params['C1']['T'],
            p=c1_p,
            fluid={self.si: 1}
            )
        self.conns['C2'].set_attr(T=t_sink_hot)
        c3_p = self.params.get('C3', {}).get('p', c1_p)
        if c3_p is not None:
            self.conns['C3'].set_attr(p=c3_p)
        c1_m = self.params['C1'].get('m', None)
        if c1_m is None:
            cons = self.params.get('cons', {})
            q_cons = cons.get('Q')
            t_c1 = self.params['C1']['T']
            t_c2 = t_sink_hot
            if q_cons is not None and t_c2 != t_c1:
                cp_w = 4180.0
                c1_m = abs(float(q_cons)) / (cp_w * abs(t_c2 - t_c1))
        if c1_m is not None:
            self.conns['C1'].set_attr(m=c1_m)

        ic2_cool = self.params.get('ic2_cool', {})
        self.conns['F1'].set_attr(
            T=ic2_cool.get('T', 25),
            p=ic2_cool.get('p', 1.013),
            m=ic2_cool.get('m', 5.0),
            fluid={self.si: 1}
        )
        ic1_cool = self.params.get('ic1_cool', {})
        self.conns['G1'].set_attr(
            T=ic1_cool.get('T', 25),
            p=ic1_cool.get('p', 1.013),
            m=ic1_cool.get('m', 5.0),
            fluid={self.si: 1}
        )

        # Perform initial simulation and unset starting values
        self._solve_model(**kwargs)

        self.conns['A0'].set_attr(h=None)
        self.conns['A3'].set_attr(p=None)
        self.conns['A5'].set_attr(h=None)
        self.conns['D0'].set_attr(p=None)
        self.conns['D3'].set_attr(p=None)
        self.conns['D5'].set_attr(h=None)
        self.conns['A4'].set_attr(h=None)
        self.conns['A6'].set_attr(h=None)
        self.conns['D4'].set_attr(h=None)
        self.conns['D6'].set_attr(h=None)
    def design_simulation(self, **kwargs):
        """Perform final parametrization and design simulation."""
        self.comps['LT_comp1'].set_attr(eta_s=self.params['LT_comp1']['eta_s'])
        self.comps['LT_comp2'].set_attr(eta_s=self.params['LT_comp2']['eta_s'])
        self.comps['HT_comp1'].set_attr(eta_s=self.params['HT_comp1']['eta_s'])
        self.comps['HT_comp2'].set_attr(eta_s=self.params['HT_comp2']['eta_s'])
        self.comps['evap'].set_attr(ttd_l=self.params['evap']['ttd_l'])
        self.comps['trans'].set_attr(ttd_l=self.params['trans']['ttd_l'])
        self.comps['inter'].set_attr(ttd_u=self.params['inter']['ttd_u'])
        self.conns['A3'].set_attr(T=self.T_mid - self.params['inter']['ttd_u'] / 2)

        ic2_params = self.params.get('ic2', {})
        ic1_params = self.params.get('ic1', {})
        if 'Td_bp' in ic2_params:
            self.conns['A5'].set_attr(Td_bp=ic2_params['Td_bp'], T=None)
        else:
            T_bp2 = PSI('T', 'P', self.conns['A4'].p.val_SI, 'Q', 1, self.wf2) - 273.15
            if abs(T_bp2 - self.conns['A4'].T.val) < abs(ic2_params['dT_ic']):
                self.conns['A5'].set_attr(Td_bp=1, T=None)
            else:
                self.conns['A5'].set_attr(
                    T=Ref(self.conns['A4'], 1, ic2_params['dT_ic']),
                    Td_bp=None
                )

        if 'Td_bp' in ic1_params:
            self.conns['D5'].set_attr(Td_bp=ic1_params['Td_bp'], T=None)
        else:
            T_bp1 = PSI('T', 'P', self.conns['D4'].p.val_SI, 'Q', 1, self.wf1) - 273.15
            if abs(T_bp1 - self.conns['D4'].T.val) < abs(ic1_params['dT_ic']):
                self.conns['D5'].set_attr(Td_bp=1, T=None)
            else:
                self.conns['D5'].set_attr(
                    T=Ref(self.conns['D4'], 1, ic1_params['dT_ic']),
                    Td_bp=None
                )

        self._solve_model(**kwargs)

        self.m_design = self.conns['A0'].m.val

         

    def intermediate_states_offdesign(self, T_hs_ff, T_cons_ff, deltaT_hs):
        """Calculates intermediate states during part-load simulation"""
        self.T_mid = ((T_hs_ff - deltaT_hs) + T_cons_ff) / 4
        self.conns['A3'].set_attr(
            T=self.T_mid - self.params['inter']['ttd_u'] / 2
        )
        _, _, p_mid1, _, _, p_mid2 = self.get_pressure_levels(
            T_evap=T_hs_ff, T_mid=self.T_mid
        )
        self.conns['A5'].set_attr(p=p_mid2)
        self.conns['D5'].set_attr(p=p_mid1)

    def get_pressure_levels(self, T_evap, T_mid):
        """Calculate evaporation, condensation amd intermediate pressure in bar
        for both cycles and heat sink outlet enthalpy (hot side)."""
        p_evap1 = PSI(
            'P', 'Q', 1,
            'T', T_evap - self.params['evap']['ttd_l'] + 273.15,
            self.wf1
        ) * 1e-5
        p_cond1 = PSI(
            'P', 'Q', 0,
            'T', T_mid + self.params['inter']['ttd_u'] / 2 + 273.15,
            self.wf1
        ) * 1e-5
        p_mid1 = np.sqrt(p_evap1 * p_cond1)
        p_evap2 = PSI(
            'P', 'Q', 1,
            'T', T_mid - self.params['inter']['ttd_u'] / 2 + 273.15,
            self.wf2
        ) * 1e-5
        t_sink_hot = self.params.get('C2', {}).get('T', self.params.get('C3', {}).get('T', self.params['C1']['T']))
        h_trans_out = PSI(
            'H', 'P', self.params['A0']['p'] * 1e5,
            'T', t_sink_hot + self.params['trans']['ttd_l'] + 273.15,
            self.wf2
        ) * 1e-3
        p_mid2 = np.sqrt(p_evap2 * self.params['A0']['p'])

        return p_evap1, p_cond1, p_mid1, p_evap2, h_trans_out, p_mid2
    def get_plotting_states(self, **kwargs):
        """Generate data of states to plot in state diagram."""
        data = {}
        if kwargs['cycle'] == 1:
            data.update(
                {self.comps['inter'].label:
                     self.comps['inter'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['valve1'].label:
                     self.comps['valve1'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['evap'].label:
                     self.comps['evap'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['LT_comp1'].label:
                     self.comps['LT_comp1'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['ic1'].label:
                     self.comps['ic1'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['LT_comp2'].label:
                     self.comps['LT_comp2'].get_plotting_data()[1]}
            )
        elif kwargs['cycle'] == 2:
            data.update(
                {self.comps['trans'].label:
                     self.comps['trans'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['valve2'].label:
                     self.comps['valve2'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['inter'].label:
                     self.comps['inter'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['HT_comp1'].label:
                     self.comps['HT_comp1'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['ic2'].label:
                     self.comps['ic2'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['HT_comp2'].label:
                     self.comps['HT_comp2'].get_plotting_data()[1]}
            )
        else:
            raise ValueError(
                f'Cycle {kwargs["cycle"]} not defined for heat pump '
                + f"'{self.params['setup']['type']}'."
            )

        for comp in data:
            if 'Compressor' in comp:
                data[comp]['starting_point_value'] *= 0.999999

        return data

    def check_consistency(self):
        """Perform all necessary checks to protect consistency of parameters."""
        super().check_consistency()

        self.check_mid_temperature(wf=self.wf1)
