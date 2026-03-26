# UPDATED: heat sink loop refactor applied
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI as PSI
from tespy.components import (Compressor, Condenser, CycleCloser,
                              DropletSeparator, HeatExchanger, Merge, Pump,
                              SimpleHeatExchanger, Sink, Source, Splitter,
                              Valve, PowerSource, PowerBus, Motor)
from tespy.connections import Connection, Ref, PowerConnection
from tespy.networks import Network
from tespy.tools.characteristics import CharLine
from tespy.tools.characteristics import load_default_char as ldc

if __name__ == '__main__':
    from HeatPumpCascadeBase import HeatPumpCascadeBase
else:
    from .HeatPumpCascadeBase import HeatPumpCascadeBase


class HeatPumpCascadeIHXPCIHX(HeatPumpCascadeBase):
    """Two stage cascading heat pump with open or closed economizer,
    parallel compression with two internal heat exchanger."""

    def __init__(self, params, econ_type='closed'):
        """Initialize model and set necessary attributes."""
        super().__init__(params)
        self.econ_type = econ_type

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

        # Main upper cycle
        self.comps['cond'] = Condenser('Condenser')
        self.comps['cc2'] = CycleCloser('Main Cycle Closer 2')
        self.comps['ihx4'] = HeatExchanger('High Pressure Internal Heat Exchanger 2')
        self.comps['mid_valve2'] = Valve('Intermediate Valve 2')
        self.comps['evap_valve2'] = Valve('Evaporation Valve 2')
        self.comps['inter'] = Condenser('Intermediate Heat Exchanger')
        self.comps['ihx3'] = HeatExchanger('Low Pressure Internal Heat Exchanger 2')
        self.comps['HT_comp1'] = Compressor('High Temperature Compressor 1')
        self.comps['merge2'] = Merge('Low Temperature Economizer Injection')
        self.comps['HT_comp2'] = Compressor('High Temperature Compressor 2')

        # Main lower cycle
        self.comps['cc1'] = CycleCloser('Main Cycle Closer 1')
        self.comps['ihx2'] = HeatExchanger('High Pressure Internal Heat Exchanger 1')
        self.comps['mid_valve1'] = Valve('Intermediate Valve 1')
        self.comps['evap_valve1'] = Valve('Evaporation Valve 1')
        self.comps['evap'] = HeatExchanger('Evaporator')
        self.comps['ihx1'] = HeatExchanger('Low Pressure Internal Heat Exchanger 1')
        self.comps['LT_comp1'] = Compressor('Low Temperature Compressor 1')
        self.comps['merge1'] = Merge('High Temperature Economizer Injection')
        self.comps['LT_comp2'] = Compressor('Low Temperature Compressor 2')

        if self.econ_type.lower() == 'closed':
            self.comps['split1'] = Splitter('Intermediate Condensate Splitter')
            self.comps['split2'] = Splitter('Condensate Splitter')
            self.comps['econ1'] = HeatExchanger('Low Temperature Economizer')
            self.comps['econ2'] = HeatExchanger('High Temperature Economizer')
        elif self.econ_type.lower() == 'open':
            self.comps['econ1'] = DropletSeparator('Low Temperature Economizer')
            self.comps['econ2'] = DropletSeparator('High Temperature Economizer')
        else:
            raise ValueError(
                f"Parameter '{self.econ_type}' is not a valid econ_type. "
                + "Supported values are 'open' and 'closed'."
            )

    def generate_connections(self):
        """Initialize and add connections and buses to network."""
        # Connections High Temperature cycle
        self.conns['A0'] = Connection(
            self.comps['cond'], 'out1', self.comps['cc2'], 'in1', 'A0'
        )
        self.conns['A1'] = Connection(
            self.comps['cc2'], 'out1', self.comps['ihx4'], 'in1', 'A1'
        )
        self.conns['A4'] = Connection(
            self.comps['econ2'], 'out1', self.comps['ihx3'], 'in1', 'A4'
        )
        self.conns['A5'] = Connection(
            self.comps['ihx3'], 'out1', self.comps['evap_valve2'], 'in1', 'A5'
        )
        self.conns['A6'] = Connection(
            self.comps['evap_valve2'], 'out1', self.comps['inter'], 'in2', 'A6'
        )
        self.conns['A7'] = Connection(
            self.comps['inter'], 'out2', self.comps['ihx3'], 'in2', 'A7'
        )
        self.conns['A8'] = Connection(
            self.comps['ihx3'], 'out2', self.comps['HT_comp1'], 'in1', 'A8'
        )
        self.conns['A9'] = Connection(
            self.comps['HT_comp1'], 'out1', self.comps['merge2'], 'in1', 'A9'
        )
        self.conns['A10'] = Connection(
            self.comps['merge2'], 'out1', self.comps['cond'], 'in1', 'A10'
        )
        self.conns['A11'] = Connection(
            self.comps['econ2'], 'out2', self.comps['ihx4'], 'in2', 'A11'
        )
        self.conns['A12'] = Connection(
            self.comps['ihx4'], 'out2', self.comps['HT_comp2'], 'in1', 'A12'
        )
        self.conns['A13'] = Connection(
            self.comps['HT_comp2'], 'out1', self.comps['merge2'], 'in2', 'A13'
        )

        if self.econ_type.lower() == 'closed':
            self.conns['A2'] = Connection(
                self.comps['ihx4'], 'out1', self.comps['split2'], 'in1', 'A2'
            )
            self.conns['A3'] = Connection(
                self.comps['split2'], 'out1', self.comps['econ2'], 'in1', 'A3'
            )
            self.conns['A14'] = Connection(
                self.comps['split2'], 'out2', self.comps['mid_valve2'], 'in1', 'A14'
            )
            self.conns['A15'] = Connection(
                self.comps['mid_valve2'], 'out1', self.comps['econ2'], 'in2', 'A15'
            )
        elif self.econ_type.lower() == 'open':
            self.conns['A2'] = Connection(
                self.comps['ihx4'], 'out1', self.comps['mid_valve2'], 'in1', 'A2'
            )
            self.conns['A3'] = Connection(
                self.comps['mid_valve2'], 'out1', self.comps['econ2'], 'in1', 'A3'
            )

        # connections Low Temperature cycle
        self.conns['D0'] = Connection(
            self.comps['inter'], 'out1', self.comps['cc1'], 'in1', 'D0'
        )
        self.conns['D1'] = Connection(
            self.comps['cc1'], 'out1', self.comps['ihx2'], 'in1', 'D1'
        )
        self.conns['D4'] = Connection(
            self.comps['econ1'], 'out1', self.comps['ihx1'], 'in1', 'D4'
        )
        self.conns['D5'] = Connection(
            self.comps['ihx1'], 'out1', self.comps['evap_valve1'], 'in1', 'D5'
        )
        self.conns['D6'] = Connection(
            self.comps['evap_valve1'], 'out1', self.comps['evap'], 'in2', 'D6'
        )
        self.conns['D7'] = Connection(
            self.comps['evap'], 'out2', self.comps['ihx1'], 'in2', 'D7'
        )
        self.conns['D8'] = Connection(
            self.comps['ihx1'], 'out2', self.comps['LT_comp1'], 'in1', 'D8'
        )
        self.conns['D9'] = Connection(
            self.comps['LT_comp1'], 'out1', self.comps['merge1'], 'in1', 'D9'
        )
        self.conns['D10'] = Connection(
            self.comps['merge1'], 'out1', self.comps['inter'], 'in1', 'D10'
        )
        self.conns['D11'] = Connection(
            self.comps['econ1'], 'out2', self.comps['ihx2'], 'in2', 'D11'
        )
        self.conns['D12'] = Connection(
            self.comps['ihx2'], 'out2', self.comps['LT_comp2'], 'in1', 'D12'
        )
        self.conns['D13'] = Connection(
            self.comps['LT_comp2'], 'out1', self.comps['merge1'], 'in2', 'D13'
        )

        if self.econ_type.lower() == 'closed':
            self.conns['DD'] = Connection(
                self.comps['ihx2'], 'out1', self.comps['split1'], 'in1', 'D2'
            )
            self.conns['D3'] = Connection(
                self.comps['split1'], 'out1', self.comps['econ1'], 'in1', 'D3'
            )
            self.conns['D14'] = Connection(
                self.comps['split1'], 'out2', self.comps['mid_valve1'], 'in1', 'D14'
            )
            self.conns['D15'] = Connection(
                self.comps['mid_valve1'], 'out1', self.comps['econ1'], 'in2', 'D15'
            )
        elif self.econ_type.lower() == 'open':
            self.conns['D2'] = Connection(
                self.comps['ihx2'], 'out1', self.comps['mid_valve1'], 'in1', 'D2'
            )
            self.conns['D3'] = Connection(
                self.comps['mid_valve1'], 'out1', self.comps['econ1'], 'in1', 'D3'
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
            self.comps['hsink_ff'], 'out1', self.comps['cond'], 'in2', 'C1'
            )
        self.conns['C2'] = Connection(
            self.comps['cond'], 'out2', self.comps['hsink_pump'], 'in1', 'C2'
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
        self.conns['A9'].set_attr(
            h=Ref(self.conns['A8'], self._init_vals['dh_rel_comp'], 0)
            )
        self.conns['A13'].set_attr(
            h=Ref(self.conns['A12'], self._init_vals['dh_rel_comp'], 0)
            )
        self.conns['D9'].set_attr(
            h=Ref(self.conns['D8'], self._init_vals['dh_rel_comp'], 0)
            )
        self.conns['D13'].set_attr(
            h=Ref(self.conns['D12'], self._init_vals['dh_rel_comp'], 0)
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
        self.comps['cond'].set_attr(
            pr1=self.params['cond']['pr1'], pr2=self.params['cond']['pr2']
        )
        self.comps['ihx2'].set_attr(
            pr1=self.params['ihx2']['pr1'], pr2=self.params['ihx2']['pr1']
        )
        self.comps['ihx1'].set_attr(
            pr1=self.params['ihx1']['pr1'], pr2=self.params['ihx1']['pr1']
        )
        self.comps['ihx4'].set_attr(
            pr1=self.params['ihx4']['pr1'], pr2=self.params['ihx4']['pr1']
        )
        self.comps['ihx3'].set_attr(
            pr1=self.params['ihx3']['pr1'], pr2=self.params['ihx3']['pr1']
        )
        if self.econ_type.lower() == 'closed':
            self.comps['econ1'].set_attr(
                pr1=self.params['econ1']['pr1'], pr2=self.params['econ1']['pr2']
            )
            self.comps['econ2'].set_attr(
                pr1=self.params['econ2']['pr1'], pr2=self.params['econ2']['pr2']
            )

        # Connections
        t_sink_hot = self.params.get('C2', {}).get('T', self.params.get('C3', {}).get('T', self.params['C1']['T']))
        self.T_mid = (self.params['B2']['T'] + t_sink_hot) / 2

        # Starting values
        p_evap1, p_cond1, p_mid1, p_evap2, p_cond2, p_mid2 = self.get_pressure_levels(
            T_evap=self.params['B2']['T'], T_mid=self.T_mid,
            T_cond=t_sink_hot
        )
        self.p_evap1 = p_evap1
        self.p_evap2 = p_evap2
        self.p_mid1 = p_mid1
        self.p_mid2 = p_mid2

        T_mid2 = PSI('T', 'Q', 1, 'P', p_mid2 * 1e5, self.wf2) - 273.15
        T_mid1 = PSI('T', 'Q', 1, 'P', p_mid1 * 1e5, self.wf1) - 273.15

        lp_h_superheat1 = PSI(
            'H', 'P', p_evap1 * 1e5,
            'T', (
                    self.params['B2']['T'] - self.params['evap']['ttd_l'] + 273.15
                    + self.params['ihx1']['dT_sh']),
            self.wf1
        ) * 1e-3
        hp_h_superheat1 = PSI(
            'H', 'P', p_mid1 * 1e5,
            'T', (T_mid1 + 273.15 + self.params['ihx2']['dT_sh']),
            self.wf1
        ) * 1e-3
        lp_h_superheat2 = PSI(
            'H', 'P', p_evap2 * 1e5,
            'T', (
                    self.T_mid - self.params['inter']['ttd_u'] / 2 + 273.15
                    + self.params['ihx3']['dT_sh']),
            self.wf2
        ) * 1e-3
        hp_h_superheat2 = PSI(
            'H', 'P', p_mid2 * 1e5,
            'T', (T_mid2 + 273.15 + self.params['ihx4']['dT_sh']),
            self.wf2
        ) * 1e-3

        # Main cycle
        self.conns['A7'].set_attr(x=self.params['A7']['x'], p=p_evap2)
        self.conns['A0'].set_attr(p=p_cond2, fluid={self.wf2: 1})
        self.conns['A8'].set_attr(h=lp_h_superheat2)
        self.conns['A11'].set_attr(p=p_mid2)
        self.conns['A12'].set_attr(h=hp_h_superheat2)
        self.conns['D7'].set_attr(x=self.params['D7']['x'], p=p_evap1)
        self.conns['D0'].set_attr(p=p_cond1, fluid={self.wf1: 1})
        self.conns['D8'].set_attr(h=lp_h_superheat1)
        self.conns['D11'].set_attr(p=p_mid1)
        self.conns['D12'].set_attr(h=hp_h_superheat1)
        if self.econ_type.lower() == 'closed':
            self.conns['A11'].set_attr(x=1)
            self.conns['A3'].set_attr(
                m=Ref(self.conns['A0'], 0.9, 0)
            )
            self.conns['D11'].set_attr(x=1)
            self.conns['D3'].set_attr(
                m=Ref(self.conns['D0'], 0.9, 0)
            )

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

        # Perform initial simulation and unset starting values
        self._solve_model(**kwargs)

        self.conns['A0'].set_attr(p=None)
        self.conns['A7'].set_attr(p=None)
        self.conns['A8'].set_attr(h=None)
        self.conns['A12'].set_attr(h=None)
        self.conns['D0'].set_attr(p=None)
        self.conns['D7'].set_attr(p=None)
        self.conns['D8'].set_attr(h=None)
        self.conns['D12'].set_attr(h=None)
        if self.econ_type == 'closed':
            self.conns['A3'].set_attr(m=None)
            self.conns['D3'].set_attr(m=None)
        self.conns['A9'].set_attr(h=None)
        self.conns['A13'].set_attr(h=None)
        self.conns['D9'].set_attr(h=None)
        self.conns['D13'].set_attr(h=None)
    def design_simulation(self, **kwargs):
        """Perform final parametrization and design simulation."""
        self.comps['HT_comp1'].set_attr(eta_s=self.params['HT_comp1']['eta_s'])
        self.comps['HT_comp2'].set_attr(eta_s=self.params['HT_comp2']['eta_s'])
        self.comps['LT_comp1'].set_attr(eta_s=self.params['LT_comp1']['eta_s'])
        self.comps['LT_comp2'].set_attr(eta_s=self.params['LT_comp2']['eta_s'])
        self.comps['evap'].set_attr(ttd_l=self.params['evap']['ttd_l'])
        self.comps['cond'].set_attr(ttd_u=self.params['cond']['ttd_u'])
        self.comps['inter'].set_attr(ttd_u=self.params['inter']['ttd_u'])
        self.conns['A7'].set_attr(T=self.T_mid - self.params['inter']['ttd_u'] / 2)
        self.conns['A8'].set_attr(
            T=Ref(self.conns['A7'], 1, self.params['ihx3']['dT_sh'])
        )
        self.conns['A12'].set_attr(
            T=Ref(self.conns['A11'], 1, self.params['ihx4']['dT_sh'])
        )
        self.conns['D8'].set_attr(
            T=Ref(self.conns['D7'], 1, self.params['ihx1']['dT_sh'])
        )
        self.conns['D12'].set_attr(
            T=Ref(self.conns['D11'], 1, self.params['ihx2']['dT_sh'])
        )

        if self.econ_type == 'closed':
            self.comps['econ1'].set_attr(ttd_l=self.params['econ1']['ttd_l'])
            self.comps['econ2'].set_attr(ttd_l=self.params['econ2']['ttd_l'])

        self._solve_model(**kwargs)

        self.m_design = self.conns['A0'].m.val

         

    def intermediate_states_offdesign(self, T_hs_ff, T_cons_ff, deltaT_hs):
        """Calculates intermediate states during part-load simulation"""
        self.T_mid = ((T_hs_ff - deltaT_hs) + T_cons_ff) / 2
        self.conns['A7'].set_attr(
            T=self.T_mid - self.params['inter']['ttd_u'] / 2
        )
        _, _, p_mid1, _, _, p_mid2 = self.get_pressure_levels(
            T_evap=T_hs_ff, T_mid=self.T_mid, T_cond=T_cons_ff
        )
        self.conns['A11'].set_attr(p=p_mid2)
        self.conns['D11'].set_attr(p=p_mid1)

    def get_pressure_levels(self, T_evap, T_mid, T_cond):
        """Calculate evaporation, condensation amd intermediate pressure in bar for both cycles."""
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
        p_cond2 = PSI(
            'P', 'Q', 0,
            'T', T_cond + self.params['cond']['ttd_u'] + 273.15,
            self.wf2
        ) * 1e-5
        p_mid2 = np.sqrt(p_evap2 * p_cond2)

        return p_evap1, p_cond1, p_mid1, p_evap2, p_cond2, p_mid2

    def get_plotting_states(self, **kwargs):
        """Generate data of states to plot in state diagram."""
        data = {}
        if kwargs['cycle'] == 1:
            data.update(
                {self.comps['inter'].label:
                     self.comps['inter'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['ihx2'].label + ' (hot)':
                     self.comps['ihx2'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['econ1'].label + ' (hot)':
                     self.comps['econ1'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['econ1'].label + ' (cold)':
                     self.comps['econ1'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['ihx2'].label + ' (cold)':
                     self.comps['ihx2'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['ihx1'].label + ' (hot)':
                     self.comps['ihx1'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['mid_valve1'].label:
                     self.comps['mid_valve1'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['evap_valve1'].label:
                     self.comps['evap_valve1'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['evap'].label:
                     self.comps['evap'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['ihx1'].label + ' (cold)':
                     self.comps['ihx1'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['LT_comp1'].label:
                     self.comps['LT_comp1'].get_plotting_data()[1]}
            )
            data.update(
                {'Injection steam cycle 1': self.comps['merge1'].get_plotting_data()[1]}
            )
            data.update(
                {'Compressed gas cycle 1': self.comps['merge1'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['LT_comp2'].label:
                     self.comps['LT_comp2'].get_plotting_data()[1]}
            )
        elif kwargs['cycle'] == 2:
            data.update(
                {self.comps['cond'].label:
                     self.comps['cond'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['ihx4'].label + ' (hot)':
                     self.comps['ihx4'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['econ2'].label + ' (hot)':
                     self.comps['econ2'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['econ2'].label + ' (cold)':
                     self.comps['econ2'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['ihx4'].label + ' (cold)':
                     self.comps['ihx4'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['ihx3'].label + ' (hot)':
                     self.comps['ihx3'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['mid_valve2'].label:
                     self.comps['mid_valve2'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['evap_valve2'].label:
                     self.comps['evap_valve2'].get_plotting_data()[1]}
            )
            data.update(
                {self.comps['inter'].label:
                     self.comps['inter'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['ihx3'].label + ' (cold)':
                     self.comps['ihx3'].get_plotting_data()[2]}
            )
            data.update(
                {self.comps['HT_comp1'].label:
                     self.comps['HT_comp1'].get_plotting_data()[1]}
            )
            data.update(
                {'Injection steam cycle 2': self.comps['merge2'].get_plotting_data()[1]}
            )
            data.update(
                {'Compressed gas cycle 2': self.comps['merge2'].get_plotting_data()[2]}
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
