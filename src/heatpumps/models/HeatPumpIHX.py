# UPDATED: heat sink loop refactor applied
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI as PSI
from tespy.components import (Compressor, Condenser, CycleCloser,
                              HeatExchanger, Pump, SimpleHeatExchanger, Sink,
                              Source, Valve,PowerSource, PowerBus, Motor  # added for TESPy 0.9 power flow
)
from tespy.connections import Connection, Ref, PowerConnection  # PowerConnection added

from tespy.tools.characteristics import CharLine
from tespy.tools.characteristics import load_default_char as ldc

if __name__ == '__main__':
    from HeatPumpBase import HeatPumpBase
else:
    from .HeatPumpBase import HeatPumpBase


class HeatPumpIHX(HeatPumpBase):
    """Heat pump with internal heat exchanger between condesate and vapor."""

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

        # Main cycle
        self.comps['cond'] = Condenser('Condenser')
        self.comps['cc'] = CycleCloser('Main Cycle Closer')
        self.comps['ihx'] = HeatExchanger('Internal Heat Exchanger')
        self.comps['valve'] = Valve('Valve')
        self.comps['evap'] = HeatExchanger('Evaporator')
        self.comps['comp'] = Compressor('Compressor')

    def generate_connections(self):
        """Initialize and add connections and busses to network."""
        # Connections
        self.conns['A0'] = Connection(
            self.comps['cond'], 'out1', self.comps['cc'], 'in1', 'A0'
            )
        self.conns['A1'] = Connection(
            self.comps['cc'], 'out1', self.comps['ihx'], 'in1', 'A1'
            )
        self.conns['A2'] = Connection(
            self.comps['ihx'], 'out1', self.comps['valve'], 'in1', 'A2'
            )
        self.conns['A3'] = Connection(
            self.comps['valve'], 'out1', self.comps['evap'], 'in2', 'A3'
            )
        self.conns['A4'] = Connection(
            self.comps['evap'], 'out2', self.comps['ihx'], 'in2', 'A4'
            )
        self.conns['A5'] = Connection(
            self.comps['ihx'], 'out2', self.comps['comp'], 'in1', 'A5'
            )
        self.conns['A6'] = Connection(
            self.comps['comp'], 'out1', self.comps['cond'], 'in1', 'A6'
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

        self.nw.add_conns(*[conn for conn in self.conns.values()])

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
        self.motor_char = CharLine(x=mot_x, y=mot_y)
        # Create power components
        self.comps['power_source'] = PowerSource('Power Input')
        self.comps['power_bus'] = PowerBus('Power Distribution', num_in=1, num_out=3)

        self.comps['motor_comp'] = Motor('comp motor')
        self.comps['motor_hs'] = Motor('HS_pump motor')
        self.comps['motor_cons'] = Motor('Cons_pump motor')
        # E0: Electrical power supply from power source to power bus
        #  Use 'power_in1' as the valid input connector for PowerBus
        self.conns['E0'] = PowerConnection(
            self.comps['power_source'], 'power',
            self.comps['power_bus'], 'power_in1', label='E0'
        )

        # E1: Electrical power from power bus to compressor motor
        #     Assuming PowerBus provides output via 'power_out1'
        self.conns['E1'] = PowerConnection(
            self.comps['power_bus'], 'power_out1',
            self.comps['motor_comp'], 'power_in', label='E1'
        )

        # e1: Mechanical shaft power from compressor motor to compressor
        self.conns['e1'] = PowerConnection(
            self.comps['motor_comp'], 'power_out',
            self.comps['comp'], 'power', label='e1'
        )

        # E2: Electrical power from power bus to heat source pump motor
        #     Assuming PowerBus provides output via 'power_out2'
        self.conns['E2'] = PowerConnection(
            self.comps['power_bus'], 'power_out2',
            self.comps['motor_hs'], 'power_in', label='E2'
        )

        # e2: Mechanical shaft power from heat source pump motor to hs_pump
        self.conns['e2'] = PowerConnection(
            self.comps['motor_hs'], 'power_out',
            self.comps['hs_pump'], 'power', label='e2'
        )

        # E3: Electrical power from power bus to consumer pump motor
        #     Assuming PowerBus provides output via 'power_out3'
        self.conns['E3'] = PowerConnection(
            self.comps['power_bus'], 'power_out3',
            self.comps['motor_cons'], 'power_in', label='E3'
        )

        # e3: Mechanical shaft power from sink pump motor to hsink_pump
        self.conns['e3'] = PowerConnection(
            self.comps['motor_cons'], 'power_out',
            self.comps['hsink_pump'], 'power', label='e3'
        )

        # Add all power connections to the network
        self.nw.add_conns(
            self.conns['E0'], self.conns['E1'], self.conns['e1'],
            self.conns['E2'], self.conns['e2'], self.conns['E3'], self.conns['e3']
        )
        for motor in ['motor_comp', 'motor_hs', 'motor_cons']:
            self.comps[motor].set_attr(
                eta_char=self.motor_char, eta=0.98,
                design=["eta"], offdesign=["eta_char"]
            )

    def init_simulation(self, **kwargs):
        """Perform initial parametrization with starting values."""
        # Components
        self.conns['A6'].set_attr(
            h=Ref(self.conns['A5'], self._init_vals['dh_rel_comp'], 0)
            )
        self.comps['hs_pump'].set_attr(eta_s=self.params['hs_pump']['eta_s'])
        hsink_params = self.params.get('hsink_pump', self.params.get('cons_pump', {}))
        if 'eta_s' in hsink_params:
            self.comps['hsink_pump'].set_attr(eta_s=hsink_params['eta_s'])

        self.comps['evap'].set_attr(
            pr1=self.params['evap']['pr1'], pr2=self.params['evap']['pr2']
            )
        self.comps['cond'].set_attr(
            pr1=self.params['cond']['pr1'], pr2=self.params['cond']['pr2']
            )
        self.comps['ihx'].set_attr(
            pr1=self.params['ihx']['pr1'], pr2=self.params['ihx']['pr1']
            )

        # Connections
        # Starting values
        t_sink_hot = self.params.get('C2', {}).get('T', self.params.get('C3', {}).get('T', self.params['C1']['T']))
        p_evap, p_cond, _ = self.get_pressure_levels(
            T_evap=self.params['B2']['T'], T_cond=t_sink_hot
            )
        h_superheat = PSI(
            'H', 'P', p_evap*1e5,
            'T', (
                self.params['B2']['T'] - self.params['evap']['ttd_l'] + 273.15
                + self.params['ihx']['dT_sh']),
            self.wf
            ) * 1e-3

        # Main cycle
        self.conns['A4'].set_attr(x=self.params['A4']['x'], p=p_evap)
        self.conns['A0'].set_attr(p=p_cond, fluid={self.wf: 1})
        self.conns['A5'].set_attr(h=h_superheat)
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

        self.conns['A4'].set_attr(p=None)
        self.conns['A0'].set_attr(p=None)
        self.conns['A5'].set_attr(h=None)
        self.conns['A6'].set_attr(h=None)
    def design_simulation(self, **kwargs):
        """Perform final parametrization and design simulation."""
        self.comps['comp'].set_attr(eta_s=self.params['comp']['eta_s'])
        self.comps['evap'].set_attr(ttd_l=self.params['evap']['ttd_l'])
        self.comps['cond'].set_attr(ttd_u=self.params['cond']['ttd_u'])
        self.conns['A5'].set_attr(
            T=Ref(self.conns['A4'], 1, self.params['ihx']['dT_sh'])
            )

        self._solve_model(**kwargs)

        self.m_design = self.conns['A0'].m.val

         

    def get_plotting_states(self, **kwargs):
        """Generate data of states to plot in state diagram."""
        data = {}
        data.update(
            {self.comps['cond'].label:
             self.comps['cond'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['ihx'].label + ' (hot)':
             self.comps['ihx'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['valve'].label:
             self.comps['valve'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['evap'].label:
             self.comps['evap'].get_plotting_data()[2]}
        )
        data.update(
            {self.comps['ihx'].label + ' (cold)':
             self.comps['ihx'].get_plotting_data()[2]}
        )
        data.update(
            {self.comps['comp'].label:
             self.comps['comp'].get_plotting_data()[1]}
        )

        for comp in data:
            if 'Compressor' in comp:
                data[comp]['starting_point_value'] *= 0.999999

        return data
