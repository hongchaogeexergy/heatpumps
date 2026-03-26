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
from tespy.tools.characteristics import CharLine
from tespy.tools.characteristics import load_default_char as ldc

if __name__ == '__main__':
    from HeatPumpBase import HeatPumpBase
else:
    from .HeatPumpBase import HeatPumpBase


class HeatPumpPCIHX(HeatPumpBase):
    """Heat pump with open/closed economizer, parallel compression and ihx."""

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

        # Main cycle
        self.comps['cond'] = Condenser('Condenser')
        self.comps['cc'] = CycleCloser('Main Cycle Closer')
        self.comps['mid_valve'] = Valve('Intermediate Valve')
        self.comps['evap_valve'] = Valve('Evaporation Valve')
        self.comps['evap'] = HeatExchanger('Evaporator')
        self.comps['ihx'] = HeatExchanger('Internal Heat Exchanger')
        self.comps['comp1'] = Compressor('Compressor 1')
        self.comps['comp2'] = Compressor('Compressor 2')
        self.comps['merge'] = Merge('Economizer Injection')

        if self.econ_type.lower() == 'closed':
            self.comps['split'] = Splitter('Condensate Splitter')
            self.comps['econ'] = HeatExchanger('Economizer')
        elif self.econ_type.lower() == 'open':
            self.comps['econ'] = DropletSeparator('Economizer')
        else:
            raise ValueError(
                f"Parameter '{self.econ_type}' is not a valid econ_type. "
                + "Supported values are 'open' and 'closed'."
                )

    def generate_connections(self):
        """Initialize and add connections and busses to network."""
        # Connections
        self.conns['A0'] = Connection(
            self.comps['cond'], 'out1', self.comps['cc'], 'in1', 'A0'
            )
        self.conns['A3'] = Connection(
            self.comps['econ'], 'out1', self.comps['ihx'], 'in1', 'A3'
            )
        self.conns['A4'] = Connection(
            self.comps['ihx'], 'out1', self.comps['evap_valve'], 'in1', 'A4'
            )
        self.conns['A5'] = Connection(
            self.comps['evap_valve'], 'out1', self.comps['evap'], 'in2', 'A5'
            )
        self.conns['A6'] = Connection(
            self.comps['evap'], 'out2', self.comps['ihx'], 'in2', 'A6'
            )
        self.conns['A7'] = Connection(
            self.comps['ihx'], 'out2', self.comps['comp1'], 'in1', 'A7'
            )
        self.conns['A8'] = Connection(
            self.comps['comp1'], 'out1', self.comps['merge'], 'in1', 'A8'
            )
        self.conns['A9'] = Connection(
            self.comps['merge'], 'out1', self.comps['cond'], 'in1', 'A9'
            )
        self.conns['A10'] = Connection(
            self.comps['econ'], 'out2', self.comps['comp2'], 'in1', 'A10'
            )
        self.conns['A11'] = Connection(
            self.comps['comp2'], 'out1', self.comps['merge'], 'in2', 'A11'
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

        if self.econ_type.lower() == 'closed':
            self.conns['A1'] = Connection(
                self.comps['cc'], 'out1', self.comps['split'], 'in1', 'A1'
                )
            self.conns['A2'] = Connection(
                self.comps['split'], 'out1', self.comps['econ'], 'in1', 'A2'
                )
            self.conns['A12'] = Connection(
                self.comps['split'], 'out2',
                self.comps['mid_valve'], 'in1', 'A12'
                )
            self.conns['A13'] = Connection(
                self.comps['mid_valve'], 'out1',
                self.comps['econ'], 'in2', 'A13'
                )
        elif self.econ_type.lower() == 'open':
            self.conns['A1'] = Connection(
                self.comps['cc'], 'out1', self.comps['mid_valve'], 'in1', 'A1'
                )
            self.conns['A2'] = Connection(
                self.comps['mid_valve'], 'out1', self.comps['econ'], 'in1', 'A2'
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
        self.comps['power_bus'] = PowerBus('Power Distribution', num_in=1, num_out=4)

        self.comps['motor_comp1'] = Motor('comp1 motor')
        self.comps['motor_comp2'] = Motor('comp2 motor')
        self.comps['motor_hs'] = Motor('HS_pump motor')
        self.comps['motor_cons'] = Motor('Cons_pump motor')

        # Power connections
        self.conns['E0'] = PowerConnection(self.comps['power_source'], 'power',
                                        self.comps['power_bus'], 'power_in1', label='E0')

        self.conns['E1'] = PowerConnection(self.comps['power_bus'], 'power_out1',
                                        self.comps['motor_comp1'], 'power_in', label='E1')
        self.conns['e1'] = PowerConnection(self.comps['motor_comp1'], 'power_out',
                                        self.comps['comp1'], 'power', label='e1')

        self.conns['E2'] = PowerConnection(self.comps['power_bus'], 'power_out2',
                                        self.comps['motor_comp2'], 'power_in', label='E2')
        self.conns['e2'] = PowerConnection(self.comps['motor_comp2'], 'power_out',
                                        self.comps['comp2'], 'power', label='e2')

        self.conns['E3'] = PowerConnection(self.comps['power_bus'], 'power_out3',
                                        self.comps['motor_hs'], 'power_in', label='E3')
        self.conns['e3'] = PowerConnection(self.comps['motor_hs'], 'power_out',
                                        self.comps['hs_pump'], 'power', label='e3')

        self.conns['E4'] = PowerConnection(self.comps['power_bus'], 'power_out4',
                                        self.comps['motor_cons'], 'power_in', label='E4')
        self.conns['e4'] = PowerConnection(self.comps['motor_cons'], 'power_out',
                                        self.comps['hsink_pump'], 'power', label='e4')

        # Add all connections to network
        self.nw.add_conns(*[conn for conn in self.conns.values()])
        

        # Set motor efficiency attributes
        for motor_label in ['motor_comp1', 'motor_comp2', 'motor_hs', 'motor_cons']:
            self.comps[motor_label].set_attr(eta=0.98, eta_char=mot)


    def init_simulation(self, **kwargs):
        """Perform initial parametrization with starting values."""
        # Components
        self.conns['A8'].set_attr(
            h=Ref(self.conns['A7'], self._init_vals['dh_rel_comp'], 0)
            )
        self.conns['A11'].set_attr(
            h=Ref(self.conns['A10'], self._init_vals['dh_rel_comp'], 0)
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
        if self.econ_type.lower() == 'closed':
            self.comps['econ'].set_attr(
                pr1=self.params['econ']['pr1'], pr2=self.params['econ']['pr2']
                )

        # Connections
        # Starting values
        t_sink_hot = self.params.get('C2', {}).get('T', self.params.get('C3', {}).get('T', self.params['C1']['T']))
        p_evap, p_cond, p_mid = self.get_pressure_levels(
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
        self.conns['A6'].set_attr(x=self.params['A6']['x'], p=p_evap)
        self.conns['A0'].set_attr(p=p_cond, fluid={self.wf: 1})
        self.conns['A7'].set_attr(h=h_superheat)
        self.conns['A10'].set_attr(p=p_mid)
        if self.econ_type.lower() == 'closed':
            self.conns['A10'].set_attr(x=1)
            self.conns['A2'].set_attr(
                m=Ref(self.conns['A0'], 0.9, 0)
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

        if self.econ_type == 'closed':
            self.conns['A2'].set_attr(m=None)
        self.conns['A6'].set_attr(p=None)
        self.conns['A0'].set_attr(p=None)
        self.conns['A7'].set_attr(h=None)
        self.conns['A8'].set_attr(h=None)
        self.conns['A11'].set_attr(h=None)
    def design_simulation(self, **kwargs):
        """Perform final parametrization and design simulation."""
        self.comps['comp1'].set_attr(eta_s=self.params['comp1']['eta_s'])
        self.comps['comp2'].set_attr(eta_s=self.params['comp2']['eta_s'])
        self.comps['evap'].set_attr(ttd_l=self.params['evap']['ttd_l'])
        self.comps['cond'].set_attr(ttd_u=self.params['cond']['ttd_u'])
        if self.econ_type == 'closed':
            self.comps['econ'].set_attr(ttd_l=self.params['econ']['ttd_l'])
        self.conns['A7'].set_attr(
            T=Ref(self.conns['A6'], 1, self.params['ihx']['dT_sh'])
            )

        self._solve_model(**kwargs)

        self.m_design = self.conns['A0'].m.val

         

    def intermediate_states_offdesign(self, T_hs_ff, T_cons_ff, deltaT_hs):
        """Calculates intermediate states during part-load simulation"""
        _, _, p_mid = self.get_pressure_levels(
            T_evap=T_hs_ff, T_cond=T_cons_ff
        )
        self.conns['A10'].set_attr(p=p_mid)

    def get_plotting_states(self, **kwargs):
        """Generate data of states to plot in state diagram."""
        data = {}
        data.update(
            {self.comps['cond'].label:
             self.comps['cond'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['mid_valve'].label:
             self.comps['mid_valve'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['econ'].label + ' (hot)':
                self.comps['econ'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['econ'].label + ' (cold)':
                self.comps['econ'].get_plotting_data()[2]}
        )
        data.update(
            {self.comps['ihx'].label + ' (hot)':
                self.comps['ihx'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['evap_valve'].label:
             self.comps['evap_valve'].get_plotting_data()[1]}
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
            {self.comps['comp1'].label:
             self.comps['comp1'].get_plotting_data()[1]}
        )
        data.update(
            {'Main gas stream': self.comps['merge'].get_plotting_data()[1]}
            )
        data.update(
            {'Parallel gas stream': self.comps['merge'].get_plotting_data()[2]}
            )
        data.update(
            {self.comps['comp2'].label:
             self.comps['comp2'].get_plotting_data()[1]}
        )

        for comp in data:
            if 'Compressor' in comp:
                data[comp]['starting_point_value'] *= 0.999999

        return data
