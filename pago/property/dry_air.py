""" dry_air.py
thermal physical properties of dry air
"""

from math import sqrt, pow

class DryAir:
    pc = 3.77e6                          # critical pressure, Pa
    Tc = 132.6                           # critical temperature, K
    mw = 28.9645                         # mol weight, kg/kmol
    Rg = 287.055                         # gas constant, J/kg-K
    def __init__(self):
        self.Tref = 293.15
        self.Pref = 101325.0
        self.rho = 1.205                # density at Tref and Pref
        self.cp = 1006.4                 # specific capacity at Tref and Pref, kJ/kg-K
        self.mu = 1.81e-5               # laminar viscosity at Tref and Pref, kg/m-s
        self.nu = 15.06e-6              # moving viscosity at Tref and Pref, m2/s
        self.Pr = 0.703                 # Prantl number at Tref and Pref
        self.k = 2.59e-2                # thermal conductivity at Tref and Pref, W/m-K
        self.a = 2.14e-5                # bthermal diffusion coefficient at Tref and Pref, m2/s
        self.h0 = 2.9341e6              # enthalpy at 1atm and 293.15 K

    def rho_t(self, T):
        """density at T, kg/m3"""
        return self.rho * self.Tref / T
    
    def cp_t(self, T):
        """specific capacity at T, kJ/kg-K"""
        return self.cp

    def k_t(self, T):
        """thermal conductivity at T, W/m-k"""
        return 2.64638e-3 * sqrt(T) / (1 + pow(10, -12.0 / T) * 245.4 / T)

    def mu_t(self, T):
        """laminar viscosity, kg/m-s at T"""
        if T < 100.:
            return 6.9e-6
        return 1.458e-6 * sqrt(T) / (1 + 110.4 / T)

    def nu_t(self, T):
        """moving viscosity at T, m2/s"""
        return self.mu_t(T) / self.rho_t(T)

    def Pr_t(self, T):
        """Prantl number at T"""
        return self.nu_t(T) / self.a_t(T)

    def a_t(self, T):
        """thermal diffusion coefficient at T, m2/s"""
        return self.k_t(T) / (self.rho_t(T) * self.cp_t(T))

    def rho_tp(self, T, p):
        """density at T and p"""
        return self.rho * self.Tref / (self.Pref * T)

    def h_t(self, T):
        """ enthalpy, J/kg """
        return self.h0 + self.cp*(T - self.Tref)

    def T_h(self, h):
        """ calculate temperature given enthalpy, K """
        return self.Tref + (h - self.h0)/self.cp

dry_air = DryAir()

