""" liquid_water.py
thermal physical properties of liquid water
"""
from numpy import interp

RHO_T = [999.84325, 999.89835, 999.57534, 998.92398, 997.98273, 996.78205,
       995.34647, 993.69607, 991.84751, 989.81476, 987.60963, 985.2422 ,
       982.72113, 980.05386, 977.24686, 974.30572, 971.23532, 968.03987,
       964.72305, 961.28799, 957.73741, 954.07359, 950.29845, 946.41355,
       942.42012, 938.3191 ]
H_T = [-80369.9681 , -59320.33554, -38327.18735, -17372.50715,
         3556.57763,  24469.44373,  45373.02357,  66272.52176,
        87171.92678, 108074.38717, 128982.49194, 149898.48044,
       170824.39858, 191762.21396, 212713.89949, 233681.49285,
       254667.1376 , 275673.11076, 296701.84039, 317755.91613,
       338838.09477, 359951.30291, 381098.63758, 402283.36622,
       423508.92652, 444778.92669]
CP_T = [4217.08528, 4203.50566, 4194.19762, 4187.90414, 4183.78652,
       4181.26387, 4179.9261 , 4179.48281, 4179.73057, 4180.53052,
       4181.79204, 4183.4606 , 4185.50838, 4187.92715, 4190.72275,
       4193.91097, 4197.51435, 4201.55999, 4206.07783, 4211.09954,
       4216.6578 , 4222.78587, 4229.51735, 4236.88622, 4244.92691,
       4253.67448]
K_T = [0.55773, 0.56967, 0.58048, 0.59036, 0.59945, 0.60784, 0.61562,
       0.62284, 0.62954, 0.63576, 0.64152, 0.64685, 0.65176, 0.65627,
       0.66039, 0.66414, 0.66751, 0.67053, 0.6732 , 0.67553, 0.67753,
       0.6792 , 0.68056, 0.6816 , 0.68235, 0.6828 ]
MU_T = [1.7401e-03, 1.4785e-03, 1.2746e-03, 1.1125e-03, 9.8111e-04, 8.7306e-04, 7.8300e-04,
        7.0708e-04, 6.4242e-04, 5.8688e-04, 5.3879e-04, 4.9687e-04, 4.6009e-04, 4.2764e-04, 
        3.9887e-04, 3.7324e-04, 3.5031e-04, 3.2971e-04, 3.1114e-04, 2.9434e-04, 2.7909e-04, 
        2.6521e-04, 2.5254e-04, 2.4093e-04, 2.3029e-04, 2.2049e-04]
TT = [274., 279., 284., 289., 294., 299., 304., 309., 314., 319., 324.,
       329., 334., 339., 344., 349., 354., 359., 364., 369., 374., 379.,
       384., 389., 394., 399.]


class LiquidWater:
    pc = 22.06e6
    Tc = 647.096 
    mw = 18.015                # mol weight, kg/kmol
    Rg = 461.                 # gas constant, J/Kkg
    def __init__(self):
        self.Tref = 293.15
        self.Pref = 101325.0
        self.h0 = 83914.14495
        # properties at temperature 20C and 1atm
        self.rho = 998.2                # density, kg/m3
        self.cp = 4183.0                # specific capacity, kJ/kg-K
        self.mu = 1.004e-3              # laminar viscosity, kg/m-s
        self.nu = 1.006e-6              # moving viscosity, m2/s
        self.Pr = 7.02                  # Prantl number
        self.k = 0.599                  # thermal conductivity, W/m-K
        self.a = 14.3e-8                # thermal diffusion coefficient 
        self.T0 = 274.
        self.T1 = 399.
        self.dT = 5.

    def _interp(self, T, pd):
        """ interpolate 1D """
        if T <= self.T0:
            return pd[0]
        if T >= self.T1:
            return pd[-1]
        x = (T-self.T0)/self.dT
        p = int(x)
        x -= p
        return x*pd[p+1] + (1-x)*pd[p]

    def rho_t(self, T):
        return self._interp(T, RHO_T)

    def cp_t(self, T):
        return self._interp(T, CP_T)    

    def h_t(self, T):
        return self.h0 + self._interp(T, H_T)    

    def k_t(self, T):
        return self._interp(T, K_T)    

    def mu_t(self, T):
        return self._interp(T, MU_T)    

    def nu_t(self, T):
        return self._interp(T, MU_T) / self.rho_t(T)

    def Pr_t(self, T):
        return self.mu_t(T) * self.cp_t(T) / self.k_t(T)    

    def a_t(self, T):
        return self.k_t(T) / (self.rho_t(T) * self.cp_t(T))

    def T_h(self, h):
        h -= self.h0
        return interp(h, H_T, TT)

liquid_water = LiquidWater()