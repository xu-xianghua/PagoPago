from math import pi
from autograd import grad
from ..property.dry_air import dry_air
from ..formula.convection_formula import friction_factor_pipe_flow
from ..solver.fluidknot import FluidKnot
    

class Pipe:

    serialable = True
    parallelable = True

    def __init__(self, name, L = 1.0, D = 0.01, LeD = 0., fluid = None):
        self.name = name
        self.L = L
        self.D = D
        self.LeD = LeD
        if fluid is None:
            self.fluid = dry_air
        else:
            self.fluid = fluid
        self.inlet = FluidKnot(fluid=self.fluid, downstream=self)
        self.outlet = FluidKnot(fluid=self.fluid, upstream=self)
        self.outlet.set_v(*self.inlet.get_v())
        self.initialize(280.)
        
    def initialize(self, T0):
        h0 = self.fluid.h_t(T0)
        self.inlet.h = h0
        self.outlet.h = h0
        self.flow_sa = pi*self.D**2/4      
      
    def set_inlet(self, mf, pin, Tin):
        self.inlet.set_v(mf, pin, self.fluid.h_t(Tin))
        return True

    def get_outlet(self):
        return self.outlet.get_v()

    def outlet_T(self):
        return self.fluid.T_h(self.outlet.h)

    def flow_eq(self, mf, pu, pd):
        T = self.fluid.T_h(self.inlet.h)
        rho = self.fluid.rho_t(T)
        mu = self.fluid.mu_t(T)
        u = mf / rho / self.flow_sa
        Re = rho * u * self.D / mu
        dp = friction_factor_pipe_flow(Re) * (self.L / self.D + self.LeD) * rho * u**2 / 2.
        return dp - pu + pd

    def flow_eq_grad1(self, mf, pu, pd):
        return grad(self.flow_eq, 0)(mf, pu, pd), -1., 1., self.flow_eq(mf, pu, pd)

    def flow_eq_grad(self, mf, pu, pd):
        T = self.fluid.T_h(self.inlet.h)
        rho = self.fluid.rho_t(T)
        mu = self.fluid.mu_t(T)
        u = mf / rho / self.flow_sa
        Re = rho * u * self.D / mu
        flu = friction_factor_pipe_flow(Re) * (self.L / self.D + self.LeD) * u
        return flu / self.flow_sa, -1., 1., flu * rho * u / 2. - pu + pd

    def pressure_drop(self, mf=None, pu=None):
        if mf is None:
            mf = self.inlet.mf
        T = self.fluid.T_h(self.inlet.h)
        rho = self.fluid.rho_t(T)
        mu = self.fluid.mu_t(T)
        u = mf / rho / self.flow_sa
        Re = rho * u * self.D / mu
        return friction_factor_pipe_flow(Re) * (self.L / self.D + self.LeD) * rho * u**2 / 2.
        
    def dissipation_heat(self, mf=None, T=None):
        if mf is None:
            mf = self.inlet.mf
        if T is None:
            T = self.fluid.T_h(self.inlet.h)
        rho = self.fluid.rho_t(T)
        mu = self.fluid.mu_t(T)
        u = mf / rho / self.flow_sa
        Re = rho * u * self.D / mu
        return mf * friction_factor_pipe_flow(Re) * (self.L / self.D + self.LeD) * u**2 / 2.
   
    def energy_eq(self, hup, hdown):
        T = self.fluid.T_h(hup)
        rho = self.fluid.rho_t(T)
        mu = self.fluid.mu_t(T)
        u = self.inlet.mf / rho / self.flow_sa
        Re = rho * u * self.D / mu
        s = friction_factor_pipe_flow(Re) * (self.L / self.D + self.LeD) * u**2 / 2.
        return hdown - hup - s

    def energy_eq_grad(self, hup, hdown):
        T = self.fluid.T_h(hup)
        rho = self.fluid.rho_t(T)
        mu = self.fluid.mu_t(T)
        u = self.inlet.mf / rho / self.flow_sa
        Re = rho * u * self.D / mu
        s = friction_factor_pipe_flow(Re) * (self.L / self.D + self.LeD) * u**2 / 2.
        return -1., 1., hdown - hup - s

    def flow_var_grad(self, var, mf, pu=None, pd=None):
        if type(var) is list:
            var = var[0]
        T = self.fluid.T_h(self.inlet.h)
        rho = self.fluid.rho_t(T)
        mu = self.fluid.mu_t(T)
        u = mf / rho / self.flow_sa
        Re = rho * u * self.D / mu
        f = friction_factor_pipe_flow(Re)
        dp = f * (self.L / self.D + self.LeD) * rho * u**2 / 2.
        _grad_Re = grad(friction_factor_pipe_flow, 0)
        cm = dp * (_grad_Re(Re) * Re + 2. * f) / mf
        cu = -1.
        cd = 1.
        cs = 0.
        if var is 'D':
            cs = mf/(rho*self.flow_sa**2)/2.*(f*mf*self.L/self.D**2 - (self.L/self.D + self.LeD)*Re*_grad_Re(Re))
        if var is 'L':
            cs = -f * mf**2 / (rho * self.D * self.flow_sa**2) / 2.
        if var is 'LeD':
            cs = -f * mf**2 / (rho * self.flow_sa**2) / 2.
        return cm, cu, cd, cs

    def energy_var_grad(self, var, hup=None, hdown=None):
        if hup is None:
            hup = self.inlet.h
        T = self.fluid.T_h(hup)
        rho = self.fluid.rho_t(T)
        mu = self.fluid.mu_t(T)
        u = self.inlet.mf / rho / self.flow_sa
        Re = rho * u * self.D / mu
        f = friction_factor_pipe_flow(Re)
        if var is 'D':
            _grad = grad(friction_factor_pipe_flow, 0)
            return -u**2 /2./self.D**2 *(f*self.L  - (self.L + self.D*self.LeD)*_grad(Re)*Re)
        if var is 'L':
            return f / self.D * u**2 / 2.
        if var is 'LeD':
            return f * u**2 / 2.
        if var is 'mf':
            _grad = grad(friction_factor_pipe_flow, 0)
            return self.inlet.mf*(self.L/self.D + self.LeD)*(f + Re*_grad(Re)/2.)/(rho**2*self.flow_sa**2)
        return 0.

    def update_flow(self):
        pass

    def update_energy(self):
        self.outlet.h = self.inlet.h + self.dissipation_heat()/self.inlet.mf
    
