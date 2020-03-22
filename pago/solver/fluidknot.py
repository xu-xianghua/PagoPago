  
class FluidKnot:
    def __init__(self, fluid=None, upstream=None, downstream=None):
        self.mf = 0.001
        self.p = 0.
        self.h = 0.
        self.fluid = fluid
        self.upstream = upstream
        self.downstream = downstream
        self.index = 0
        self.given_mf = False
        self.given_p = False
        self.is_two_phase = False

    def initialize(self, mf, p, h):
        self.mf = mf
        self.p = p
        self.h = h

    def get_hf(self):
        return self.mf*self.h

    def get_rho(self):
        if not self.is_two_phase:
            T = self.fluid.T_h(self.h)
            return self.fluid.rho_t(T)
        else:
            return self.fluid.rho_ph(self.p, self.h)

    def get_vf(self):
        return self.mf/self.get_rho()

    def set_vf(self, vf):
        self.mf = vf*self.get_rho()

    def set_mf(self, mf):
        self.mf = mf

    def set_p(self, pin):
        self.p = pin

    def set_h(self, h):
        self.h = h

    def set_T(self, T, p=None):
        if p is None:
            p = self.p
        if not self.is_two_phase:
            self.h = self.fluid.h_t(T)
        else:
            self.h = self.fluid.h_tp(T, p)

    def set_v(self, mf, p, h):
        self.mf = mf
        self.p = p
        self.h = h
        return True

    def get_v(self):
        return self.mf, self.p, self.h

    def get_T(self):
        if not self.is_two_phase:
            return self.fluid.T_h(self.h)
        return self.fluid.T_ph(self.p, self.h)

    def get_x(self):
        if not self.is_two_phase:
            return 1.
        return (self.h - self.fluid.h_sl(self.p))/(self.fluid.h_sg(self.p) - self.fluid.h_sl(self.p))

