import numpy as np
from scipy.optimize import brentq

class Branch:

    branch_type = 'solo'

    def __init__(self, name=None, module=None):
        self.name = name
        self.id = None
        self.module = module
        self.mf = 0
        if module is None:
            self.inlet = None
            self.outlet = None
        else:
            self.inlet = module.inlet
            self.outlet = module.outlet
        self.upstream=None
        self.downstream=None
        self.is_constant = False

    def get_modules(self):
        return [self.module]

    def set_module(self, m):
        self.module = m
        self.inlet = m.inlet
        self.outlet = m.outlet

    def flow_eq(self, mf, p_up, p_down):
        return self.module.flow_eq(mf, p_up, p_down)

    def flow_eq_grad(self, mf, p_up, p_down):
        return self.module.flow_eq_grad(mf, p_up, p_down)

    def flow_var_grad(self, var, mf, p_up, p_down):
        if var:
            return self.module.flow_var_grad(var[1:], mf, p_up, p_down)
        return self.module.flow_var_grad(None, mf, p_up, p_down)

    def energy_eq(self, hup, hdown):
        return self.module.energy_eq(hup, hdown)

    def energy_eq_grad(self, hup, hdown):
        return self.module.energy_eq_grad(hup, hdown)

    def pressure_drop(self, mf, pu=None):
        self.mf = mf
        return self.module.pressure_drop(mf, pu)

    def calc_mf(self, dp, pu=None, mfmin=1e-16, mfmax=1e9):
        if abs(dp) < mfmin:
            return 0.
        m = brentq(lambda x:abs(dp)-self.pressure_drop(x, pu), mfmin, mfmax)
        self.mf = m*dp/abs(dp)
        return self.mf

    def reverse(self):
        self.upstream, self.downstream = self.downstream, self.upstream
        if self.upstream:
            self.upstream.upstream_knots.pop(self)
            self.upstream.downstream_knots[self] = self.inlet
        if self.downstream:
            self.downstream.downstream_knots.pop(self)
            self.downstream.upstream_knots[self] = self.outlet

    def update_flow(self):
        self.module.inlet.mf = self.mf
        self.module.outlet.mf = self.mf
        self.module.update_flow()

    def update_energy(self):
        self.module.update_energy()

    def energy_var_grad(self, var, h_up):
        return self.module.energy_var_grad(var[1:], h_up)

    def check_flow_ballance(self, err=1e-10):
        return abs(self.upstream.p - self.downstream.p - self.pressure_drop(self.mf)) < err

    def flow_ballance_error(self):
        return abs(self.upstream.p - self.downstream.p - self.pressure_drop(self.mf))


class HexBranch(Branch):

    branch_type = 'heatexchanger'

    def __init__(self, name=None, module=None):
        super().__init__(name, module)
        self.heat_transfer = None       # branch to heat exchange

    def energy_coef(self, h_up, h_o):
        return self.module.energy_coef(h_up, h_o)


class SerialBranch:

    branch_type = 'serial'

    def __init__(self, name=None):
        self.name = name
        self.id = None
        self.modules = []
        self.mf = 0
        self.inlet = None
        self.outlet = None
        self.upstream=None
        self.downstream=None
        self.is_constant = False

    def get_modules(self):
        mm = []
        for m in self.modules:
            if hasattr(m, 'branch_type'):
                mm += m.get_modules()
            else:
                mm.append(m)
        return mm

    def add_module(self, m):
        self.modules.append(m)
        self.outlet = m.outlet

    def pressure_drop(self, mf, pu=None):
        self.mf = mf
        if pu is None:
            return sum([m.pressure_drop(mf, None) for m in self.modules])
        else:
            dp = 0.
            for m in self.modules:
                dp += m.pressure_drop(mf, pu - dp)
            return dp

    def flow_eq(self, mf, pu, pd):
        dp = 0.
        for m in self.modules:
            dp += m.pressure_drop(mf)
        return dp - pu - dp

    def flow_eq_grad(self, mf, pu, pd):
        assert len(self.modules) > 0, "modules can not be empty"
        a = 0.
        f = 0.
        for m in self.modules:
            cm, _, _, s = m.flow_eq_grad(mf, 0, 0)
            a += cm
            f += s
        return a, -1., 1., f - pu + pd

    def flow_var_grad(self, var, mf, pu, pd):
        a = 0.
        cs = 0.
        for m in self.modules:
            a += m.flow_var_grad(None, mf, 0., 0.)[0]
        if var:
            cs = var[0].flow_var_grad(var[1:], mf, 0., 0.)[-1]
        return a, -1., 1., cs

    def energy_eq(self, h_up, h_down):
        hu = h_up
        for m in self.modules:
            hu = -m.energy_eq(hu, 0)
        return h_down - hu

    def energy_eq_grad(self, h_up, h_down):
        return -1., 1., self.energy_eq(h_up, h_down)

    def calc_mf(self, dp, pu=None, mfmin=1e-16, mfmax=1e9):
        if abs(dp) < 1e-32:
            return 0.
        m = brentq(lambda x:abs(dp)-self.pressure_drop(x, pu), mfmin, mfmax)
        self.mf = m*dp/abs(dp)
        return self.mf

    def set_modules_p(self, pu, mf=None):
        if mf is None:
            mf = self.mf
        for m in self.modules:
            m.inlet.p = pu
            pu -= m.pressure_drop(mf, pu)
            m.outlet.p = pu

    def update_flow(self):
        self.set_modules_p(self.inlet.p)
        for m in self.modules:
            m.inlet.mf = self.mf
            m.outlet.mf = self.mf
            m.update_flow()

    def update_energy(self):
        for m in self.modules:
            m.update_energy()

    def connect_ports(self):
        for i in range(len(self.modules) - 1):
            self.modules[i + 1].inlet = self.modules[i].outlet
        self.inlet = self.modules[0].inlet
        self.outlet = self.modules[-1].outlet

    def reverse(self):
        self.upstream, self.downstream = self.downstream, self.upstream
        self.modules.reverse()
        for m in self.modules:
            m.inlet, m.outlet = m.outlet, m.inlet
        self.inlet = self.modules[0].inlet
        self.outlet = self.modules[-1].outlet  
        if self.upstream:
            self.upstream.upstream_knots.pop(self)
            self.upstream.downstream_knots[self] = self.inlet
        if self.downstream:
            self.downstream.downstream_knots.pop(self)
            self.downstream.upstream_knots[self] = self.outlet

    def energy_var_grad(self, var, h_up, h_down):
        m0 = var[0]
        return m0.energy_var_grad(var[1:], h_up, h_down)

    def check_flow_ballance(self, err=1e-10):
        return abs(self.upstream.p - self.downstream.p - self.pressure_drop(self.mf)) < err

    def flow_ballance_error(self):
        return abs(self.upstream.p - self.downstream.p - self.pressure_drop(self.mf))


class ParallelBranch:

    branch_type = 'parallel'

    def __init__(self, name, inlet=None, outlet=None):
        self.name = name
        self.id = None
        self.branches = []
        self.inlet = inlet
        self.outlet = outlet
        self.bmf = []
        self.mf = 0.
        self.upstream=None
        self.downstream=None
        self.is_constant = False

    def get_modules(self):
        mm = []
        for b in self.branches:
            if hasattr(b, 'branch_type'):
                mm += b.get_modules()
            else:
                mm.append(b)
        return mm

    def add_branch(self, b):
        self.branches.append(b)
        self.bmf.append(0.)

    def update_flow(self):
        self.bmf = self._solve_mfx_dp(self.mf)[1]
        for i in range(len(self.branches)):
            self.branches[i].inlet.p = self.inlet.p
            self.branches[i].outlet.p = self.outlet.p
            self.branches[i].mf = self.bmf[i]
            self.branches[i].update_flow()
        self.inlet.mf = self.mf
        self.outlet.mf = self.mf

    def update_energy(self):
        for b in self.branches:
            b.inlet.h = self.inlet.h
            b.outlet.h = self.outlet.h
            b.update_energy()

    def outlet_h(self):
        return sum([m*b.outlet.h for m, b in zip(self.bmf, self.branches)])/self.mf

    def _solve_mfx_dp(self, mf):
        n = len(self.branches)
        X = np.ones(n)*(mf/n)
        y = self.branches[0].pressure_drop(mf)
        while True:
            z = [self.branches[i].pressure_drop(X[i]) for i in range(n)]
            d = [self.branches[i].flow_eq_grad(X[i], self.inlet.p, self.outlet.p)[0] for i in range(n)]
            dy = -(y - z[-1] + d[-1]*sum([(y-z[i])/d[i] for i in range(n-1)])) \
                    /(1. + d[-1]*sum([1./d[i] for i in range(n-1)]))
            y += dy
            X[:-1] += [(y - z[i])/d[i] for i in range(n-1)]
            X[-1] = mf - sum(X[:-1])
            if abs(dy) < 1e-9:
                break
        return y, X

    def pressure_drop(self, mf, pu=None):
        return self._solve_mfx_dp(mf)[0]

    def calc_mf(self, dp, pu, mfmin=1e-8, mfmax=1e5):
        self.bmf = [b.calc_mf(dp, pu, mfmin, mfmax) for b in self.branches]
        self.mf = sum(self.bmf)
        return self.mf

    def flow_eq(self, mf, pu, pd):
        assert len(self.branches) > 0, "branches can not be empty"
        return self.pressure_drop(mf) - pu + pd

    def flow_eq_grad(self, mf, pu, pd):
        n = len(self.branches)
        X = np.ones(n)*(mf/n)
        y = self.branches[0].pressure_drop(mf)
        while True:
            z = [self.branches[i].pressure_drop(X[i]) for i in range(n)]
            d = [self.branches[i].flow_eq_grad(X[i], pu, pd)[0] for i in range(n)]
            dy = -(y - z[-1] + d[-1]*sum([(y-z[i])/d[i] for i in range(n-1)])) \
                    /(1. + d[-1]*sum([1./d[i] for i in range(n-1)]))
            y += dy
            X[:-1] += [(y - z[i])/d[i] for i in range(n-1)]
            X[-1] = mf - sum(X[:-1])
            if abs(dy) < 1e-9:
                break
        dm = 1./sum([1./d[i] for i in range(n)])
        return dm, -1., 1., y - pu + pd        

    def flow_var_grad(self, var, mf, pu, pd):
        pu = self.inlet.p
        pd = self.outlet.p
        d = [b.flow_var_grad(None, b.mf, pu, pd)[0] for b in self.branches]
        cm = 1. / sum([1./x for x in d])
        cs = 0.
        if var:
            b = var[0]
            bd, _, _, bcs = b.flow_var_grad(var[1:], b.mf, pu, pd)
            cs = d[-1] / bd * bcs / (d[-1]*sum([1./x for x in d[:-1]]) + 1.)
        return cm, -1., 1., cs

    def energy_eq(self, h_up, h_down):
        S = 0.
        for b in self.branches:
            S += b.mf * b.energy_eq(h_up, 0.)
        return h_down + S/self.mf

    def energy_eq_grad(self, h_up, h_down):
        cu, cs = 0., 0.
        for b in self.branches:
            u, _, s = b.energy_eq_grad(h_up, h_down)
            cu += u*b.mf
            cs += s*b.mf
        return cu/self.mf, 1., cs/self.mf

    def set_modules_p(self, pu, mf=None):
        if mf is None:
            mf = self.mf
        pd = self.pressure_drop(mf, pu)
        for b in self.branches:
            b.inlet.p = pu
            b.outlet.p = pd
            b.update_flow()

    def reverse(self):
        self.upstream, self.downstream = self.downstream, self.upstream
        for b in self.branches:
            b.reverse()
        if self.upstream:
            self.upstream.upstream_knots.pop(self)
            self.upstream.downstream_knots[self] = self.inlet
        if self.downstream:
            self.downstream.downstream_knots.pop(self)
            self.downstream.upstream_knots[self] = self.outlet

    def energy_var_grad(self, var, h_up, h_down):
        b0 = var[0]
        return b0.mf * b0.energy_coef_grad(var[1:], h_up, h_down)/self.mf

    def check_flow_ballance(self, err=1e-10):
        return abs(self.upstream.p - self.downstream.p - self.pressure_drop(self.mf)) < err

    def flow_ballance_error(self):
        return abs(self.upstream.p - self.downstream.p - self.pressure_drop(self.mf))
