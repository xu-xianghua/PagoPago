
class Hub:

    def __init__(self, name):
        self.name = name
        self.id = None
        self.p = 0.
        self.h = 0.
        self.branches = []
        self.upstream_knots = {}
        self.downstream_knots = {}
        self.is_constant = False

    def add_branch(self, b, upstream=True):
        self.branches.append(b)
        if upstream:
            self.upstream_knots[b] = b.outlet
            b.downstream = self
        else:
            self.downstream_knots[b] = b.inlet
            b.upstream = self

    def check_connection(self):
        for b in self.branches:
            if b in self.upstream_knots:
                if b in self.downstream_knots:
                    print('branch {0}: is both upstream and downsream of hub {1}'.format(b.name, self.name))
                    return False
                if self.upstream_knots[b] is not b.outlet:
                    print('branch {0}: outlet is not in upstream of hub {1}'.format(b.name, self.name))
                    return False
            if not b in self.upstream_knots:
                if not b in self.downstream_knots:
                    print('branch {0}: is neither upstream or downsream of hub {1}'.format(b.name, self.name))
                    return False
                if self.downstream_knots[b] is not b.inlet:
                    print('branch {0}: inlet is not in upstream of hub {1}'.format(b.name, self.name))
                    return False
        if len(self.branches) != len(self.upstream_knots) + len(self.downstream_knots):
            print('hub {0}: branch number is less than knot number'.format(self.name))
            return False
        return True

    def total_flux(self):
        return sum([k.mf for k in self.upstream_knots.values()])

    def check_flow_ballance(self, err=1e-10):
        m_in = sum([k.mf for k in self.upstream_knots.values()])
        m_out = sum([k.mf for k in self.downstream_knots.values()])
        return abs(m_in - m_out) < err

    def flow_ballance_error(self):
        m_in = sum([k.mf for k in self.upstream_knots.values()])
        m_out = sum([k.mf for k in self.downstream_knots.values()])
        return abs(m_in - m_out)

    def calc_h(self):
        return sum([k.get_hf() for k in self.upstream_knots.values()])/self.total_flux()

    def update_flow(self):
        for k in self.upstream_knots.values():
            k.p = self.p
        for k in self.downstream_knots.values():
            k.p = self.p

    def update_energy(self):
        for k in self.downstream_knots.values():
            k.h = self.h

    def set_h(self, h):
        self.h = h
        self.update_energy()
    
    def set_p(self, p):
        self.p = p
        self.update_flow()

    def initialize(self, p, h):
        self.set_p(p)
        self.set_h(h)
