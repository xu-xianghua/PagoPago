
from pago.solver.branch import ParallelBranch, SerialBranch, Branch
from pago.solver.hub import Hub
from pago.solver.solver import Solver
from pago.solver.fluidknot import FluidKnot
from pago.spmodule.pipe import Pipe
from pago.property.dry_air import dry_air

def test_solver():
    # create modules
    pp1 = Pipe('pp1',L=2.0,D=0.01,LeD=3.,fluid=dry_air)
    pp2 = Pipe('pp2',L=2.0,D=0.01,LeD=3.,fluid=dry_air)
    pp3a = Pipe('pp3a',L=4.0,D=0.01,LeD=60.,fluid=dry_air)
    pp3b = Pipe('pp3b',L=3.0,D=0.01,LeD=30.,fluid=dry_air)
    pp3c = Pipe('pp3c',L=1.0,D=0.01,LeD=30.,fluid=dry_air)
    pp4 = Pipe('pp4',L=1.0,D=0.01,LeD=1000.,fluid=dry_air)
    pp5 = Pipe('pp5',L=1.0,D=0.01,LeD=800.,fluid=dry_air)

    # create hub and branch
    h0 = Hub('h0')
    h1 = Hub('h1')
    h2 = Hub('h2')
    h3 = Hub('h3')
    h0.is_constant = True
    h3.is_constant = True
    b1 = Branch('b1', pp1)
    b2 = Branch('b2', pp2)
    b4 = Branch('b4', pp4)
    b5 = Branch('b5', pp5)
    b3a = Branch('b3a', pp3a)
    b3b = SerialBranch('b3b')
    b3b.modules = [pp3b, pp3c]
    b3 = ParallelBranch('b3')
    b3.add_branch(b3a)
    b3.add_branch(b3b)
    b3.inlet = FluidKnot(fluid = dry_air)
    b3.outlet = FluidKnot(fluid = dry_air)
    pp3c.inlet = pp3b.outlet
    # connect branches and hubs
    h0.add_branch(b1, upstream=False)
    h0.add_branch(b2, upstream=False)
    h1.add_branch(b1, upstream=True)
    h1.add_branch(b3, upstream=True)
    h1.add_branch(b4, upstream=False)
    h2.add_branch(b2, upstream=True)
    h2.add_branch(b3, upstream=False)
    h2.add_branch(b5, upstream=False)
    h3.add_branch(b4, upstream=True)
    h3.add_branch(b5, upstream=True)
    # construct solver
    solver = Solver()
    solver.method = 'direct'
    solver.hubs = [h1,h2]
    solver.branches = [b1,b2,b3,b4,b5]
    solver.boundary_hubs = [h0,h3]
    # initialize
    h0.p = 1e6
    h3.p = 1e5
    T0 = 300.
    pp1.initialize(T0)
    pp2.initialize(T0)
    pp3a.initialize(T0)
    pp3b.initialize(T0)
    pp3c.initialize(T0)
    pp4.initialize(T0)
    pp5.initialize(T0)

    if solver.check_network():
        print('initial incident matrix:')
        solver.print_incident_matrix()
        solver.solve_flow(tol=1e-9)
        solver.update_flow()
        print('update incident matrix:')
        solver.print_incident_matrix()
        print('check flow ballance: {:}'.format(solver.check_flow_ballance(err=1e-6)))
        print('calculation results:')
        for b in solver.branches:
            print('branch {0:} mf: {1:f}'.format(b.name, b.mf))
        for h in solver.hubs:
            print('hub {0:} p: {1:.2f}'.format(h.name, h.p))
        for h in solver.boundary_hubs:
            print('hub {0:} p: {1:.2f}'.format(h.name, h.p))
        print('in branch b3:')
        for b in b3.branches:
            print('module {} mf: {:f}'.format(b.name, b.mf))
        for m in b3b.modules:
            print('module {} inlet pressure: {:.2f}'.format(m.name, m.inlet.p))

    return solver
                   
if __name__ == '__main__':
    test_solver()
    
    
    
