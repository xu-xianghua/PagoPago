
from namua.solver.branch import SerialBranch, ParallelBranch, Branch
from namua.solver.hub import Hub
from namua.solver.fluidknot import FluidKnot
from namua.solver.solver import Solver
from namua.spmodule.pipe import Pipe
from namua.property.dry_air import dry_air
from namua.property.liquid_water import liquid_water

#gz = liquid_water
gz = dry_air

def test_solver(output = False):
    # create modules
    pp1 = Pipe('pp1',L=2.0,D=0.01,LeD=3.,fluid=gz)
    pp2 = Pipe('pp2',L=2.0,D=0.01,LeD=3.,fluid=gz)
    pp3a = Pipe('pp3a',L=4.0,D=0.01,LeD=60.,fluid=gz)
    pp3b = Pipe('pp3b',L=3.0,D=0.01,LeD=30.,fluid=gz)
    pp3c = Pipe('pp3c',L=1.0,D=0.01,LeD=30.,fluid=gz)
    pp4 = Pipe('pp4',L=1.0,D=0.01,LeD=1000.,fluid=gz)
    pp5 = Pipe('pp5',L=1.0,D=0.01,LeD=800.,fluid=gz)

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
    b3.inlet = FluidKnot(fluid = gz)
    b3.outlet = FluidKnot(fluid = gz)
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
    h0.set_p(2e5)
    h0.set_h(gz.h_t(280.))
    h3.set_p(1e5)
    h3.set_h(gz.h_t(350.))

    if solver.check_network():
        x0 = sum([h1.h, h2.h])
        solver.solve_energy(solveflow=True, initialize=True, relax=0.5)
        solver.update_energy()
        while True:
            solver.solve_energy(solveflow=True, initialize=False, relax=0.5)
            solver.update_energy()
            x = sum([h1.h, h2.h])
            if abs(x - x0) < 1e-12:
                print('converged')
                break
            x0 = x
        if output:
            print('check flow ballance: {:}'.format(solver.check_flow_ballance(err=1e-6)))
            print('calculation results:')
            for b in solver.branches:
                print('branch {0:} mf: {1:f}, Tout: {2:.2f}'.format(b.name, b.mf, b.outlet.get_T()))
            for h in solver.hubs:
                print('hub {0:} p: {1:.2f}, h: {2:.3f}'.format(h.name, h.p, h.h))
            for h in solver.boundary_hubs:
                print('hub {0:} p: {1:.2f}, h: {2:.3f}'.format(h.name, h.p, h.h))
            print('in branch b3:')
            for b in b3.branches:
                print('module {} mf: {:f}, Tout: {:.2f}'.format(b.name, b.mf, b.outlet.get_T()))
            for m in b3b.modules:
                print('module {} Pin: {:.2f}, Tin: {:.2f}, Tout: {:.2f}'.format(m.name, m.inlet.p, m.inlet.get_T(), m.outlet.get_T()))

    return solver
                   
if __name__ == '__main__':
    test_solver(True)
    
