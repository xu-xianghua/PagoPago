
from pago.solver.branch import Branch
from pago.solver.hub import Hub
from pago.solver.solver import Solver
from pago.spmodule.pipe import Pipe
from pago.property.dry_air import dry_air

def test_solver(solve=True):
    # create modules
    pp1 = Pipe('pp1',L=2.0,D=0.01,LeD=300.,fluid=dry_air)
    pp2 = Pipe('pp2',L=2.0,D=0.01,LeD=500.,fluid=dry_air)
    pp3 = Pipe('pp3',L=2.0,D=0.01,LeD=30.,fluid=dry_air)
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
    b3 = Branch('b3', pp3)
    b4 = Branch('b4', pp4)
    b5 = Branch('b5', pp5)
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
    pp3.initialize(T0)
    pp4.initialize(T0)
    pp5.initialize(T0)

    m4 = 0.025
    m5 = 0.02
    if solve and solver.check_network():
        print('solve grad control to pp1.LeD and pp2.LeD')
        var1 = [b4, pp4, 'LeD']
        var2 = [b5, pp5, 'LeD']
        while True:
            solver.solve_flow(tol=1e-9, relax=1.)
            solver.update_flow()
            solver.solve_flow_grad(var1, False)
            a = solver.dX[b4.id] 
            c = solver.dX[b5.id]
            solver.solve_flow_grad(var2, False)
            b = solver.dX[b4.id] 
            d = solver.dX[b5.id]
            y1 = m4 - b4.mf
            y2 = m5 - b5.mf
            dx1 = (d*y1 - b*y2)/(a*d - b*c)
            dx2 = (a*y2 - c*y1)/(a*d - b*c)
            if abs(dx1) > 0.3*pp1.LeD:
                dx1 = 0.3*pp1.LeD*dx1/abs(dx1)
            if abs(dx2) > 0.3*pp2.LeD:
                dx2 = 0.3*pp2.LeD*dx2/abs(dx2)

            pp4.LeD += dx1
            pp5.LeD += dx2
            if abs(dx1) + abs(dx2) < 1e-8:
                print('finished, pp4.LeD={:.3f}, pp4.LeD={:.3f}, error={:e},{:e}'.format(pp4.LeD, pp5.LeD, b4.mf-m4, b5.mf-m5))
                break
            print('{:.3f}, {:.3f}'.format(pp4.LeD, pp5.LeD))
        print(b4.mf, b5.mf)
        print('{:.3f}, {:.3f}'.format(pp4.LeD, pp5.LeD))
        solver.solve_flow(tol=1e-9, relax=1.)
        solver.update_flow()
        print('check flow ballance: {:}'.format(solver.check_flow_ballance(err=1e-6)))
        print('calculation results:')
        for b in solver.branches:
            print('branch {0:} mf: {1:f}'.format(b.name, b.mf))
        for h in solver.hubs:
            print('hub {0:} p: {1:.2f}'.format(h.name, h.p))

    return solver
                   
if __name__ == '__main__':
    test_solver()
    
    
    
