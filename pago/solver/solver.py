import numpy as np
import random
from scipy import sparse

class Solver:

    def __init__(self):
        self.branches = []
        self.hubs = []
        self.const_branches = []          # constant flow branch
        self.boundary_hubs = []           # boundary hub, with given pressure
        self.X = None
        self.jacobian = None
        self.rhs = None
        self.v_ind = None
        self.H = None
        self.Hjacobian = None
        self.Hrhs = None
        self.Hv_ind = None

        self.bh_ind = None
        self.method = 'bicg'              # solve method for linear equations
        
    def get_all_modules(self):
        mm = []
        for b in self.branches + self.const_branches:
            mm += b.get_modules()
        return mm

    def incident_matrix(self):
        n = len(self.branches)
        n1 = len(self.const_branches)
        m = len(self.hubs)
        m1 = len(self.boundary_hubs)
        self.set_id()
        im = np.zeros((n+n1, m+m1), np.int)
        for b in self.branches:
            im[b.id, b.upstream.id] = -1
            im[b.id, b.downstream.id] = 1
        for b in self.const_branches:
            im[b.id, b.upstream.id] = -1
            im[b.id, b.downstream.id] = 1
        return im

    def print_incident_matrix(self, colw=6):
        im = self.incident_matrix()
        n, m = im.shape
        strformat = '{:>%ds} |'%colw
        numformat = '{:>%dd} |'%colw
        print(strformat.format(' '), end='')
        for h in self.hubs + self.boundary_hubs:
            print(strformat.format(h.name), end='')
        print('')
        for i,b in zip(range(n), self.branches + self.const_branches):
            print(strformat.format(b.name), end='')
            for j in range(m):
                print(numformat.format(im[i,j]), end='')
            print('')

    def check_network(self):
        self.set_id()
        state = True
        for b in self.branches:
            if b.is_constant:
                state = False
                print('branch {:} is constant'.format(b.name))
            if b.upstream is None or b.downstream is None:
                state = False
                print('branch {:} lacks upstream or downstream'.format(b.name))
                # more check for the hubs
        for b in self.const_branches:
            if not b.is_constant:
                state = False
                print('branch {:} is not constant flow'.format(b.name))
            if b.upstream is None or b.downstream is None:
                state = False
                print('const branch {:} lacks upstream or downstream'.format(b.name))
                # more check for the hubs
        for h in self.hubs:
            if h.is_constant:
                state = False
                print('hub {:} is boundary'.format(h.name))
            if len(h.branches) == 0:
                state = False
                print('hub {:} is isolated'.format(h.name))
        if len(self.boundary_hubs) < 1:
            state = False
            print('no boundary hub')
        for h in self.boundary_hubs:
            if not h.is_constant:
                state = False
                print('boundary hub {:} is not boundary'.format(h.name))
            if len(h.branches) == 0:
                state = False
                print('hub {:} is isolated'.format(h.name))
                
        return state
        
    def set_id(self):
        branch = self.branches + self.const_branches
        hub = self.hubs + self.boundary_hubs
        for i, b in zip(range(len(branch)), branch):
            b.id = i
        for i, h in zip(range(len(hub)), hub):
            h.id = i

    def _init_Jacobian(self):
        n = len(self.branches)
        m = len(self.hubs)
        I = []
        J = []
        V = []
        self.v_ind = np.ones((n+m, 3), np.int)*-1
        for b in self.branches:
            I.append(b.id)
            J.append(b.id)
            V.append(0.)
            self.v_ind[b.id, 0] = len(V) - 1
            if not b.upstream.is_constant:
                I.append(b.id)
                J.append(n+b.upstream.id)
                V.append(-1.)
                self.v_ind[b.id, 1] = len(V) - 1
            if not b.downstream.is_constant:
                I.append(b.id)
                J.append(n+b.downstream.id)
                V.append(1.)
                self.v_ind[b.id, 2] = len(V) - 1
                if not b.upstream.is_constant and b.upstream.id > b.downstream.id:
                    self.v_ind[b.id, 2] -= 1
                    self.v_ind[b.id, 1] += 1
            
        for h in self.hubs:
            for b in h.branches:
                if not b.is_constant:
                    I.append(n+h.id)
                    J.append(b.id)
                    if b.upstream == h:
                        V.append(1.0)
                    else:
                        V.append(-1.0)
        self.jacobian = sparse.csr_matrix((V,(I,J)), shape=(m+n,m+n))
        self.rhs = np.zeros(m + n)
        if self.X is None:
            self.X = np.zeros(m + n + len(self.boundary_hubs))

    def _update_matrix(self, X):
        n = len(self.branches)
        m = len(self.hubs)
        self.rhs = np.zeros(m + n)
        for b in self.branches:
            cm, cu, cd, f = b.flow_eq_grad(X[b.id], X[b.upstream.id + n], X[b.downstream.id + n])
            self.jacobian.data[self.v_ind[b.id, 0]] = cm
            if not b.upstream.is_constant:
                self.jacobian.data[self.v_ind[b.id, 1]] = cu
            if not b.downstream.is_constant:
                self.jacobian.data[self.v_ind[b.id, 2]] = cd
            self.rhs[b.id] = -f
        for h in self.hubs:
            s = 0.
            for b in h.branches:
                if b.upstream == h:
                    if b.is_constant:
                        s += b.mf
                    else:
                        s += X[b.id]
                else:
                    if b.is_constant:
                        s -= b.mf
                    else:
                        s -= X[b.id]
            self.rhs[n+h.id] = -s            

    def _init_solution(self, X):
        n = len(self.branches)
        pb = [h.p for h in self.boundary_hubs]
        if len(pb) == 0:
            print('no boundary hub')
            return False
        if len(pb) == 1:
            pmax = 1.2*pb[0]
            pmin = 0.7*pb[0]
        else:
            pmax = 1.1*max(pb)
            pmin = 0.9*min(pb)
        for i in range(len(self.hubs)):
            X[i + n] = pmin+(pmax-pmin)*random.random()
            self.hubs[i].p = X[i + n]
        for b in self.branches:
            X[b.id] = b.calc_mf(b.upstream.p - b.downstream.p, b.upstream.p)
        for h in self.boundary_hubs:
            X[h.id + n] = h.p

    def newton_step(self, X, n=1, relax=0.6):
        mn = len(self.branches) + len(self.hubs)
        for i in range(n):
            self._update_matrix(X)
            if self.method == 'bicg':
                dx, _ = sparse.linalg.bicg(self.jacobian, self.rhs)
            else:
                dx = sparse.linalg.spsolve(self.jacobian, self.rhs)
            X[:mn] += dx*relax
        return X

    def solve_newton(self, initialize=False, maxiter=10000, tol=1e-7, relax=1.0):
        mn = len(self.branches) + len(self.hubs)
        if initialize:
            self._init_solution(self.X)
        for i in range(maxiter):
            self._update_matrix(self.X)
            if self.method == 'bicg':
                dx, _ = sparse.linalg.bicg(self.jacobian, self.rhs)
            else:
                dx = sparse.linalg.spsolve(self.jacobian, self.rhs)
            self.X[:mn] += dx*relax
            if max(abs(dx)) < tol:
                return
        print('iteration is not converged')

    def solve_flow(self, initialize=True, tol=1e-7, relax=1.0):
        if not self.check_network():
            print('network check failed')
            return
        self.set_id()
        self._init_Jacobian()
        self.solve_newton(initialize=initialize, relax=relax, tol=tol)

    def update_flow(self, X=None):
        if X is None:
            X = self.X
        n = len(self.branches)
        for i, b in zip(range(n), self.branches):
            if X[i] < 0:
                X[i] *= -1
                b.reverse()
            b.mf = X[i]
            b.inlet.mf = X[i]
            b.outlet.mf = X[i]            
        for i, h in zip(range(n, n + len(self.hubs)), self.hubs):
            h.p = X[i]
            h.update_flow()
        for b in self.branches:
            b.update_flow()

    def check_flow_ballance(self, err=1e-10, ouput=True):
        rslt = True
        for b in self.branches:
            if not b.check_flow_ballance(err):
                rslt = False
                if ouput:
                    print('branch {0:} flow ballance check failed, error={1:e}'.format(b.name, b.flow_ballance_error()))
        for h in self.hubs:
            if not h.check_flow_ballance(err):
                rslt = False
                if ouput:
                    print('hub {0:} flow ballance check failed, error={1:e}'.format(h.name, h.flow_ballance_error()))
        return rslt
    
    def _update_matrix_grad(self):
        n = len(self.branches)
        for b in self.branches:
            cm, cu, cd, _ = b.flow_var_grad(None, self.X[b.id], self.X[b.upstream.id + n], self.X[b.downstream.id + n])
            self.jacobian.data[self.v_ind[b.id, 0]] = cm
            if not b.upstream.is_constant:
                self.jacobian.data[self.v_ind[b.id, 1]] = cu
            if not b.downstream.is_constant:
                self.jacobian.data[self.v_ind[b.id, 2]] = cd

    def solve_flow_grad(self, var, solveflow=False):
        mn = len(self.branches) + len(self.hubs)
        self.dX = np.zeros(mn)
        rhs = np.zeros(mn)
        if solveflow:
            self.solve_flow()
            self.update_flow()
        #self._update_matrix_grad() # do not need this, use flow jacobian instead
        b = var[0]
        rhs[b.id] = b.flow_var_grad(var[1:], b.mf, b.upstream.p, b.downstream.p)[-1]
        if self.method == 'bicg':
            self.dX, _ = sparse.linalg.bicg(self.jacobian, rhs)
        else:
            self.dX = sparse.linalg.spsolve(self.jacobian, rhs)

    def _init_Jacobian_H(self):
        n = len(self.branches) + len(self.const_branches)
        m = len(self.hubs)
        I = []
        J = []
        V = []
        self.Hv_ind = np.ones((n+m, 2), np.int)*-1
        self.bh_ind = []
        self.hh_ind = np.zeros(m, np.int)
        for b in self.branches + self.const_branches:
            I.append(b.id)
            J.append(b.id)
            V.append(1.)
            self.Hv_ind[b.id, 0] = len(V) - 1
            if not b.upstream.is_constant:
                I.append(b.id)
                J.append(n+b.upstream.id)
                V.append(-1.)
                self.Hv_ind[b.id, 1] = len(V) - 1
            if b.branch_type is 'heatexchanger' and not b.heat_transfer.upstream.is_constant:
                I.append(b.id)
                J.append(n+b.heat_transfer.upstream.id)
                V.append(1.)
                self.Hv_ind[b.id, 2] = len(V) - 1
                if not b.upstream.is_constant and b.upstream.id > b.heat_transfer.upstream.id:
                    self.Hv_ind[b.id, 2] -= 1
                    self.Hv_ind[b.id, 1] += 1
            
        for h in self.hubs:
            for b in h.branches:
                if b.downstream == h:
                    I.append(n+h.id)
                    J.append(b.id)
                    V.append(1.0)
                    self.bh_ind.append([b, len(V)-1])
            I.append(n+h.id)
            J.append(n+h.id)
            V.append(1.0)
            self.hh_ind[h.id] = len(V)-1
        self.Hjacobian = sparse.csr_matrix((V,(I,J)), shape=(m+n,m+n))
        self.Hrhs = np.zeros(m+n)
        if self.H is None:
            self.H = np.zeros(m + n + len(self.boundary_hubs))

    def _update_matrix_H(self, H):
        n = len(self.branches) + len(self.const_branches)
        for b in self.branches + self.const_branches:
            if b.branch_type is 'heatexchanger':
                cu, ct, cd, f = b.energy_eq_grad(H[n + b.upstream.id], H[b.id], H[n + b.heat_transfer.upstream.id])
                self.Hjacobian.data[self.Hv_ind[b.id, 0]] = cd
                if not b.upstream.is_constant:
                    self.Hjacobian.data[self.Hv_ind[b.id, 1]] = cu
                if not b.heat_transfer.upstream.is_constant:
                    self.Hjacobian.data[self.Hv_ind[b.id, 2]] = ct
            else:
                cu, cd, f = b.energy_eq_grad(H[n + b.upstream.id], H[b.id])
                self.Hjacobian.data[self.Hv_ind[b.id, 0]] = cd
                if not b.upstream.is_constant:
                    self.Hjacobian.data[self.Hv_ind[b.id, 1]] = cu
            self.Hrhs[b.id] = -f

        for x in self.bh_ind:
            self.Hjacobian.data[x[1]] = -x[0].mf
            
        for h in self.hubs:  
            mft = h.total_flux()
            self.Hjacobian.data[self.hh_ind[h.id]] = mft
            s = 0.
            for b in h.branches:
                if b.downstream == h:
                    s += b.mf*H[b.id]
            self.Hrhs[n+h.id] = s - mft*H[n + h.id]
        
    def _init_solution_H(self, H):
        n = len(self.branches) + len(self.const_branches)
        hb = [h.h for h in self.boundary_hubs]
        if len(hb) == 0:
            print('no boundary hub')
            return False
        if len(hb) == 1:
            hmax = 1.02*hb[0]
            hmin = 0.99*hb[0]
        else:
            hmax = 1.01*max(hb)
            hmin = 1.0*min(hb)
        for i in range(n,n + len(self.hubs)):
            H[i] = hmin+(hmax-hmin)*random.random()
            self.hubs[i-n].h = H[i]
        for b in self.branches + self.const_branches:
            H[b.id] = hmin+(hmax-hmin)*random.random()
        for h in self.boundary_hubs:
            H[n + h.id] = h.h

    def newton_step_H(self, H, n=1, relax=1.):
        mn = len(self.branches) + len(self.const_branches) + len(self.hubs)
        for _ in range(n):
            self._update_matrix_H(H)
            if self.method == 'bicg':
                dH, _ = sparse.linalg.bicg(self.Hjacobian, self.Hrhs)
            else:
                dH = sparse.linalg.spsolve(self.Hjacobian, self.Hrhs)
        H[:mn] += relax*dH
        return H

    def solve_newton_H(self, initialize=False, maxiter=10000, tol=1e-6, relax=0.6):
        mn = len(self.branches) + len(self.const_branches) + len(self.hubs)
        if initialize:
            self._init_solution_H(self.H)
        for _ in range(maxiter):
            self._update_matrix_H(self.H)
            if self.method == 'bicg':
                dH, _ = sparse.linalg.bicg(self.Hjacobian, self.Hrhs)
            else:
                dH = sparse.linalg.spsolve(self.Hjacobian, self.Hrhs)
            self.H[:mn] += relax*dH
            if max(abs(dH)) < tol:
                return
        print('iteration is not converged')
                
    def solve_energy(self, solveflow=False, initialize=True, relax=0.6):
        if solveflow:
            self.solve_flow(initialize=initialize, relax=relax)
            self.update_flow()
        self._init_Jacobian_H()
        self.solve_newton_H(initialize=initialize, relax=relax)

    def update_energy(self, H=None):
        if H is None:
            H = self.H
        n = len(self.branches) + len(self.const_branches)
        for i, h in zip(range(n, n+len(self.hubs)), self.hubs):
            h.h = H[i]
            h.update_energy()
        for h in self.boundary_hubs:
            h.update_energy()
        for i, b in zip(range(n), self.branches + self.const_branches):
            b.outlet.h = H[i]
            b.update_energy()

    def solve_energy_grad(self, var, solve_energy=False, solve_flowgrad=False):
        n = len(self.branches) + len(self.const_branches)
        rhs = np.zeros(len(self.H))
        if solve_flowgrad:
            self.solve_flow_grad(var)
        if solve_energy:
            self.solve_energy()
            self.update_energy()
        b0 = var[0]
        if b0.branch_type is 'heatexchanger':
            b1 = b0.heat_transfer
        var_mf = var[1:-1] + ['mf']
        for b in self.branches + self.const_branches:
            if b.is_constant:
                mx1 = 0.
            else:
                mx1 = self.dX[b.id]
            if b is b0:
                if b.branch_type is 'heatexchanger':
                    xs0 = b.energy_var_grad(var[1:], self.H[b.upstream.id], self.H[n + b.heat_transfer.upstream.id])
                    xm1, xm2 = b.energy_var_grad(var_mf[1:], self.H[b1.upstream.id], self.H[n + b1.heat_transfer.upstream.id])
                    if b1.is_constant:
                        mx2 = 0.
                    else:
                        mx2 = self.dX[b1.id]
                    rhs[b.id] = xs0 + xm1*mx1 + xm2*mx2
                    xs1 = b1.energy_var_grad(var[1:], self.H[b1.upstream.id], self.H[n + b1.heat_transfer.upstream.id])
                    xm1, xm2 = b1.energy_var_grad(var_mf[1:], self.H[b1.upstream.id], self.H[n + b1.heat_transfer.upstream.id])
                    rhs[b1.id] = xs1 + xm1*mx1 + xm2*mx2
                else:
                    rhs[b.id] = b.energy_var_grad(var[1:], self.H[b.upstream.id]) +\
                         b.energy_var_grad(var_mf[1:], self.H[b.upstream.id])*mx1
            elif not b.is_constant:
                if b.branch_type is not 'heatexchanger':
                    rhs[b.id] = b.energy_var_grad(var_mf, self.H[b.upstream.id])*self.dX[b.id]
                elif b is not b1:
                    xm1, xm2 = b.energy_var_grad(var_mf[1:], self.H[b1.upstream.id], self.H[n + b1.heat_transfer.upstream.id])
                    if b.heat_transfer.is_constant:
                        mx2 = 0.
                    else:
                        mx2 = self.dX[b.heat_transfer.id]
                    rhs[b.id] = xm1*mx1 + xm2*mx2

        for h in self.hubs:
            s1 = s2 = 0.
            for b in h.branches:
                if b.downstream == h:
                    s1 += self.dX[b.id]*self.H[b.id]
                    s2 += self.dX[b.id]
            rhs[n + h.id] = s1 - s2*self.H[n + h.id]
        if self.method == 'bicg':
            self.dH, _ = sparse.linalg.bicg(self.Hjacobian, rhs)
        else:
            self.dH = sparse.linalg.spsolve(self.Hjacobian, rhs)

