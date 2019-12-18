import pepsi

class TFI_Hamiltonian:
    def __init__(self, J, h, nrows, ncols, tb='numpy', tau=0.01, threshold=None):
        self.J = J
        self.h = h
        self.nrows = nrows
        self.ncols = ncols
        self.tb = tb
        self.tau = tau
        self.threshold = threshold
        self.ite_single_gate = tb.expm(tb.array([[ 1.+0.j,  0.+0.j],\
                                    [ 0.+0.j, -1.+0.j]])*(-tau*h))
        self.ite_double_gate = tb.expm(tb.array([[[[ 1.+0.j,  0.+0.j],\
                                    [ 0.+0.j,  0.+0.j]], \
                                    [[ 0.+0.j, -1.+0.j], \
                                    [ 0.+0.j, -0.+0.j]]], \
                                    [[[ 0.+0.j,  0.+0.j], \
                                      [-1.+0.j, -0.+0.j]], \
                                    [[ 0.+0.j, -0.+0.j], \
                                    [-0.+0.j,  1.-0.j]]]])*(-J*tau))
        self.double_gate = tb.expm(tb.array([[[[ 1.+0.j,  0.+0.j],\
                                    [ 0.+0.j,  0.+0.j]], \
                                    [[ 0.+0.j, -1.+0.j], \
                                    [ 0.+0.j, -0.+0.j]]], \
                                    [[[ 0.+0.j,  0.+0.j], \
                                      [-1.+0.j, -0.+0.j]], \
                                    [[ 0.+0.j, -0.+0.j], \
                                    [-0.+0.j,  1.-0.j]]]])*(-J))
        self.single_gate = tb.expm(tb.array([[ 1.+0.j,  0.+0.j],\
                                    [ 0.+0.j, -1.+0.j]])*(-h))
    def ite_gate_pos(self):
        double_gate_pos = []
        single_gate_pos = []
        # horizontal double gate positions
        for i in range(self.nrows-1):
            for j in range(self.ncols-1):
                double_gate_pos.append((i,j), (i, j+1))
        # vertical double gate positions
        for i in range(self.nrows-1):
            for j in range(self.ncols-1):
                double_gate_pos.append((i,j), (i+1, j))
        # single qubit gates
        for i in range(self.nrows):
            for j in range(self.ncols):
                single_gate_pos.append((i,j))
        return double_gate_pos, single_gate_pos
    
    def run_ite(self, peps, time_steps=1000, normalization_freq = 10):
        dg, sg = self.ite_double_gate, ite_single_gate
        dgpos, sgpos = self.ite_gate_pos()
        for it in time_steps:
            for sites in dgpos:
                peps.apply_operator(dg,sites, threshold = self.threshold)
            for site in sgpos:
                peps.apply_operator(sg, site, threshold = self.threshold)
        # normalize
            if it % normalization_freq==0: 
                peps/= peps.norm()
        return peps
    
    def energy(self, peps, use_cache=True):
        dglist = [self.double_gate for i in range((self.ncols-1*(self.nrows-1)))]
        sglist = [self.double_gate for i in range (self.ncols*self.nrows)]
        dgpos, sgpos = self.ite_gate_pos()
        observable = [].append(zip(dglist, dgpos))
        observable.append(zip(sglist, sgpos))
        return peps.expectation(observable, use_cache =use_cache)
# end class
