## Koala

Koala is a quantum circuit/state simulator based on projected entangled–pair states (PEPS) tensor networks.

### Installation

NumPy and [TensorBackends](https://github.com/cyclops-community/tensorbackends) are required.
It is recommended to install Koala and TensorBackends in the editable (`pip install -e`) mode, as they are currently under development.

```console
git clone https://github.com/cyclops-community/tensorbackends.git
pip install -e ./tensorbackends
git clone https://github.com/YiqingZhouKelly/koala.git
pip install -e ./koala
```

Parallelization is provided by [Cyclops Tensor Framework](https://github.com/cyclops-community/ctf), and is optional.

### Testing
```console
python -m unittest
```

### Get Started
```python
from koala import peps, Observable, Gate
from tensorbackends.interface import ImplicitRandomizedSVD

# initialize a 2 by 3 state with peps approach and numpy backend
qstate = peps.computational_zeros(2, 3, backend='numpy')

# we also provide the state vector approach and a parallel backend
# statevector.computational_zeros(2, 3, backend='ctf')

# apply one gate or a list of gates
qstate.apply_gate(Gate('X', [], [0])) # (name, parameters, qubits)
qstate.apply_circuit([
    Gate('R', [0.42], [2]),
    Gate('CX', [], [1,4])
], update_option=peps.LocalGramQRUpdate(rank=4))
# choose from a list of update algorithms and optionally specify the cap bond dimension for approximate state evolution

# or apply arbitrary single-site or two-site operators
# qstate.apply_operator(np.array(...), [0])

# compute the amplitude, probability, and expectation value
qstate.amplitude([1,0,0,1,0,0])
qstate.probability([1,0,0,1,0,0])
observable = 1.5 * Observable.sum([
    Observable.Z(0),
    Observable.XY(3, 4) * 2
])
qstate.expectation(observable, 
    contract_option=peps.BMPS(ImplicitRandomizedSVD(rank=8)), 
    use_cache=True
)
# choose from a list of contraction algorithms and optionally specify the cap bond dimension for approximate contraction
# use built-in caching option to trade memory for time in expectation value calculations
```
