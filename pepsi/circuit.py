
class Circuit:
    pass


class Primitive(Circuit):
    pass


class Gate(Primitive):
    pass


def gate(name, fields):
    def initialize(self, *args):
        if len(args) != len(fields):
            raise ValueError(f'{name} needs {len(fields)} fields: {fields}')
        for attr, value in zip(fields, args):
            setattr(self, attr, value)
    return type(name, (Gate,), {'__init__': initialize})


I = gate('I', ['qubit'])
H = gate('H', ['qubit'])
X = gate('X', ['qubit'])
Y = gate('Y', ['qubit'])
Z = gate('Z', ['qubit'])
S = gate('S', ['qubit'])
T = gate('T', ['qubit'])
R = gate('R', ['qubit', 'theta'])
Rx = gate('Rx', ['qubit', 'theta'])
Ry = gate('Ry', ['qubit', 'theta'])
Rz = gate('Rz', ['qubit', 'theta'])

CX = gate('CX', ['ctrl', 'target'])
CY = gate('CY', ['ctrl', 'target'])
CZ = gate('CZ', ['ctrl', 'target'])
SWAP = gate('SWAP', ['first', 'second'])

CCX = gate('CCX', ['ctrl1', 'ctrl2', 'target'])


class Measure(Primitive):
    """Realistic measurement"""
    def __init__(self, qubits):
        self.qubits = qubits


class Peek(Primitive):
    """Non-collapse measurement (possible for simulators)"""
    def __init__(self, qubits, nsample):
        self.qubits = qubits
        self.nsample = nsample


class Sequential(Circuit):
    """A sequence of subcircuits being applied one by one"""
    def __init__(self, *circuits):
        self.circuits = list(circuit)

    def __iter__(self):
        yield from self.circuits


def sequential(generator):
    """Create a subclass of Sequential from a generator function

    Example:
        @sequential
        def Hadmards(qubits):
            for qubit in qubits:
                yeild H(qubit)
    """
    def initialize(self, *args, **kwargs):
        super(self.__class__, self).__init__(generator(*args, **kwargs))
    return type(generator.__name__, (Sequential,), {'__init__': initialize})
