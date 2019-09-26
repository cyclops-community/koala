
from collections import namedtuple

Circuit = namedtuple('Circuit', ['gates'])
Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])
