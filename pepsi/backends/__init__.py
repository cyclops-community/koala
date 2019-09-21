"""
This module defines backends and backends are imported on demand.
"""


def get_backend(name):
    if name not in _BACKENDS:
        raise ValueError(f"Backend {name} does not exsit")
    return  _BACKENDS[name]()


def numpy_backend():
    from .numpy_backend import NumPyBackend
    return NumPyBackend


def ctf_backend():
    from .ctf_backend import CTFBackend
    return CTFBackend


_BACKENDS = {
    'numpy': numpy_backend,
    'ctf': ctf_backend,
}
