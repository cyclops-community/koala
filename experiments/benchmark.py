from time import time
import tensorbackends


class Benchmark:
    def __init__(self, name, backend, standard=None, path=None, printout=True, reps=1, profile_time=True, profile_memory=True, additional_info={}):
        self.name = name
        self.backend = tensorbackends.get(backend)
        self.standard = standard
        self.path = path
        self.printout = printout
        self.reps = reps
        self.profile_time = profile_time
        self.profile_memory = profile_memory
        self.data = {}
        self.update_data({
            'backend': backend.name,
            'nproc': backend.nproc,
            **additional_info
        })
        self.result = None

    def __enter__(self):
        if self.profile_memory:
            import tracemalloc
            tracemalloc.start()

        if self.profile_time:
            if self.backend.name in ('ctf', 'ctfview'):
                import ctf
                ctf.initialize_flops_counter()
                self.timer_epoch = ctf.timer_epoch(self.name[:51])
                self.timer_epoch.begin()
            else:
                import cProfile
                self.pr = cProfile.Profile()
                self.pr.enable()
        self.start_time = time()

    def __exit__(self, type, value, traceback):
        duration = time() - self.start_time
        if self.profile_time:
            if self.backend.name in ('ctf', 'ctfview'):
                import ctf
                self.timer_epoch.end()
                self.update_data('flops', ctf.get_estimated_flops())
            else:
                self.pr.disable()
        self.update_data('time', duration / self.reps)

        if self.profile_memory:
            import tracemalloc
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.update_data('peak_memory', peak_memory)

        if self.result:
            self.update_data('result', self.result.real)
            if self.standard:
                abs_err = abs(self.result - self.standard)
                self.update_data({
                    'standard': self.standard,
                    'abs_err': abs_err,
                    'rel_err': abs(abs_err / self.standard)
                })

        try:
            import subprocess
            self.update_data('git_hash', subprocess.check_output(["git", "describe", "--always"]).strip().decode())
        except:
            pass

        if all((self.backend.rank == 0, self.printout, self.profile_time, self.backend.name not in ('ctf', 'ctfview'))):
            import pstats
            from io import StringIO
            s = StringIO()
            ps = pstats.Stats(self.pr, stream=s)
            ps.sort_stats('cumtime').print_stats(15)
            ps.sort_stats('tottime').print_stats(15)
            print(s.getvalue(), flush=True)

        self.save_data()

    def add_PEPS_info(self, p):
        return self.update_data({
            'shape': p.shape,
            'dims': str(p.dims)
        })

    def add_contract_info(self, contract_option):
        return self.update_data({
            'contract_approach': contract_option.name,
            'contract_option': str(contract_option)
        })

    def update_data(self, data, value=None):
        if not isinstance(data, dict):
            data = {data: value}
        if self.printout and self.backend.rank == 0:
            for k, v in data.items():
                print(f'{k}: {v}', flush=True)
        self.data.update(data)
        return self.data

    def save_data(self):
        if self.path and self.backend.rank == 0:
            import json
            try:
                with open(self.path, 'r') as fd:
                    try:
                        database = json.load(fd)
                    except json.decoder.JSONDecodeError:
                        database = []
            except FileNotFoundError:
                database = []
            with open(self.path, 'w+') as fd:
                database.append(self.data)
                json.dump(database, fd, indent=4)
        return self.data

