import re
import subprocess
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
from cobyqa import minimize
from joblib import Parallel, delayed


class SOLARProblem:

    def __init__(self, pb_id):
        if isinstance(pb_id, float) and pb_id.is_integer():
            pb_id = int(pb_id)
        if not isinstance(pb_id, int) or pb_id < 1 or pb_id > 10:
            raise ValueError('pb_id must be an integer between 1 and 10')
        self._pb_id = pb_id

        self._path_tests = Path(__file__).parent / 'solar' / 'tests'
        self._path_solar = Path(__file__).parent / 'solar' / 'bin' / 'solar'
        if not self._path_tests.is_dir():
            raise FileNotFoundError('solar tests directory not found')
        if not self._path_solar.is_file():
            raise FileNotFoundError('solar executable not found')

        self._history = {}

    @property
    def n(self):
        return self.x0.size

    @property
    def x0(self):
        return self._get_param('X0')

    @property
    def xl(self):
        return self._get_param('LOWER_BOUND', -np.inf)

    @property
    def xu(self):
        return self._get_param('UPPER_BOUND', np.inf)

    def fun(self, x):
        try:
            return self._eval(x)[0]
        except:
            return np.inf

    def cub(self, x):
        return self._eval(x)[1]

    def _get_param(self, param, fill_unknown=None):
        pb_id = self._pb_id
        if pb_id == 10:
            pb_id = 6
        try:
            path_param = next(self._path_tests.glob(f'{pb_id}_*/param.txt'))
            with open(path_param) as f:
                for line in f:
                    match = re.compile(fr'^{param}\s*\((?P<value>.*)\)$').match(line)
                    if match:
                        param_value = match.group('value')
                        if fill_unknown is not None:
                            param_value = param_value.replace('-', str(fill_unknown))
                        return np.fromstring(param_value, sep=' ')
        except StopIteration:
            path_x0 = next(self._path_tests.glob(f'{pb_id}_*/x0.txt'))
            x0 = np.fromfile(path_x0, sep=' ')
            if param == 'X0':
                return x0
            elif param == 'LOWER_BOUND':
                return np.full(x0.size, -np.inf)
            elif param == 'UPPER_BOUND':
                return np.full(x0.size, np.inf)
        raise ValueError(f'parameter {param} not found')

    def _eval(self, x):
        x_str = ' '.join(map(str, x))
        if x_str in self._history:
            return self._history[x_str]

        with tempfile.NamedTemporaryFile(delete_on_close=False) as f:
            f.write(str.encode(x_str))
            f.close()
            result = subprocess.run([self._path_solar, str(self._pb_id), f.name], stdout=subprocess.PIPE)
        output = np.fromstring(result.stdout.decode(), sep=' ')
        self._history[x_str] = (output[0], output[1:])
        return output


@delayed
def solve(pb_id, i_restart):
    print(f'Solving SOLAR{pb_id}({i_restart})')

    rng = np.random.default_rng(i_restart)
    p = SOLARProblem(pb_id)
    x0 = np.clip(p.x0 + rng.normal(0.0, 1.0, p.n), p.xl, p.xu)
    res = minimize(p.fun, x0, xl=p.xl, xu=p.xu, cub=p.cub, options={'store_history': True})

    output = Path(__file__).parent / 'out' / f'solar{pb_id}_{i_restart}'
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'fun_history.npy', res.fun_history)
    np.save(output / 'cub_history.npy', res.cub_history)
    print(f'Results saved in {output}')


if __name__ == '__main__':
    n_restart = 10
    # Parallel(n_jobs=-1)(solve(pb_id, i_restart) for pb_id, i_restart in product([6, 10], range(n_restart)))
    for pb_id in [6, 10]:
        for i_restart in range(n_restart):
            try:
                output = Path(__file__).parent / 'out' / f'solar{pb_id}_{i_restart}'
                fun_history = np.load(output / 'fun_history.npy')
                cub_history = np.load(output / 'cub_history.npy')
                if cub_history.size > 0:
                    maxcv_history = np.max(cub_history, 1, initial=0.0)
                    fun_history = fun_history[maxcv_history <= 0.0]
                print(f'SOLAR{pb_id}({i_restart}): {np.min(fun_history)}')
            except FileNotFoundError:
                print(f'SOLAR{pb_id}({i_restart}): not solved')
