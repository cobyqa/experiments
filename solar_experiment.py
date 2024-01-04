import pickle
import re
import subprocess
from enum import Enum
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time

import numpy as np
from cobyqa import minimize
from joblib import Parallel, delayed


class SOLARInputType(str, Enum):
    REAL = 'R'
    INTEGER = 'I'


class SOLAROutputType(str, Enum):
    OBJECTIVE = 'OBJ'
    CONSTRAINT = 'CSTR'


class SOLARProblem:

    def __init__(self, pb_id):
        if isinstance(pb_id, float) and pb_id.is_integer():
            pb_id = int(pb_id)
        if not isinstance(pb_id, int) or pb_id < 1 or pb_id > 10:
            raise ValueError('pb_id must be an integer between 1 and 10')
        self._pb_id = pb_id

        self._path_solar = Path(__file__).parent / 'solar' / 'bin' / 'solar'
        if not self._path_solar.is_file():
            raise FileNotFoundError('solar executable not found')

        self._raw_params = self._get_raw_params()

        self._history = {}

    @property
    def n(self):
        return int(self._raw_params['DIMENSION'])

    @property
    def x0(self):
        return self._get_array('X0')

    @property
    def xl(self):
        return self._get_array('LOWER_BOUND')

    @property
    def xu(self):
        return self._get_array('UPPER_BOUND')

    def fun(self, x):
        try:
            obj = self._eval(x)[0]
            if obj.size == 1:
                return obj[0]
            return obj
        except:
            return np.inf

    def cub(self, x):
        try:
            return self._eval(x)[1]
        except:
            output_types = self._raw_params['BB_OUTPUT_TYPE'].split()
            return np.full(len([output_type for output_type in output_types if output_type == SOLAROutputType.CONSTRAINT]), np.inf)

    def _get_raw_params(self):
        result = subprocess.run([self._path_solar, str(self._pb_id)], stdout=subprocess.PIPE)
        output = result.stdout.decode()
        lines = output.splitlines()
        raw_params = {}
        for line in lines[lines.index('NOMAD parameters:') + 2:]:
            match = re.compile(r'^\s*(?P<param>[A-Z0-9_]+)\s*(?P<value>.*)$').match(line)
            if match:
                raw_params[match.group('param')] = match.group('value')
        return raw_params

    def _get_array(self, param):
        input_types = self._raw_params['BB_INPUT_TYPE'][1:-1].split()
        array = {}
        raw_array = self._raw_params[param][1:-1].split()
        for i in range(self.n):
            if raw_array[i] == '-':
                array[f'x{i + 1}'] = None
            elif input_types[i] == SOLARInputType.REAL:
                array[f'x{i + 1}'] = float(raw_array[i])
            elif input_types[i] == SOLARInputType.INTEGER:
                array[f'x{i + 1}'] = int(raw_array[i])
            else:
                raise ValueError(f'Unknown input type: {input_types[i]}')
        return array

    def _eval(self, x):
        if not isinstance(x, dict):
            if hasattr(x, '__len__'):
                x = {f'x{i + 1}': x[i] for i in range(self.n)}
            else:
                raise ValueError('x must be a dict or an array-like')

        x_str = ' '.join(map(str, x.values()))
        if x_str in self._history:
            return self._history[x_str]

        with NamedTemporaryFile(delete_on_close=False) as f:
            f.write(str.encode(x_str))
            f.close()
            result = subprocess.run([self._path_solar, str(self._pb_id), f.name], stdout=subprocess.PIPE)
        output = np.fromstring(result.stdout.decode(), sep=' ')
        output_types = self._raw_params['BB_OUTPUT_TYPE'].split()
        self._history[x_str] = (
            output[[j for j, output_type in enumerate(output_types) if output_type == SOLAROutputType.OBJECTIVE]],
            output[[j for j, output_type in enumerate(output_types) if output_type == SOLAROutputType.CONSTRAINT]],
        )
        return self._history[x_str]


def get_saving_path(pb_id, i_restart):
    output = Path(__file__).parent / 'out'
    output.mkdir(exist_ok=True)
    return output / f'solar{pb_id}_{i_restart}.p'


@delayed
def solve(pb_id, i_restart):
    print(f'Solving SOLAR{pb_id}({i_restart})')
    p = SOLARProblem(pb_id)

    rng = np.random.default_rng(i_restart)
    xl = list(p.xl.values())
    xu = list(p.xu.values())
    x0_rand = rng.uniform(xl, xu)

    time_start = time()
    res = minimize(p.fun, x0_rand, xl=xl, xu=xu, cub=p.cub, options={'store_history': True})
    res.time = time() - time_start

    saving_path = get_saving_path(pb_id, i_restart)
    with open(saving_path, 'wb') as f:
        pickle.dump(res, f)
    print(f'Results for SOLAR{pb_id}({i_restart}) saved in {saving_path}')


if __name__ == '__main__':
    n_restart = 32
    Parallel(n_jobs=-1)(solve(pb_id, i_restart) for pb_id, i_restart in product([6, 10], range(n_restart)))
    print()
    print('Results:')
    for pb_id in [6, 10]:
        for i_restart in range(n_restart):
            with open(get_saving_path(pb_id, i_restart), 'rb') as f:
                res = pickle.load(f)
                print(f'SOLAR{pb_id}({i_restart}): fun={res.fun}, maxcv={res.maxcv}, nfev={res.nfev}, time={res.time}')
