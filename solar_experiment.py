import re
import subprocess
import tempfile
from enum import Enum
from itertools import product
from pathlib import Path

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
            prog = re.compile(r'^\s*(?P<param>[A-Z0-9_]+)\s*(?P<value>.*)$')
            match = prog.match(line)
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
        x_str = ' '.join(map(str, x.values()))
        if x_str in self._history:
            return self._history[x_str]

        with tempfile.NamedTemporaryFile(delete_on_close=False) as f:
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
    p = SOLARProblem(1)
    print(p.fun(p.x0))
    print(p.cub(p.x0))
    print(p.fun(p.x0))
    # n_restart = 10
    # # Parallel(n_jobs=-1)(solve(pb_id, i_restart) for pb_id, i_restart in product([6, 10], range(n_restart)))
    # for pb_id in [6, 10]:
    #     for i_restart in range(n_restart):
    #         try:
    #             output = Path(__file__).parent / 'out' / f'solar{pb_id}_{i_restart}'
    #             fun_history = np.load(output / 'fun_history.npy')
    #             cub_history = np.load(output / 'cub_history.npy')
    #             if cub_history.size > 0:
    #                 maxcv_history = np.max(cub_history, 1, initial=0.0)
    #                 fun_history = fun_history[maxcv_history <= 0.0]
    #             print(f'SOLAR{pb_id}({i_restart}): {np.min(fun_history)}')
    #         except FileNotFoundError:
    #             print(f'SOLAR{pb_id}({i_restart}): not solved')
