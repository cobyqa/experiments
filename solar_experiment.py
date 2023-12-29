import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class SOLARProblem:

    def __init__(self, pb_id):
        if isinstance(pb_id, float) and pb_id.is_integer():
            pb_id = int(pb_id)
        if not isinstance(pb_id, int) or pb_id < 1 or pb_id > 10:
            raise ValueError('pb_id must be an integer between 1 and 10')
        self._pb_id = pb_id
        self._history = {}

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
        path_tests = Path(__file__).parent / 'solar' / 'tests'
        try:
            path_param = next(path_tests.glob(f'{pb_id}_*/param.txt'))
            with open(path_param) as f:
                for line in f:
                    match = re.compile(fr'^{param}\s*\((?P<value>.*)\)$').match(line)
                    if match:
                        param_value = match.group('value')
                        if fill_unknown is not None:
                            param_value = param_value.replace('-', str(fill_unknown))
                        return np.fromstring(param_value, sep=' ')
        except StopIteration:
            path_x0 = next(path_tests.glob(f'{pb_id}_*/x0.txt'))
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
            path_solar = Path(__file__).parent / 'solar' / 'bin' / 'solar'
            result = subprocess.run([path_solar, str(self._pb_id), f.name], stdout=subprocess.PIPE)
        output = np.fromstring(result.stdout.decode(), sep=' ')
        self._history[x_str] = (output[0], output[1:])
        return output


if __name__ == '__main__':
    pb_id = 1
    p = SOLARProblem(pb_id)
    print(p.fun(p.x0))
