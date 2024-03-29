import logging
import pickle
import re
import subprocess
import time
from enum import Enum
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from cobyqa import minimize
from scipy.optimize import Bounds, NonlinearConstraint


class SOLARInputType(str, Enum):
    """
    Input types for the SOLAR simulator.
    """
    REAL = 'R'
    INTEGER = 'I'


class SOLAROutputType(str, Enum):
    """
    Output types for the SOLAR simulator.
    """
    OBJECTIVE = 'OBJ'
    CONSTRAINT = 'CSTR'


class SOLARProblem:
    """
    Wrapper for a SOLAR problem.
    """

    def __init__(self, pb_id):
        """
        Initialize a SOLAR problem.

        Parameters
        ----------
        pb_id : int
            Problem identifier (between 1 and 10).
        """
        if isinstance(pb_id, float) and pb_id.is_integer():
            pb_id = int(pb_id)
        if not isinstance(pb_id, int) or pb_id < 1 or pb_id > 10:
            raise ValueError('pb_id must be an integer between 1 and 10')
        self._pb_id = pb_id

        # Get the path to the SOLAR executable.
        self._path_solar = Path(__file__).parent / 'solar' / 'bin' / 'solar'
        if not self._path_solar.is_file():
            raise FileNotFoundError('solar executable not found')

        # Get the raw parameters of the problem. They include the problem
        # dimension, the bound constraints, the initial guess, etc.
        self._raw_params = self._get_raw_params()

        # Initialize the history of evaluations. We must keep track of the
        # evaluations because the SOLAR simulator returns the objective and
        # constraints values in a single array. We want to be able to retrieve
        # them separately, without having to re-evaluate the problem.
        self._history = {}

    @property
    def n(self):
        """
        Problem dimension.

        Returns
        -------
        int
            Problem dimension.
        """
        return int(self._raw_params['DIMENSION'])

    @property
    def m(self):
        """
        Number of constraints.

        Returns
        -------
        int
            Number of constraints.
        """
        output_types = self._raw_params['BB_OUTPUT_TYPE'].split()
        return len([output_type for output_type in output_types if output_type == SOLAROutputType.CONSTRAINT])

    @property
    def x0(self):
        """
        Initial guess.

        Returns
        -------
        dict
            Initial guess.
        """
        return self._get_array('X0')

    @property
    def xl(self):
        """
        Lower bound.

        Returns
        -------
        dict
            Lower bound.
        """
        return self._get_array('LOWER_BOUND')

    @property
    def xu(self):
        """
        Upper bound.

        Returns
        -------
        dict
            Upper bound.
        """
        return self._get_array('UPPER_BOUND')

    def fun(self, x):
        """
        Evaluate the objective function.

        Parameters
        ----------
        x : dict
            Point at which to evaluate the objective function.

        Returns
        -------
        {float, `numpy.ndarray`}
            Objective function value(s).
        """
        try:
            obj = self._eval(x)[0]
            if obj.size == 1:
                return obj[0]
            return obj
        except Exception as err:
            logger = get_logger(__name__)
            logger.warning(f'Evaluation of SOLAR{self._pb_id} at {x} failed: {err}')
            return np.inf

    def con(self, x):
        """
        Evaluate the constraints ``con(x) <= 0``.

        Parameters
        ----------
        x : dict
            Point at which to evaluate the constraints.

        Returns
        -------
        `numpy.ndarray`
            Constraints values.
        """
        try:
            return self._eval(x)[1]
        except Exception as err:
            logger = get_logger(__name__)
            logger.warning(f'Evaluation of SOLAR{self._pb_id} at {x} failed: {err}')
            return np.full(self.m, np.inf)

    def _get_raw_params(self):
        """
        Extract the raw parameters of the problem from the SOLAR simulator.

        Returns
        -------
        dict
            Raw parameters of the problem.
        """
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
        """
        Extract an array from the raw parameters of the problem.

        Parameters
        ----------
        param : {'X0', 'LOWER_BOUND', 'UPPER_BOUND'}
            Name of the parameter to extract.

        Returns
        -------
        dict
            Extracted array.
        """
        input_types = self._raw_params['BB_INPUT_TYPE'][1:-1].split()
        raw_array = self._raw_params[param][1:-1].split()
        array = {}
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
        """
        Evaluate the SOLAR simulator at a given point.

        Parameters
        ----------
        x : dict
            Point at which to evaluate the SOLAR simulator.

        Returns
        -------
        tuple
            Objective and constraint functions values.
        """
        # Check that the point is valid.
        if not isinstance(x, dict):
            raise TypeError('x must be a dict')
        if len(x) != self.n:
            raise ValueError(f'x must have {self.n} entries')
        input_types = self._raw_params['BB_INPUT_TYPE'][1:-1].split()
        for i in range(self.n):
            if f'x{i + 1}' not in x:
                raise ValueError(f'x must have an entry for x{i + 1}')
            if input_types[i] == SOLARInputType.INTEGER and not isinstance(x[f'x{i + 1}'], int):
                raise TypeError(f'x{i + 1} must be an integer')
            if input_types[i] == SOLARInputType.REAL and not isinstance(x[f'x{i + 1}'], float):
                raise TypeError(f'x{i + 1} must be a real number')

        # Convert the point to a string in the SOLAR simulator format.
        x_str = ' '.join([str(x[f'x{i + 1}']) for i in range(self.n)])

        # If the point has already been evaluated, return the cached values.
        if x_str in self._history:
            return self._history[x_str]

        # Otherwise, evaluate the SOLAR simulator. It expects the point to be
        # stored in a file. We use a temporary file that will be deleted on
        # context manager exit.
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


def get_logger(name=None, level=logging.INFO):
    """
    Get a logger.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int
        Logging level.

    Returns
    -------
    `logging.Logger`
        Logger with the given name. If a logger with the given name already
        exists, it is returned instead of creating a new one.
    """
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger.setLevel(level)

        # Attach a console handler (thread-safe).
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)-8s] %(message)s'))
        logger.addHandler(handler)
    return logger


def get_saving_path(pb_id, i_restart):
    """
    Get the path where to save the results of a SOLAR problem.

    Parameters
    ----------
    pb_id : int
        Problem identifier.
    i_restart : int
        Restart identifier.

    Returns
    -------
    `pathlib.Path`
        Path where to save the results of the SOLAR problem.
    """
    output = Path(__file__).parent / 'out' / 'solar'
    output.mkdir(parents=True, exist_ok=True)
    return output / f'solar{pb_id}_{i_restart}.pickle'


def solve(pb_id, i_restart):
    """
    Solve a SOLAR problem.

    Parameters
    ----------
    pb_id : {6, 10}
        Problem identifier.
    i_restart : int
        Restart identifier.
    """
    logger = get_logger(__name__)
    logger.info(f'Starting the computations for SOLAR{pb_id}({i_restart})')
    p = SOLARProblem(pb_id)

    def fun(x):
        """Wrapper for the objective function."""
        return p.fun({f'x{i + 1}': x[i] for i in range(p.n)})

    def con(x):
        """Wrapper for the constraint function."""
        return p.con({f'x{i + 1}': x[i] for i in range(p.n)})

    # Generate a random initial guess.
    rng = np.random.default_rng(i_restart)
    xl = list(p.xl.values())
    xu = list(p.xu.values())
    x0_rand = rng.uniform(xl, xu)

    # Solve the problem.
    bounds = Bounds(xl, xu)
    constraints = NonlinearConstraint(con, -np.inf, np.zeros(p.m))
    time_start = time.time()
    res = minimize(fun, x0_rand, bounds=bounds, constraints=constraints, options={'store_history': True})
    res.time = time.time() - time_start

    # Save the results.
    saving_path = get_saving_path(pb_id, i_restart)
    with open(saving_path, 'wb') as f:
        pickle.dump(res, f)
    logger.info(f'Results for SOLAR{pb_id}({i_restart}) saved in {saving_path}')


if __name__ == '__main__':
    # Solve the problems 6 and 10 with 32 random restarts.
    n_restart = 32
    with Pool() as p:
        p.starmap(solve, product([6, 10], range(n_restart)))
    time.sleep(1)  # wait for the logging messages to be printed.

    # Print the results.
    for pb_id in [6, 10]:
        print()
        print(f'Results for SOLAR{pb_id}:')
        print()
        print('+---------+--------------------+----------------------+----------------------+------------------+')
        print('| Restart | Objective function | Constraint violation | Function evaluations | Computation time |')
        print('+---------+--------------------+----------------------+----------------------+------------------+')
        for i_restart in range(n_restart):
            with open(get_saving_path(pb_id, i_restart), 'rb') as f:
                res = pickle.load(f)
                print(f'| {i_restart:^7} | {res.fun:^18.4e} | {res.maxcv:^20.4e} | {res.nfev:^20} | {res.time:^16.4e} |')
        print('+---------+--------------------+----------------------+----------------------+------------------+')
