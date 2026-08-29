import numpy as np
from cobyqa import minimize as cobyqa_minimize
from optiprofiler import benchmark
from pdfo import pdfo as pdfo_minimize
from pybobyqa import solve as pybobyqa_minimize
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize as scipy_minimize


def uobyqa(fun, x0):
    """
    Solve an unconstrained optimization problem using UOBYQA.
    """
    res = pdfo_minimize(fun, x0, method='uobyqa')
    return res.x


def newuoa(fun, x0):
    """
    Solve an unconstrained optimization problem using NEWUOA.
    """
    res = pdfo_minimize(fun, x0, method='newuoa')
    return res.x


def bobyqa(fun, x0, lb=None, ub=None):
    """
    Solve a bound-constrained optimization problem using BOBYQA.
    """
    bounds = _build_bounds(lb, ub)
    res = pdfo_minimize(fun, x0, method='bobyqa', bounds=bounds)
    return res.x


def pybobyqa(fun, x0, lb=None, ub=None):
    """
    Solve a bound-constrained optimization problem using Py-BOBYQA.
    """
    res = pybobyqa_minimize(fun, x0, bounds=(lb, ub))
    return res.x


def lincoa(fun, x0, lb=None, ub=None, a_ub=None, b_ub=None, a_eq=None, b_eq=None):
    """
    Solve a linearly constrained optimization problem using LINCOA.
    """
    bounds = _build_bounds(lb, ub)
    constraints = _build_linear_constraints(a_ub, b_ub, a_eq, b_eq)
    res = pdfo_minimize(fun, x0, method='lincoa', bounds=bounds, constraints=constraints)
    return res.x


def cobyla(fun, x0, lb=None, ub=None, a_ub=None, b_ub=None, a_eq=None, b_eq=None, c_ub=None, c_eq=None):
    """
    Solve a nonlinearly constrained optimization problem using COBYLA.
    """
    bounds = _build_bounds(lb, ub)
    constraints = _build_linear_constraints(a_ub, b_ub, a_eq, b_eq)
    constraints += _build_nonlinear_constraints(c_ub, c_eq, x0)
    res = scipy_minimize(fun, x0, method='cobyla', bounds=bounds, constraints=constraints)
    # res = pdfo_minimize(fun, x0, method='cobyla', bounds=bounds, constraints=constraints)
    return res.x


def cobyqa(fun, x0, lb=None, ub=None, a_ub=None, b_ub=None, a_eq=None, b_eq=None, c_ub=None, c_eq=None):
    """
    Solve a nonlinearly constrained optimization problem using COBYQA.
    """
    bounds = _build_bounds(lb, ub)
    constraints = _build_linear_constraints(a_ub, b_ub, a_eq, b_eq)
    constraints += _build_nonlinear_constraints(c_ub, c_eq, x0)
    res = cobyqa_minimize(fun, x0, bounds=bounds, constraints=constraints)
    return res.x


def _build_bounds(lb, ub):
    """
    Build the bound constraints.
    """
    if lb is None or ub is None:
        return None
    return Bounds(lb, ub)


def _build_linear_constraints(a_ub, b_ub, a_eq, b_eq):
    """
    Build the linear constraints.
    """
    constraints = []
    if a_ub is not None and b_ub is not None:
        if b_ub.size > 0:
            constraints.append(LinearConstraint(a_ub, -np.inf, b_ub))
    if a_eq is not None and b_eq is not None:
        if b_eq.size > 0:
            constraints.append(LinearConstraint(a_eq, b_eq, b_eq))
    return constraints


def _build_nonlinear_constraints(c_ub, c_eq, x0):
    """
    Build the nonlinear constraints.
    """
    constraints = []
    if c_ub is not None:
        c_ub_x0 = c_ub(x0)
        if c_ub_x0.size > 0:
            constraints.append(NonlinearConstraint(c_ub, -np.inf, np.zeros_like(c_ub_x0)))
    if c_eq is not None:
        c_eq_x0 = c_eq(x0)
        if c_eq_x0.size > 0:
            constraints.append(NonlinearConstraint(c_eq, np.zeros_like(c_eq_x0), np.zeros_like(c_eq_x0)))
    return constraints


if __name__ == '__main__':
    # Run the benchmark on all unconstrained problems with up to 50 variables.
    benchmark(
        [cobyqa, newuoa, cobyla],
        solver_names=['COBYQA', 'NEWUOA', 'COBYLA'],
        benchmark_id='out_unconstrained',
        maxdim=50,
    )
    exit(0)
    benchmark(
        [cobyqa, newuoa, cobyla],
        solver_names=['COBYQA', 'NEWUOA', 'COBYLA'],
        benchmark_id='out_unconstrained',
        maxdim=50,
        feature_name='noisy',
    )

    # Run the benchmark on all bound-constrained problems with up to 50 variables.
    benchmark(
        [cobyqa, bobyqa, cobyla],
        solver_names=['COBYQA', 'BOBYQA', 'COBYLA'],
        benchmark_id='out_bound-constrained',
        ptype='b',
        maxdim=50,
        maxb=100,
        project_x0=True,
    )
    benchmark(
        [cobyqa, bobyqa, cobyla],
        solver_names=['COBYQA', 'BOBYQA', 'COBYLA'],
        benchmark_id='out_bound-constrained',
        ptype='b',
        maxdim=50,
        minb=1,
        maxb=100,
        feature_name='unrelaxable_constraints',
        project_x0=True,
    )
    benchmark(
        [cobyqa, bobyqa, cobyla],
        solver_names=['COBYQA', 'BOBYQA', 'COBYLA'],
        benchmark_id='out_bound-constrained',
        ptype='b',
        maxdim=50,
        maxb=100,
        feature_name='noisy',
        project_x0=True,
    )

    # Run the benchmark on all linearly constrained problems with up to 50 variables and 5000 constraints.
    benchmark(
        [cobyqa, lincoa, cobyla],
        solver_names=['COBYQA', 'LINCOA', 'COBYLA'],
        benchmark_id='out_linearly-constrained',
        ptype='l',
        maxdim=50,
        maxb=100,
        maxlcon=5000,
        project_x0=True,
    )
    benchmark(
        [cobyqa, lincoa, cobyla],
        solver_names=['COBYQA', 'LINCOA', 'COBYLA'],
        benchmark_id='out_linearly-constrained',
        ptype='l',
        maxdim=50,
        minb=1,
        maxb=100,
        maxlcon=5000,
        feature_name='unrelaxable_constraints',
        project_x0=True,
    )
    benchmark(
        [cobyqa, lincoa, cobyla],
        solver_names=['COBYQA', 'LINCOA', 'COBYLA'],
        benchmark_id='out_linearly-constrained',
        ptype='l',
        maxdim=50,
        maxb=100,
        maxlcon=5000,
        feature_name='noisy',
        project_x0=True,
    )

    # Run the benchmark on all nonlinearly constrained problems with up to 50 variables and 5000 constraints.
    benchmark(
        [cobyqa, cobyla],
        solver_names=['COBYQA', 'COBYLA'],
        benchmark_id='out_nonlinearly-constrained',
        ptype='n',
        maxdim=50,
        maxb=100,
        maxlcon=5000,
        maxnlcon=5000,
        project_x0=True,
    )
    benchmark(
        [cobyqa, cobyla],
        solver_names=['COBYQA', 'COBYLA'],
        benchmark_id='out_nonlinearly-constrained',
        ptype='n',
        maxdim=50,
        minb=1,
        maxb=100,
        maxlcon=5000,
        maxnlcon=5000,
        feature_name='unrelaxable_constraints',
        project_x0=True,
    )
    benchmark(
        [cobyqa, cobyla],
        solver_names=['COBYQA', 'COBYLA'],
        benchmark_id='out_nonlinearly-constrained',
        ptype='n',
        maxdim=50,
        maxb=100,
        maxlcon=5000,
        maxnlcon=5000,
        feature_name='noisy',
        project_x0=True,
    )

    # Run the benchmark on all problems with up to 50 variables and 5000 constraints.
    benchmark(
        [cobyqa, cobyla],
        solver_names=['COBYQA', 'COBYLA'],
        benchmark_id='out_all',
        ptype='ubln',
        maxdim=50,
        maxb=100,
        maxlcon=5000,
        maxnlcon=5000,
        project_x0=True,
    )
    benchmark(
        [cobyqa, cobyla],
        solver_names=['COBYQA', 'COBYLA'],
        benchmark_id='out_all',
        ptype='ubln',
        maxdim=50,
        minb=1,
        maxb=100,
        maxlcon=5000,
        maxnlcon=5000,
        feature_name='unrelaxable_constraints',
        project_x0=True,
    )
    benchmark(
        [cobyqa, cobyla],
        solver_names=['COBYQA', 'COBYLA'],
        benchmark_id='out_all',
        ptype='ubln',
        maxdim=50,
        maxb=100,
        maxlcon=5000,
        maxnlcon=5000,
        feature_name='noisy',
        project_x0=True,
    )
