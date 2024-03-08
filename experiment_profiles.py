import numpy as np
from cobyqa import minimize as cobyqa_minimize
from optiprofiler import set_cutest_problem_options, find_cutest_problems, run_benchmark
from pdfo import pdfo as pdfo_minimize
from pybobyqa import solve as pybobyqa_minimize
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint


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


def bobyqa(fun, x0, lb, ub):
    """
    Solve a bound-constrained optimization problem using BOBYQA.
    """
    bounds = _build_bounds(lb, ub)
    res = pdfo_minimize(fun, x0, method='bobyqa', bounds=bounds)
    return res.x


def pybobyqa(fun, x0, lb, ub):
    """
    Solve a bound-constrained optimization problem using Py-BOBYQA.
    """
    res = pybobyqa_minimize(fun, x0, bounds=(lb, ub))
    return res.x


def lincoa(fun, x0, lb, ub, a_ub, b_ub, a_eq, b_eq):
    """
    Solve a linearly constrained optimization problem using LINCOA.
    """
    bounds = _build_bounds(lb, ub)
    constraints = _build_linear_constraints(a_ub, b_ub, a_eq, b_eq)
    res = pdfo_minimize(fun, x0, method='lincoa', bounds=bounds, constraints=constraints)
    return res.x


def cobyla(fun, x0, lb, ub, a_ub, b_ub, a_eq, b_eq, c_ub, c_eq):
    """
    Solve a nonlinearly constrained optimization problem using COBYLA.
    """
    bounds = _build_bounds(lb, ub)
    constraints = _build_linear_constraints(a_ub, b_ub, a_eq, b_eq)
    constraints += _build_nonlinear_constraints(c_ub, c_eq, x0)
    res = pdfo_minimize(fun, x0, method='cobyla', bounds=bounds, constraints=constraints)
    return res.x


def cobyqa(fun, x0, lb, ub, a_ub, b_ub, a_eq, b_eq, c_ub, c_eq):
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
    return Bounds(lb, ub)


def _build_linear_constraints(a_ub, b_ub, a_eq, b_eq):
    """
    Build the linear constraints.
    """
    constraints = []
    if b_ub.size > 0:
        constraints.append(LinearConstraint(a_ub, -np.inf, b_ub))
    if b_eq.size > 0:
        constraints.append(LinearConstraint(a_eq, b_eq, b_eq))
    return constraints


def _build_nonlinear_constraints(c_ub, c_eq, x0):
    """
    Build the nonlinear constraints.
    """
    constraints = []
    c_ub_x0 = c_ub(x0)
    if c_ub_x0.size > 0:
        constraints.append(NonlinearConstraint(c_ub, -np.inf, np.zeros_like(c_ub_x0)))
    c_eq_x0 = c_eq(x0)
    if c_eq_x0.size > 0:
        constraints.append(NonlinearConstraint(c_eq, np.zeros_like(c_eq_x0), np.zeros_like(c_eq_x0)))
    return constraints


if __name__ == '__main__':
    # Consider only problems with up to 50 variables and 5000 constraints.
    set_cutest_problem_options(n_max=50, m_max=5000)

    # Run the benchmark on all unconstrained problems.
    cutest_problem_names = find_cutest_problems('unconstrained')
    run_benchmark([cobyqa, newuoa, cobyla], ['COBYQA', 'NEWUOA', 'COBYLA'], cutest_problem_names, benchmark_id='unconstrained')
    run_benchmark([cobyqa, newuoa, cobyla], ['COBYQA', 'NEWUOA', 'COBYLA'], cutest_problem_names, feature_name='noisy', benchmark_id='unconstrained')

    # Run the benchmark on all bound-constrained problems.
    cutest_problem_names = find_cutest_problems('bound')
    run_benchmark([cobyqa, bobyqa, pybobyqa, cobyla], ['COBYQA', 'BOBYQA', 'Py-BOBYQA', 'COBYLA'], cutest_problem_names, benchmark_id='bound-constrained', project_x0=True)
    run_benchmark([cobyqa, bobyqa, pybobyqa, cobyla], ['COBYQA', 'BOBYQA', 'Py-BOBYQA', 'COBYLA'], cutest_problem_names, feature_name='unrelaxable_constraints', benchmark_id='bound-constrained', project_x0=True)
    run_benchmark([cobyqa, bobyqa, pybobyqa, cobyla], ['COBYQA', 'BOBYQA', 'Py-BOBYQA', 'COBYLA'], cutest_problem_names, feature_name='noisy', benchmark_id='bound-constrained', project_x0=True)

    # Run the benchmark on all linearly constrained problems.
    cutest_problem_names = find_cutest_problems('adjacency linear')
    run_benchmark([cobyqa, lincoa, cobyla], ['COBYQA', 'LINCOA', 'COBYLA'], cutest_problem_names, benchmark_id='linearly-constrained', project_x0=True)
    run_benchmark([cobyqa, lincoa, cobyla], ['COBYQA', 'LINCOA', 'COBYLA'], cutest_problem_names, feature_name='unrelaxable_constraints', benchmark_id='linearly-constrained', project_x0=True)
    run_benchmark([cobyqa, lincoa, cobyla], ['COBYQA', 'LINCOA', 'COBYLA'], cutest_problem_names, feature_name='noisy', benchmark_id='linearly-constrained', project_x0=True)

    # Run the benchmark on all nonlinearly constrained problems.
    cutest_problem_names = find_cutest_problems('quadratic other')
    run_benchmark([cobyqa, cobyla], ['COBYQA', 'COBYLA'], cutest_problem_names, benchmark_id='nonlinearly-constrained')
    run_benchmark([cobyqa, cobyla], ['COBYQA', 'COBYLA'], cutest_problem_names, feature_name='unrelaxable_constraints', benchmark_id='nonlinearly-constrained')
    run_benchmark([cobyqa, cobyla], ['COBYQA', 'COBYLA'], cutest_problem_names, feature_name='noisy', benchmark_id='nonlinearly-constrained')

    # Run the benchmark on all problems.
    cutest_problem_names = find_cutest_problems('unconstrained bound adjacency linear quadratic other')
    run_benchmark([cobyqa, cobyla], ['COBYQA', 'COBYLA'], cutest_problem_names, benchmark_id='all')
    run_benchmark([cobyqa, cobyla], ['COBYQA', 'COBYLA'], cutest_problem_names, feature_name='noisy', benchmark_id='all')
