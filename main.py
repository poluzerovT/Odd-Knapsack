import time
import itertools
import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
import logging
import sys

np.random.seed(27)

INPUT = 'input.txt'
OUPUT = 'output.txt'


def read_input():
    with open(INPUT, 'r') as f:
        N = int(f.readline())
        c = np.array([float(x) for x in f.readline().split(' ')])
        a = np.array([float(x) for x in f.readline().split(' ')])
        b = float(f.readline())
    return N, c, a, b


def write_output(res):
    with open(OUPUT, 'w') as f:
        f.write(' '.join([str(int(x)) for x in res.x]))


def odd_subsets(n):
    """
    Generate S - set of masks representing all subsets of N, |S| is odd
    """
    total_masks = 2**n
    result = []
    for i in range(total_masks):
        mask = []
        for j in range(n):
            if i & (1 << j) != 0:
                mask.append(1)
            else:
                mask.append(0)
        power = sum(mask)
        if power % 2 == 1:
            result.append((mask, power))
    return result


if __name__ == '__main__':
    N, c, a, b = read_input()

    # ADDING CONSTRAINTS
    odd_constraint = []
    odd_bound = []
    for mask, power in odd_subsets(N):
        constr = np.array(mask) * 2 - 1
        odd_constraint.append(constr)
        odd_bound.append(power - 1)
            
    constraints = LinearConstraint(
        A=[a] + odd_constraint, 
        ub=[b] + odd_bound,
        )

    integrality = np.ones(N)
    bounds = Bounds(0, 1)

    # CREATING SOLVER
    res = milp(
        c=-c,
        integrality=integrality,
        constraints=constraints,
        bounds=bounds,
        )

    # SAVE RESULTS IF OK
    if res.success:
        write_output(res)
