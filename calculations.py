from sympy import symbols, diff, integrate
from sympy.functions.elementary.complexes import conjugate
from sympy.printing.pretty.pretty import pretty

a_1, a_2, b_1, b_2, x, k = symbols('a_1 a_2 b_1 b_2 x k')

p_1 = a_1*x + b_1
p_2 = a_2*x + b_2

def inner_product(v, u):
    return integrate(v * conjugate(u), (x, 0 ,1))

def a(v, u):
    grad_v = diff(v, x)
    grad_u = diff(u, x)
    part_1 = inner_product(grad_v, grad_u)
    part_2 = inner_product(v, k**2 * u)
    return part_1 - part_2


# TEST
print(pretty(a(p_1, p_2)))