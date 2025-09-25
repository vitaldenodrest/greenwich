from sympy import integrate
from sympy.functions.elementary.complexes import conjugate
from sympy import symbols

x = symbols('x')

def L2_inner_product(a, b):
    
    def L2_inner_product_a_b(u, v):
        
        return integrate(u * conjugate(v), (x, 0, 1))
    
    return L2_inner_product_a_b

def norm

if __name__ == '__main__':
    
    a, b = 0, 1 # define domain
    print(L2_inner_product(0, 1)(x, 2*x))