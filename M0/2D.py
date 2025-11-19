import sympy as sp

# 2D
# Continuous piecewise bilinear functions
drawing = f"""
        _  .___._____.
        |  |   |     |
        |  |   |     |
    hy+ |  |   |     |
        |  |   |     |
        _  .___._____.
        |  |   |     |
    hy- |  |   |     |
        _  .___._____.
        
           |___|_____|
             hx-  hx+
             
"""
print(drawing)

# !!! Indicates a hypothesis (should be studied and considered)

# !!! Mode selection
mode_values = "real" # real or complex values

# Wavenumbers
k = sp.Symbol('k',
              #real=True, # !!!
              #positive=True, # !!!
              )
kh = sp.symbols('k^h')

# 2D coordinates
x = sp.symbols('x')
y = sp.symbols('y')

# 2D variational formulations over the [a, b] x [c, d]
tau = sp.symbols('tau')
def M(u, v, a, b, c, d):
    """_Mass term._
    """
    return
def M(u, v, a, b, c, d):
    """_Stiffness term._
    """
    return
def a_G(u, v, a, b, c, d):
    """_Galerkine._
    """
    return
def a_LS(u, v, a, b, c, d):
    """_Least Squares._
    """
    return
def a_GLS(u, v, a, b, c, d):
    """_Galerkine Least Squares._
    """
    return

# Mesh dimensions