import sympy as sp

# 1D
# Continuous piecewise polynomials of order 1
drawing = f"""
    .____.________.
    x-1  x0      x1
    |____|________|
      hx-    hx+

"""
print(drawing)

# !!! Indicates a hypothesis (should be studied and considered)

# !!! Mode selection
mode_h = "2h" # 1h for h- = h+ or 2h for distinct values
mode_values = "real" # real or complex values

# Wavenumbers
k = sp.Symbol('k',
              #real=True, # !!!
              #positive=True, # !!!
              )
kh = sp.symbols('k^h')

# 1D variable
x = sp.symbols('x')

# 1D variational formulations over [a, b]
tau = sp.symbols('tau')
def a_G(u, v, a, b):
    D_u = sp.diff(u, x)
    D_v = sp.diff(v, x)
    return sp.integrate(D_u*D_v, (x, a, b)) - ( k**2 * sp.integrate(u*v, (x, a, b)) )
def L(u):
    return sp.diff(u, x, 2) + ( k**2 * u )
def a_GLS(u, v, a, b):
    return a_G(u, v, a, b) + ( tau * sp.integrate(L(u)*L(v), (x, a, b)) )

# h definitions
h_x_ = dict()
if mode_h == "1h":
    h_x_["-"] = sp.symbols('h_x')
    h_x_["+"] = h_x_["-"]
elif mode_h == "2h":
    h_x_["-"], h_x_["+"] = sp.symbols('h_x- h_x+')

# Grid sample
x_ = dict()
x_["-1"], x_["0"], x_["+1"] = -h_x_["-"], sp.sympify(0), +h_x_["+"]

# Shape functions
N_ = dict()
N_["-1"] = - x / h_x_["-"]
N_["0-"] = (x / h_x_["-"]) + 1
N_["0+"] = (- x / h_x_["+"]) + 1
N_["+1"] = x / h_x_["+"]

# Linear system coefficients (Galerkine)
A_G_ = dict()
A_G_["0, -1"] = a_G(N_["0-"], N_["-1"], x_["-1"], x_["0"])
A_G_["0, 0"] = a_G(N_["0-"], N_["0-"], x_["-1"], x_["0"]) + a_G(N_["0+"], N_["0+"], x_["0"], x_["+1"])
A_G_["0, +1"] = a_G(N_["0+"], N_["+1"], x_["0"], x_["+1"])
# Checkpoint: another result should be equal to lambda * A_G_ with lambda any scalar (coefficients of a homogeneous system)
print("Galerkine coefficients")
sp.pprint(A_G_["0, -1"])
sp.pprint(A_G_["0, 0"])
sp.pprint(A_G_["0, +1"])
print("\n")

"""
# Exact solution
u = sp.exp(1j * k * x)
U_ = dict()
U_["-1"] = u.subs(x, x_["-1"])
U_["0"] = u.subs(x, x_["0"])
U_["+1"] = u.subs(x, x_["+1"])
"""

# Numerical solution is supposedly
if mode_values == "complex":
    uh = sp.exp(1j * kh * x)
elif mode_values == "real":
    uh = sp.cos(kh * x)
Uh_ = dict()
Uh_["-1"] = uh.subs(x, x_["-1"])
Uh_["0"] = uh.subs(x, x_["0"])
Uh_["+1"] = uh.subs(x, x_["+1"])

# Solve the A_G * Uh = 0 Galerkine stencil
if mode_h == "1h":
    # for kh (possible in 1h)
    kh_G = sp.solve( (A_G_["0, -1"] * Uh_["-1"]) + (A_G_["0, 0"] * Uh_["0"]) + (A_G_["0, +1"] * Uh_["+1"]), kh )
    print("Galerkine dispersion (k^h compared to k)")
    print("k^h = ")
    sp.pprint(kh_G)
    # !!! Choose positive solution
    kh_G = kh_G[1]
elif mode_h == "2h":
    # for k (kh is non analytic in 2h)
    kh_G = sp.solve( (A_G_["0, -1"] * Uh_["-1"]) + (A_G_["0, 0"] * Uh_["0"]) + (A_G_["0, +1"] * Uh_["+1"]), k )
    print("Galerkine dispersion (k compared to kh)")
    print("k = ")
    sp.pprint(kh_G)
    # !!! Choose positive solution
    kh_G = kh_G[0]
print("\n")

# Linear system coefficients (GLS)
A_GLS_ = dict()
A_GLS_["0, -1"] = a_GLS(N_["0-"], N_["-1"], x_["-1"], x_["0"])
A_GLS_["0, 0"] = a_GLS(N_["0-"], N_["0-"], x_["-1"], x_["0"]) + a_GLS(N_["0+"], N_["0+"], x_["0"], x_["+1"])
A_GLS_["0, +1"] = a_GLS(N_["0+"], N_["+1"], x_["0"], x_["+1"])
# Checkpoint: another result should be equal to lambda * A_GLS_ with lambda any scalar (coefficients of a homogeneous system)
print("GLS coefficients")
sp.pprint(A_GLS_["0, -1"])
sp.pprint(A_GLS_["0, 0"])
sp.pprint(A_GLS_["0, +1"])
print("\n")

# Solve the A_GLS * Uh = 0 GLS stencil for a relation between k and kh
if mode_h == "1h":
    # for kh (possible in 1h)
    kh_GLS = sp.solve( (A_GLS_["0, -1"] * Uh_["-1"]) + (A_GLS_["0, 0"] * Uh_["0"]) + (A_GLS_["0, +1"] * Uh_["+1"]), kh )
    print("GLS dispersion (k^h compared to k)")
    print("k^h = ")
    sp.pprint(kh_GLS)
    # !!! Choose positive solution
    kh_GLS = kh_GLS[1]
elif mode_h == "2h":
    # for k (kh is non analytic in 2h)
    kh_GLS = sp.solve( (A_GLS_["0, -1"] * Uh_["-1"]) + (A_GLS_["0, 0"] * Uh_["0"]) + (A_GLS_["0, +1"] * Uh_["+1"]), k )
    print("GLS dispersion (k compared to kh)")
    print("k = ")
    sp.pprint(kh_GLS)
    # !!! Choose positive solution
    kh_GLS = kh_GLS[0]
print("\n")

# Find tau that yields the right wavenumber
tau_GLS = sp.solve( kh_GLS - k, tau )
print("tau_GLS = ")
sp.pprint(tau_GLS)
print("\n")