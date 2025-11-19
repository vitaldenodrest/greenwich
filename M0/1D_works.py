import sympy as sp

# 1D
# Continuous piecewise linear functions
drawing = f"""
    .____.________.
    x-1  x0      x1
    |____|________|
      hx-    hx+

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
h_x_["-"], h_x_["+"] = sp.symbols('h_x- h_x+')
h_x = sp.symbols('h_x')

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
# for k (kh is non analytic in k when hx+ != hx-)
k2_G = sp.solve( (A_G_["0, -1"] * Uh_["-1"]) + (A_G_["0, 0"] * Uh_["0"]) + (A_G_["0, +1"] * Uh_["+1"]), k**2 )
print("Galerkine dispersion")
print("k^2 = ")
sp.pprint(k2_G)
k2_G = k2_G[0]
print("\n")
# Check for consistency with hx+ = hx-
k2_G_equal = k2_G.subs(h_x_["+"], h_x).subs(h_x_["-"], h_x)
print("Galerkine dispersion with hx+ = hx-")
print("k^2 = ")
sp.pprint(k2_G_equal)
print("\n")
# Now k^h is analytic
khh_G_equal = sp.solve(k**2 - k2_G_equal, kh*h_x)
print("Galerkine dispersion with hx+ = hx-")
print("kh*h = ")
sp.pprint(khh_G_equal)
print("\n")


# Linear system coefficients (GLS)
A_GLS_ = dict()
A_GLS_["0, -1"] = a_GLS(N_["0-"], N_["-1"], x_["-1"], x_["0"])
A_GLS_["0, 0"] = a_GLS(N_["0-"], N_["0-"], x_["-1"], x_["0"]) + a_GLS(N_["0+"], N_["0+"], x_["0"], x_["+1"])
A_GLS_["0, +1"] = a_GLS(N_["0+"], N_["+1"], x_["0"], x_["+1"])

print("--- GLS Optimization ---")

# 1. Construct the Stencil Equation
stencil_eq = (A_GLS_["0, -1"] * Uh_["-1"]) + \
             (A_GLS_["0, 0"]  * Uh_["0"]) + \
             (A_GLS_["0, +1"] * Uh_["+1"])

# 2. Enforce "Exact Transport" (No pollution error)
# We want the numerical wavenumber (kh) to match the physical wavenumber (k) exactly.
# Substitute kh -> k
stencil_eq_ideal = stencil_eq.subs(kh, k)

# 3. Solve for tau directly
tau_solutions = sp.solve(stencil_eq_ideal, tau)

print(f"Number of tau solutions found: {len(tau_solutions)}")
tau_opt = tau_solutions[0] # Usually there is only one unique solution for linear tau

print("Optimal tau (general):")
sp.pprint(tau_opt) # Warning: This might be huge for non-uniform grids

# 4. Simplify and Check Consistency for Uniform Grid (hx+ = hx- = h)
print("\nChecking consistency for uniform grid (hx+ = hx- = h)...")
tau_opt_uniform = tau_opt.subs(h_x_["+"], h_x).subs(h_x_["-"], h_x)
tau_opt_uniform = sp.simplify(tau_opt_uniform)

print("Optimal tau (uniform):")
sp.pprint(tau_opt_uniform)

# 5. Check Taylor Expansion (limit as h -> 0)
# Standard GLS usually scales with h^2 * constant in the small h limit
print("\nSeries expansion of tau (uniform) around h=0:")
tau_series = sp.series(tau_opt_uniform, h_x, 0, 4)
sp.pprint(tau_series)