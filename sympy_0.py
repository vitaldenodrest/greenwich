from sympy import symbols
from sympy import integrate, pprint


ksi = symbols("ksi")

pprint(
    integrate(
        (
            (1-ksi)/2
        )**2,
        (ksi, -1, 1)
    )
       )

pprint(
    integrate(
        (
            (1+ksi)/2
        )**2,
        (ksi, -1, 1)
    )
       )

pprint(
    integrate(
        (
            (1+ksi)/2 * (1-ksi)/2
        ),
        (ksi, -1, 1)
    )
       )