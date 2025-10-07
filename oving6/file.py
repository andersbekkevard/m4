# 4*((1/s**3)*(s+3)/(s+4))

from sympy import symbols, exp, laplace_transform, inverse_laplace_transform


s = symbols("s")
t = symbols("t")

expr = 4 * ((1 / s**3) * (s + 3) / (s + 4))

print(inverse_laplace_transform(expr, s, t))
