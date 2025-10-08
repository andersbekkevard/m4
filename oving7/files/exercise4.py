import numpy as np
import matplotlib.pyplot as plt

# Butcher table:
# 0  | 0   0
# 0.5| 0.5 0
# ---|-------
#    | 0   1


def f(t, y):
    return np.exp(-(y**2))


t = 0
y = 1
h = 0.01

t_target = 1

while t < t_target:
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    y = y + h * k2
    t = t + h

print(f"y = {y:.4f}")
print(f"t = {t:.4f}")

# y = 1.2699
# t = 1.0000
