import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return t * np.exp(-y)

def rk4_method(t, y, h):
    """Standard RK4 method"""
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + 3*h/4, y + 3*h*k2/4)
    k4 = f(t + h, y + h*(2*k1 + 3*k2 + 4*k3)/9)
    return y + h*(7*k1/24 + k2/4 + k3/3 + k4/8)

def rk3_method(t, y, h):
    """Standard RK3 method"""
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + 3*h/4, y + 3*h*k2/4)
    return y + h*(7*k1/24 + k2/4 + k3/3)

# Parameters
T = 2.0
y0 = 0.0
h = 0.25

# RK4 solution
t = 0.0
y = y0
ys_rk4 = [y0]
ts_rk4 = [t]

while t + h <= T:
    y = rk4_method(t, y, h)
    t += h
    ys_rk4.append(y)
    ts_rk4.append(t)

# RK3 solution
t = 0.0
y = y0
ys_rk3 = [y0]
ts_rk3 = [t]

while t + h <= T:
    y = rk3_method(t, y, h)
    t += h
    ys_rk3.append(y)
    ts_rk3.append(t)

# Compare first step
rk4_first_step = ys_rk4[1]
rk3_first_step = ys_rk3[1]
error_first_step = abs(rk4_first_step - rk3_first_step)

print(f"RK4 first step: t={ts_rk4[1]:.3f}, y={rk4_first_step:.6f}")
print(f"RK3 first step: t={ts_rk3[1]:.3f}, y={rk3_first_step:.6f}")
print(f"Error in first step: {error_first_step:.6f}")

# Plot both solutions
plt.figure(figsize=(10, 6))
plt.plot(ts_rk4, ys_rk4, 'o-', label='RK4', markersize=4)
plt.plot(ts_rk3, ys_rk3, 's-', label='RK3', markersize=4)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparison of RK4 and RK3 Methods')
plt.legend()
plt.grid(True)
plt.show()