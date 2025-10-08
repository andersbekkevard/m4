def f(t, y):
    return -2 * t * y**2


t = 0
y = 1
h = 0.4

for i in range(0, 5):
    print("=" * 50)
    print(f"Step {i+1:2d}")
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    print(f"k1 = {k1:.6f}, k2 = {k2:.6f}, k3 = {k3:.6f}, k4 = {k4:.6f}")
    y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t = t + h
    print(f"Step {i+1:2d}: t = {t:.4f}, y = {y:.6f}")
