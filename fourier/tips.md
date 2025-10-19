
---

## (a)

$$f(x) = \begin{cases}
0, & -\pi < x < 0 \text{ or } \frac{\pi}{2} < x \le \pi,\\
x, & 0 \le x \le \frac{\pi}{2}.
\end{cases}$$

### Structure and tricks

1. **Support limited to one small interval**
   The function is zero almost everywhere on $[-π, π]$ except $[0, π/2]$.
   That means:
   $$a_n = \frac{1}{\pi}\int_0^{\pi/2} x\cos(nx)\,dx, \quad b_n = \frac{1}{\pi}\int_0^{\pi/2} x\sin(nx)\,dx$$
   The rest of the interval contributes nothing.
   You can reuse the *same base integrals* for $x\cos(nx)$ and $x\sin(nx)$ repeatedly (as I showed earlier).

2. **No symmetry ⇒ all terms exist**
   It's neither even nor odd, so both $a_n$ and $b_n$ are nonzero.
   But you can reuse the "triangle pulse" idea: this is effectively a shifted and truncated ramp.

3. **Reuse between problems**
   (a) is literally a building block for (b): you can think of (b) as extending this to a symmetric sawtooth around zero.

---

## (b)

$$f(x) = \begin{cases}
0, & -\pi < x < 0,\\
x, & 0 < x < \frac{\pi}{2},\\
\pi - x, & \frac{\pi}{2} < x \le \pi.
\end{cases}$$

### Structure and tricks

1. **Piecewise linear and continuous**
   $f(x)$ is continuous and rises from 0 to $\pi/2$ at $x=\pi/2$, then decreases symmetrically to 0 at $x=\pi$.
   This makes it a **half-period of a triangular wave**.

2. **Effective symmetry:**
   Extend $f(x)$ periodically to $[-\pi, \pi]$ and check that
   $f(-x)\neq \pm f(x)$, but $f(x+\pi) = f(x)$.
   So the function is **periodic and continuous**, but not even/odd — still, it's composed of *two ramps* that can reuse integrals from part (a).

3. **Reuse trick:**
   Split as
   $$f(x) = x \cdot 1_{(0, \pi/2)} + (\pi - x) \cdot 1_{(\pi/2, \pi)}$$
   Then use the linearity of Fourier coefficients to combine two "ramp" integrals.

4. **Derivative shortcut**
   $f'(x)$ is a piecewise-constant square wave:
   $$f'(x) = \begin{cases}
   0, & x<0,\\
   1, & 0<x<\pi/2,\\
   -1, & \pi/2<x<\pi.
   \end{cases}$$
   Since you know the Fourier series of a square wave (sine-only terms ∝ 1/n), you can integrate that series term-by-term to get the series for $f(x)$.
   That's a huge shortcut.

---

## (c)

$$f(x) = \begin{cases}
-\pi - x, & -\pi < x < -\frac{\pi}{2},\\
x, & -\frac{\pi}{2} < x < \frac{\pi}{2},\\
\pi - x, & \frac{\pi}{2} < x \le \pi.
\end{cases}$$

### Structure and tricks

1. **Perfect triangular symmetry**
   This one *is even*: $f(-x) = f(x)$.
   That means **only cosine terms** survive:
   $$b_n = 0 \quad \text{for all } n.$$
   So you only need $a_0$ and $a_n$.

2. **Reuse from (b)**
   Notice that for $x>0$, the definition is the same as in (b).
   The negative side is mirrored instead of flattened.
   Therefore, you can take the result from (b), reflect it, and drop sine terms.

3. **Derivative shortcut again**
   The derivative is a square wave symmetric about zero:
   $$f'(x) = \begin{cases}
   -1, & -\pi < x < -\pi/2,\\
   1, & -\pi/2 < x < \pi/2,\\
   -1, & \pi/2 < x < \pi.
   \end{cases}$$
   This is a **standard even square wave**, which has only sine coefficients for $f'(x)$.
   Integrate those to get the cosine-only Fourier coefficients for $f(x)$.

---

## Summary of applicable tricks

| Function | Symmetry                   | Trick                                                                 | Reuse                                |
| -------- | -------------------------- | --------------------------------------------------------------------- | ------------------------------------ |
| (a)      | None                       | Integrate only over support, reuse $\int x\sin(nx)$ and $\int x\cos(nx)$            | Base for others                      |
| (b)      | None, but piecewise linear | Derivative is square wave (use known Fourier series of sign function) | Combine two "ramps"                  |
| (c)      | Even                       | Drop all sine terms; derivative gives square wave with sine series    | Reuse from (b), symmetry halves work |

---

If you plan to implement in Python, (b) and (c) benefit *massively* from defining `fprime(x)` and integrating its Fourier series symbolically or numerically; (a) is better done directly.

Would you like me to show the explicit shortcut for (b) using the derivative square wave (so you don't need to integrate piecewise)?