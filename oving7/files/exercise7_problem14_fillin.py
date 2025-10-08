# %% [markdown]
# Numerical solution of ODEs
# In this problem we will implement Euler's method, second order Taylor's method, and Heun's method, and use them to approximate the solution to the ODE,
# \begin{align*}
# 	y' = (1-2t) y, \quad y(0) = 1.
# \end{align*}
# The exact solution to this equation is $y(t) = \mathrm{e}^{t-t^2}$.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# define the derivatives of y

def d_y(t, y):
    # first derivative of y
    return 

def dd_y(t, y):
    # second derivative (for second order Taylor method)
    return 

# define y

def y(t):
    # y(t) (for error computations)
    return 

# %% [markdown]
# >***
# >a) Implement Euler's method, and compute an approximation of $y(1)$, using a step size equal to $0.5$.
# >***

# %%
# Define Euler's method
def Euler(y0, T, f, h):
    """
        Euler(y0, T, f, h)
    
    Use Euler's method to approximate the solution of the ODE (scalar or system) y'(t) = f(t,y)
    
    Input:
        y0 - initial value y(0)
        f  - definition of the right-hand-side function
        T  - simulation time (starting at t=0, ending at t=T)
        h  - time-step size (fixed)
    Output:
        y  - array containing all discrete solution values y0,y1,y2,...
        t  - array containing all discrete time instants 0,h,2h,...
    """
    ys = [y0] #array where all y_n will be stored
    ts = [0]  #array where all t_n will be stored

    #Fill in the rest here

    return (np.array(ts), np.array(ys))

# %% [markdown]
# >***
# >b) Do the same using Heun's method and the second order Taylor method. The second order Taylor method is given by
# >$$y_{n+1} = y_n + hf(t_n,y_n) + \frac{h^2}{2}f'(t_n,y_n),$$
# >where $f(t,y) = y'(t)$ and thus $f'(t,y) := \frac{\partial f}{\partial t}(t,y) = y''(t)$.
# >***

# %%
# Define Heun's method
def Heun(y0, T, f, h):
    """
        Heun(y0, T, f, h)
    
    Use Heun's method to approximate the solution of the ODE (scalar or system) y'(t) = f(t,y)
    
    Input:
        y0 - initial value y(0)
        f  - definition of the right-hand-side function
        T  - simulation time (starting at t=0, ending at t=T)
        h  - time-step size (fixed)
    Output:
        y  - array containing all discrete solution values y0,y1,y2,...
        t  - array containing all discrete time instants 0,h,2h,...
    """
    ys = [y0] #array where all y_n will be stored
    ts = [0]  #array where all t_n will be stored

    #Fill in the rest here

    return (np.array(ts), np.array(ys))

# %%
# Define the second-order Taylor method
def scalarTaylor2(y0, T, f, DfDt, h):
    """
        scalarTaylor2(y0, T, f, h)
    
    Use the second-order Taylor method to approximate the solution of the scalar ODE y'(t) = f(t,y)
    
    Input:
        y0   - initial value y(0)
        f    - definition of the right-hand-side function
        DfDt - definition of the total time derivative of f
        T    - simulation time (starting at t=0, ending at t=T)
        h    - time-step size (fixed)
    Output:
        y  - vector containing all discrete solution values y0,y1,y2,...
        t  - vector containing all discrete time instants 0,h,2h,...
    """
    ys = [y0] #array where all y_n will be stored
    ts = [0]  #array where all t_n will be stored
    while(ts[-1] < T):
        t, y = ts[-1], ys[-1]
        ys.append(y + h*f(t,y) + 0.5*h*h*DfDt(t,y))
        ts.append(t + h)
    return (np.array(ts), np.array(ys))

# %%
# compute the approximations using the different methods and h = 0.5

# %% [markdown]
# >***
# >c) We now want to approximate the convergence orders of these methods numerically. Recall that we defined the global error,
# \begin{align*}
# 	\epsilon_g := \max_i \lvert y(t_i) - y_i \rvert.
# \end{align*}
# >If we assume that $\epsilon_g(h) \approx M h^p$, for some $M > 0$, we have,
# \begin{align*}
# 	\log \bigg(\frac{\epsilon_g(h_1)}{\epsilon_g(h_2)} \bigg) \approx p \log \bigg(\frac{h_1}{h_2} \bigg).
# \end{align*}
# >Compute the global error of the methods from a)-c) using $h_1 = 10^{-2}$ and $h_2 = 10^{-3}$, where $t_i = i h, i = 1, \dots, \frac{1}{h}$. Use this to approximate the convergence order, $p$, for each of the three methods.
# >***

# %%
# approximate convergence order for each method

h1 = 1e-2

h2 = 1e-3

# Euler's method


# Heun's method


# second order Taylor's method
ts,ys = scalarTaylor2(1., 1., d_y, dd_y, h1)
e1 = np.linalg.norm(ys - y(ts), np.inf)

# %% [markdown]
# >***
# >d) We can also approximate the convergence order by plotting $\log(\epsilon_g(h)) = \log(M) + p \log(h)$ versus $\log(h)$, and inspecting the slope of the function.\\
# >Plot $\log(\epsilon_g(h))$ versus $\log(h)$ for $h = 10^{-2}, 10^{-3}, 10^{-4}$ for each of the three methods.
# >***

# %%
# plot log(error) versus log(h) 

# values of h
hs = [1e-2, 1e-3, 1e-4]

errors_Euler = []
errors_Heun = []
errors_Taylor2 = []
for h in hs:
    # Euler's method error

    # Heun's method error

    # second order Taylor method error


# plot of Euler's method errors


# plot of Heun's method errors

# plot of second order Taylor error
plt.plot(np.log(hs), np.log(errors_Taylor2), label="Taylor's method")

plt.legend()
