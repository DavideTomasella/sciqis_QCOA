# In[] quantum stochastic differential equation (qsde) for the intracavity field
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from odeintw import odeintw

def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z

# Parameters
a, b, c, d, e, f = -1.1, -10, -2+1j*10, 0.3, 0.2,100  # Example values

# System of equations
def model(vars, t, *args):
    x, y, z = vars
    a, b, c, d, e, f = args
    dxdt = -1j*5*x+a*x +(a+1)*x.conjugate()+ f*y.conjugate()*z
    dydt = b*y + f*x.conjugate()*z+d
    dzdt = c*z + f*x.conjugate()*y.conjugate()+e
    return dxdt, dydt, dzdt

# Initial conditions
initial_conditions = np.complex128([0, 0, 0])
t = np.linspace(0, 12, 1200)  # Time from 0 to 10

# Solving the system
solution, infodict = odeintw(model, initial_conditions, t, args=(a, b, c, d, e, f), full_output=True)

print(infodict["message"])
print(solution[-1, :])
# Plotting results
plt.plot(t, solution[:, 0], label='x(t)')
plt.plot(t, solution[:, 1], label='y(t)')
plt.plot(t, solution[:, 2], label='z(t)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Values of x, y, z')
plt.title('Solution of the system of equations')
plt.show()
print(solution[-1, :])

def model2(vars, t, *args):
    x1,x2, y1, y2, z1, z2 = vars
    x=x1+1j*x2
    y=y1+1j*y2
    z=z1+1j*z2
    a, b, c, d, e, f = args
    dxdt = -1j*5*x+a*x +(a+1)*x.conjugate()+ f*y.conjugate()*z
    dydt = b*y + f*x.conjugate()*z+d
    dzdt = c*z + f*x.conjugate()*y.conjugate()+e
    return [np.float64(dxdt.real), np.float64(dydt.real), np.float64(dzdt.real),
            np.float64(dxdt.imag), np.float64(dydt.imag), np.float64(dzdt.imag)]
initial_guess = np.float64([0, 0, 0,0,0,0])
eq = fsolve(model2, initial_guess, args=(0, a, b, c, d, e, f))
steady_state = eq[0]+1j*eq[1], eq[2]+1j*eq[3], eq[4]+1j*eq[5]
print(steady_state)


def fsolvew(func, x0, t, args=(), **kwargs):
    """An fsolve-like function for complex valued functions."""

    # Make sure x0 is a numpy array of type np.complex128.
    x0 = np.array(x0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        f = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(f, dtype=np.complex128).view(np.float64)

    result = fsolve(realfunc, x0.view(np.float64), args=(t,*args), **kwargs)

    return result.view(np.complex128)

steady_state2 = fsolvew(model, initial_conditions, t=0, args=(a, b, c, d, e, f))
print(steady_state2)
# %%
