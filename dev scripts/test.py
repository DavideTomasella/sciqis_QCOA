# In[] quantum stochastic differential equation (qsde) for the intracavity field
import numpy as np
from scipy.integrate import odeint
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
def model(vars, t, a, b, c, d, e, f):
    x, y, z = vars
    dxdt = -1j*5*x+a*x +(a+1)*x.conjugate()+ f*y.conjugate()*z
    dydt = b*y + f*x.conjugate()*z+d
    dzdt = c*z + f*x.conjugate()*y.conjugate()+e
    return [dxdt, dydt, dzdt]

# Initial conditions
initial_conditions = [0, 0, 1j*0]
t = np.linspace(0, 10, 1000)  # Time from 0 to 10

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
# %%
