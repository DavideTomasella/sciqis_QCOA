"""
Numerical solution of the Lindbladian steady state equation for an optical cavity spectral response
Author: D. Tomasella

"""
# In[]
import numpy as np
from qutip import *
from numba import njit

# still using:
# resonant pump
# rotating wave approximation (only for the sideband)
# classical approximation for the pump
# undepleted pump approximation

#NOTE: incoupling = 1j*np.sqrt(kappa_ext1_s)*alpha_in*(a.dag()-a)
#      if I want a non classical pump when using multiple fields or without the rotating wave approximation, 
#      I need alpha_in*(a.dag()np.exp(1j*delta_omega*t)-a*np.exp(-1j*delta_omega*t))
#      thus the hamiltonian becomes time dependent!

def reflectivity_ss_optical_cavity(omega, omega_in1, kappa, kappa_ext1, N=10):
    """
    Calculate the reflectivity spectrum of an optical cavity with the Master Equation Solver.
    ```
            | alpha_in - sqrt(kappa_ext1) * <a> |^2
        R = | ----------------------------------|     with <a> = expectation of the cavity field
            |             alpha_in              |
    ```

    Parameters
    ----------
    omega : float
        The cavity resonance frequency.
    omega_in1 : float or np.ndarray
        The input field frequency, i.e., the frequency we are probing the response at.
    kappa : float
        The cavity decay rate.
    kappa_ext1 : float
        The external coupling rate for the input field.
    N : int, optional
        The number of modes for the cavity field. The default is 10.

    Returns
    -------
    R : float or np.ndarray
        The reflectivity of the cavity.
    """
    alpha_in1 = 1
    alpha_out1 = np.zeros_like(omega_in1, np.complex128)
    for i,o_in in enumerate(omega_in1):
        alpha_out1[i] = alpha_in1 - np.sqrt(kappa_ext1) * get_steady_state_field_optical_cavity(omega, o_in, kappa, kappa_ext1, alpha_in1, N=N)
    
    return np.abs(alpha_out1/alpha_in1) ** 2

def transmissivity_ss_optical_cavity(omega, omega_in1, kappa, kappa_ext1, kappa_ext2, N=10):
    """
    Calculate the transmissivity spectrum of an optical cavity with the Master Equation Solver.
    ```
            | sqrt(kappa_ext2) * <a> |^2
        T = | -----------------------|     with <a> = expectation of the cavity field
            |        alpha_in        |
    ```

    Parameters
    ----------
    omega : float
        The cavity resonance frequency.
    omega_in1 : float or np.ndarray
        The input field frequency, i.e., the frequency we are probing the response at.
    kappa : float
        The cavity decay rate.
    kappa_ext1 : float
        The external coupling rate for the input field.
    kappa_ext2 : float
        The external coupling rate for the transmitted field (port 2).
    alpha_in : complex
        The input field amplitude outside the cavity.
    N : int, optional
        The number of modes for the cavity field. The default is 10.

    Returns
    -------
    T : float or np.ndarray
        The transmissivity of the cavity.
    """    
    alpha_in1 = 1
    alpha_out2 = np.zeros_like(omega_in1, np.complex128)
    for i,o_in in enumerate(omega_in1):
        alpha_out2[i] = np.sqrt(kappa_ext2) * get_steady_state_field_optical_cavity(omega, o_in, kappa, kappa_ext1, alpha_in1, N=N)
    
    return np.abs(alpha_out2/alpha_in1) ** 2


def get_steady_state_field_optical_cavity(omega, omega_in1, kappa, kappa_ext1, alpha_in, calculate_time_evolution=False, N=10):
    """
    Calculate the steady state solution cavity field for an optical cavity with a single standing wave given an input field alpha_in.
    The cavity is described by the Hamiltonian in the rotating wave approximation and the Landbladian collapse operators:
    ```
    H = (omega-omega_in1)*a†a + 1j*sqrt(kappa_ext1)*alpha_in*(a†-a)
    L = sqrt(kappa)*a
    ```

    Parameters
    ----------
    omega : float
        The cavity resonance frequency.
    omega_in1 : float or np.ndarray
        The input field frequency, i.e., the frequency we are probing the response at.
    kappa : float
        The cavity decay rate.
    kappa_ext1 : float
        The external coupling rate for the input field.
    alpha_in : complex
        The input field amplitude outside the cavity.
    calculate_time_evolution : bool, optional
        Whether to calculate the time evolution of the cavity field with mesolver() or 
        the steady state solution with steadystate() from qutip library.
    N : int, optional
        The number of modes for the cavity field. The default is 10.

    Returns
    -------
    a_ss : complex
        Return the steady state solution for the field inside the cavity
    """
    if kappa_ext1 > kappa:
        raise ValueError("The external coupling rate must be smaller than the total cavity decay rate.")
    if abs(alpha_in)**2 > N/2:
        raise ValueError("Warning: The input field is too large for the given Hilbert space dimension.")

    # init fields
    a = destroy(N)
    num_a = a.dag()*a

    # Rotating wave approximation with the input field that correspond to the frequency we are probing
    delta = omega-omega_in1
    free_evolution = delta*num_a
    incoupling_fields = 1j*np.sqrt(kappa_ext1)*alpha_in*(a.dag()-a)
    decay_channel_a = np.sqrt(kappa)*a

    Hamiltonian = free_evolution + incoupling_fields
    collapse_operators = [decay_channel_a]

    if calculate_time_evolution:
        # time evolution, the time array length considers the decay rate of the cavity to know when we reach the staedy state
        t = np.linspace(0, 12/kappa, 120)
        # init vacuum state
        rho_0 = tensor(basis(N,0))
        # solve time evolution with master equation
        result = mesolve(Hamiltonian, rho_0, t, collapse_operators, [a,num_a])
        a_me, num_a_me = result.expect
        if False:
            # You have to check that the time evolution is converging to the steady state
            import matplotlib.pyplot as plt
            plt.plot(t, num_a_me)
            plt.show()
        a_ss = a_me[-1]
        num_a_ss = num_a_me[-1]
    else:
        # calculate the steady state solution of the Lindbladian problem and the expectations
        rho_ss = steadystate(Hamiltonian, collapse_operators)
        a_ss = expect(a, rho_ss)
        num_a_ss = expect(num_a, rho_ss)
    print("Steady state cavity field: %.4e photons" % num_a_ss)
    return a_ss #,num_a_ss


if __name__=="__main__":
    import matplotlib.pyplot as plt
    def get_axis_values(values, n=7):
        return np.linspace(min(values), max(values), n), ["%.2f"%(i/1e6) for i in np.linspace(min(values), max(values), n)]
    
    # Init parameters
    lambda_to_omega = lambda l: 2 * np.pi * 3e8 / l
    omega = lambda_to_omega(1550e-9)
    omega_in1 = omega + np.linspace(-1e7,1e7,101)
    kappa_ext1 = 1e6
    kappa_ext2 = 1e6
    kappa = kappa_ext1 + kappa_ext2 + 1e6
    from time import time
    start = time()
    r=reflectivity_ss_optical_cavity(omega, omega_in1, kappa, kappa_ext1)
    t=transmissivity_ss_optical_cavity(omega, omega_in1, kappa, kappa_ext1, kappa_ext2)
    print(time()-start)

    plt.plot(omega_in1.T-omega, r.T, "--",label='Reflectivity')
    plt.plot(omega_in1.T-omega, t.T, label='Transmissivity')
    plt.ylim(-0.1,2.1)
    plt.xticks(*get_axis_values(omega_in1.T-omega))
    plt.xlabel("Detuning frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.legend()
    plt.grid()
    plt.show()