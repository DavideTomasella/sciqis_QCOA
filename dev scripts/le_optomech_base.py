"""
WIP
Author: D. Tomasella

"""
# In[]
import numpy as np
from qutip import *
import sympy as sp
def reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, is_sideband_stokes=True, N=10, N_m=10):
    """
    Calculate the reflectivity spectrum of an optomechanical cavity with the Master Equation Solver.
    ```
            | alpha_in1_s - sqrt(kappa_ext1_s) * <a> |^2
        R = | ----------------------------------------|     with <a> = expectation of the cavity field
            |               alpha_in1_s               |
    ```

    Parameters
    ----------
    omega_in1_s: (float or np.ndarray)
        frequency of the stokes input field [Hz]
    kappa_ext1_s:
        external loss rate of the cavity stokes field = coupling of stokes port 1 field inside the cavity [Hz]
    omega_s: (float)
        frequency of the stokes cavity field [Hz]
    kappa_s: (float)
        total loss rate of the cavity stokes field [Hz]
    omega_p: (float)
        frequency of the pump cavity field [Hz]
    alpha_p: (complex)
        complex amplitude of the pump cavity field
    G_0: (float)
        single-photon optomechanical coupling strength [Hz]
    Omega_m: (float)
        frequency of the mechanical cavity field [Hz]
    gamma_m: (float)
        total loss rate of the mechanical cavity field [Hz]
    is_sideband_stokes : bool, optional
        If True, the sideband is the stokes field, otherwise it is the anti-stokes field
    N : int, optional
        The number of modes for the cavity field. The default is 10.
    Nm : int, optional
        The number of modes for the mechanical field. The default is 10.

    Returns
    -------
    reflectivity: (float or np.ndarray)
        reflectivity of the stokes field of the cavity
    """
    alpha_in1_s = np.sqrt(kappa_ext1_s)
    alpha_out1_s = np.zeros_like(omega_in1_s, np.complex128)
    for i,o_in in enumerate(omega_in1_s):
        alpha_out1_s[i] = alpha_in1_s - np.sqrt(kappa_ext1_s) * get_steady_state_field_optomechanical_cavity(omega_s-o_in, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, 
                                                                                                             (omega_p-o_in)+(-1 if is_sideband_stokes else 1)*Omega_m, gamma_m,
                                                                                                             is_sideband_stokes=is_sideband_stokes, N=N, N_m=N_m, calculate_time_evolution=True)

    return np.abs(alpha_out1_s/alpha_in1_s) ** 2


def transmittivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, is_sideband_stokes=True, N=10, N_m=10):
    """
    Calculate the transmissivity spectrum of an optomechanical cavity with the Master Equation Solver.
    ```
            | sqrt(kappa_ext2) * <a> |^2
        T = | -----------------------|     with <a> = expectation of the cavity field
            |        alpha_in        |
    ```

    Parameters
    ----------
    omega_in1_s: (float or np.ndarray)
        frequency of the stokes input field [Hz]
    kappa_ext1_s:
        external loss rate of the cavity stokes field = coupling of stokes port 1 field inside the cavity [Hz]
    kappa_ext2_s:
        external loss rate of the cavity stokes field = coupling of stokes port 2 field inside the cavity [Hz]
    omega_s: (float)
        frequency of the stokes cavity field [Hz]
    kappa_s: (float)
        total loss rate of the cavity stokes field [Hz]
    omega_p: (float)
        frequency of the pump cavity field [Hz]
    alpha_p: (complex)
        complex amplitude of the pump cavity field
    G_0: (float)
        single-photon optomechanical coupling strength [Hz]
    Omega_m: (float)
        frequency of the mechanical cavity field [Hz]
    gamma_m: (float)
        total loss rate of the mechanical cavity field [Hz]
    is_sideband_stokes : bool, optional
        If True, the sideband is the stokes field, otherwise it is the anti-stokes field
    N : int, optional
        The number of modes for the cavity field. The default is 10.
    Nm : int, optional
        The number of modes for the mechanical field. The default is 10.

    Returns
    -------
    transmissivity: (float or np.ndarray)
        transmissivity of the stokes field of the cavity
    """
    alpha_in1_s = np.sqrt(kappa_ext1_s)
    alpha_out1_s = np.zeros_like(omega_in1_s, np.complex128)
    for i,o_in in enumerate(omega_in1_s):
        alpha_out1_s[i] = np.sqrt(kappa_ext2_s) * get_steady_state_field_optomechanical_cavity(omega_s-o_in, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, 
                                                                                               (omega_p-o_in)+(-1 if is_sideband_stokes else 1)*Omega_m, gamma_m, 
                                                                                               is_sideband_stokes=is_sideband_stokes, N=N, N_m=N_m, calculate_time_evolution=True)

    return np.abs(alpha_out1_s/alpha_in1_s) ** 2


def get_steady_state_field_optomechanical_cavity(delta_s, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, delta_m, gamma_m, 
                                                 is_sideband_stokes=True, N=10, N_m=10, calculate_time_evolution=True):
    """
    Calculate the steady state field of an optomechanical cavity with the Master Equation Solver.
    ```
        H = delta_s * a.dag() * a + delta_m * b.dag() * b - G_0 * abs(alpha_p) * (a.dag() * b + a * b.dag()) + 1j * sqrt(kappa_ext1_s) * alpha_in1_s * (a.dag() - a)
                                                            G_0 * abs(alpha_p) * (a.dag() * b.dag() + a * b)
        L = [sqrt(kappa_s) * a, sqrt(gamma_m) * b]
    ```
    We are using the rotating wave approximation with the input field that correspond to the frequency we are probing
    and for the mechanical mode corresponding to the difference between sideband and pump frequency.

    Parameters
    ----------
    delta_s: float
        detuning of the cavity stokes field [Hz]
    kappa_ext1_s:
        external loss rate of the cavity stokes field = coupling of stokes port 1 field inside the cavity [Hz]
    kappa_s: float
        total loss rate of the cavity stokes field [Hz]
    alpha_in1_s: complex
        complex amplitude of the stokes input field [sqrt(Hz)]
        it is the "amplitude" of the travelling field derived from the input power in photon/s
    alpha_p: complex
        complex amplitude of the pump cavity field
        This is a standing wave, so the amplitude is the sqrt of the number of photons in the cavity
    G_0: float
        single-photon optomechanical coupling strength [Hz]
    delta_m: float
        detuning of the mechanical cavity field [Hz]
    gamma_m: float
        total loss rate of the mechanical cavity field [Hz]
    is_sideband_stokes : bool, optional
        If True, the sideband is the stokes field, otherwise it is the anti-stokes field
    N : int, optional
        The number of modes for the cavity field. The default is 10.
    N_m : int, optional
        The number of modes for the mechanical field. The default is 10.
    calculate_time_evolution : bool, optional
        If True, the time evolution is calculated, otherwise the steady state is calculated. The default is True.
        You can also display the Monte Carlo vs Master Equation time evolution.

    Returns
    -------
    a_s: complex
        complex amplitude of the stokes cavity field (steady state solution)
    """

    if kappa_ext1_s > kappa_s:
        raise ValueError("The external coupling rate must be smaller than the total cavity decay rate.")
    if abs(alpha_in1_s)**2/kappa_s > N/2 or abs(alpha_in1_s)**2/kappa_s*G_0*abs(alpha_p)/gamma_m > N_m/2:
        raise ValueError("Warning: The input field is too large for the given Hilbert space dimension.")

    # sytem of equations
    delta_p = 0
    def model(vars : list[np.complex128], t):
        b_m, a_s, a_as, a_p = vars
        b_m_dt = 1j*delta_m*b_m - gamma_m/2*b_m + 1j*G_0*abs(alpha_p)*(a_p*a_s.conjugate() - a_p.conjugate()*a_as)
        a_s_dt = 1j*(delta_s-delta_m-delta_p)*a_s - kappa_s/2*a_s + 1j*G_0*abs(alpha_p)*b_m.conjugate()*a_p + np.sqrt(kappa_ext1_s)*alpha_in1_s
    a = tensor(destroy(N), qeye(N_m))
    b = tensor(qeye(N), destroy(N_m))
    num_a = a.dag()*a
    num_b = b.dag()*b

    # Rotating wave approximation with the input field that correspond to the frequency we are probing
    # and for the mechanical mode, to the difference between sideband and mechanical frequency
    # delta_s = omega-omega_in1
    # delta_m = omega_m - (1 if is_sideband_stokes else -1)*(omega_p-omega_in1_s)
    print(delta_s, delta_m)
    free_evolution = delta_s*num_a + delta_m*num_b
    incoupling_fields = 1j*np.sqrt(kappa_ext1_s)*alpha_in1_s*(a.dag()-a)
    #interaction = GO*abs(alpha_p)*(a.dag()+a)*(b.dag()+b)
    interaction = -G_0*abs(alpha_p)*(a.dag()*b.dag() + a*b if is_sideband_stokes else a.dag()*b + a*b.dag())
    decay_channel_a = np.sqrt(kappa_s)*a
    decay_channel_b = np.sqrt(gamma_m)*b

    Hamiltonian = free_evolution + incoupling_fields + interaction
    collapse_operators = [decay_channel_a,decay_channel_b]

    if calculate_time_evolution:
        # time evolution, the time array length considers the decay rate of the cavity to know when we reach the staedy state
        t = np.linspace(0, max(15/kappa_s,15/gamma_m), 1200)
        # init vacuum state
        rho_0 = tensor(coherent(N,2e-4),coherent(N_m,5e-4))
        # solve time evolution with master equation
        result = mesolve(Hamiltonian, rho_0, t, collapse_operators, [a,num_a,b,num_b])
        a_me, num_a_me, b_me, num_b_me = result.expect
        if False:
            result_mc = mcsolve(Hamiltonian, rho_0, t, collapse_operators, [a,num_a,b,num_b], ntraj=10)
            a_mc, num_a_mc, b_mc, num_b_mc = result_mc.expect
            #rho_ss = steadystate(Hamiltonian, collapse_operators)
            #num_a_ss = expect(num_a, rho_ss)
            #num_b_ss = expect(num_b, rho_ss)
            # You have to check that the time evolution is converging to the steady state
            import matplotlib.pyplot as plt
            plt.plot(t, num_a_me)
            plt.plot(t, num_b_me)
            plt.plot(t, num_a_mc)
            plt.plot(t, num_b_mc)
            #plt.plot(t, [num_a_ss]*len(t))
            #plt.plot(t, [num_b_ss]*len(t))
            plt.show()
        a_ss = np.mean(a_me[-10:])
        num_a_ss = np.mean(num_a_me[-10:])
        b_ss = np.mean(b_me[-10:])
        num_b_ss = np.mean(num_b_me[-10:])
    else:
        # calculate the steady state solution of the Lindbladian problem and the expectations
        rho_ss = steadystate(Hamiltonian, collapse_operators)
        a_ss = expect(a, rho_ss)
        num_a_ss = expect(num_a, rho_ss)
        b_ss = expect(b, rho_ss)
        num_b_ss = expect(num_b, rho_ss)
    print("Steady state cavity field: %.4e photons and %.4e phonons" % (num_a_ss, num_b_ss))
    return a_ss #,num_a_ss, b_ss, num_b_ss


if __name__=="__main__":
    import matplotlib.pyplot as plt
    def get_axis_values(values, n=5):
        return np.linspace(min(values), max(values), n), ["%.4f"%(i/1e9) for i in np.linspace(min(values), max(values), n)]
    # Test the analytical model
    is_sideband_stokes = True
    lambda_to_omega = lambda l: 2 * np.pi * 3e8 / l
    kappa_ext1_s = 1e6
    kappa_ext2_s = 1e6
    kappa_s = kappa_ext1_s + kappa_ext2_s + 1e6
    omega_p = lambda_to_omega(1550e-9)
    omega_s = omega_p + (-1 if is_sideband_stokes else 1) * 12.0008e9 #+ np.linspace(-8e6, 8e6, 10).reshape(-1,1)
    omega_in1_s = omega_s + np.linspace(-1e7, 1e7, 101)
    alpha_p = 7e3*(1 if is_sideband_stokes else 3) #* np.linspace(0,1.2,6).reshape(-1,1)
    G_0 = 100
    Omega_m = 12e9
    gamma_m = 1e6

    r=reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, 
                               is_sideband_stokes, N=5, N_m=5)
    t=transmittivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, 
                                 is_sideband_stokes, N=5, N_m=5)

    plt.plot(omega_in1_s.T-omega_p, r.T, "--",label='Reflectivity')
    plt.plot(omega_in1_s.T-omega_p, t.T, label='Transmissivity')
    plt.ylim(-0.1,2.1)
    plt.xticks(*get_axis_values(omega_in1_s.T-omega_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.grid()
    plt.show()