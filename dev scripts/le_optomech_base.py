"""
WIP
Author: D. Tomasella

"""
# In[]
import numpy as np
from qutip import *
import sympy as sp
from odeintw import odeintw
from scipy.optimize import fsolve

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
        a_s, _ = get_steady_state_field_optomechanical_cavity(omega_s-o_in, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, 
                                                              (omega_p-o_in)+(-1 if is_sideband_stokes else 1)*Omega_m, gamma_m,
                                                              is_sideband_stokes=is_sideband_stokes, N=N, N_m=N_m, calculate_time_evolution=False)
        alpha_out1_s[i] = alpha_in1_s - np.sqrt(kappa_ext1_s) * a_s
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
    a_p = np.zeros_like(omega_in1_s, np.complex128)
    for i,o_in in enumerate(omega_in1_s):
        a_s, a_p[i] = get_steady_state_field_optomechanical_cavity(omega_s-o_in, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, 
                                                                   (omega_p-o_in)+(-1 if is_sideband_stokes else 1)*Omega_m, gamma_m, 
                                                                   is_sideband_stokes=is_sideband_stokes, N=N, N_m=N_m, calculate_time_evolution=False)
        alpha_out1_s[i] = np.sqrt(kappa_ext2_s) * a_s
    return np.abs(alpha_out1_s/alpha_in1_s) ** 2, a_p


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

    #if kappa_ext1_s > kappa_s:
    #    raise ValueError("The external coupling rate must be smaller than the total cavity decay rate.")
    #if abs(alpha_in1_s)**2/kappa_s > N/2 or abs(alpha_in1_s)**2/kappa_s*G_0*abs(alpha_p)/gamma_m > N_m/2:
    #    raise ValueError("Warning: The input field is too large for the given Hilbert space dimension.")

    # sytem of equations
    # delta_s, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, delta_m, gamma_m
    delta_p = 0
    kappa_p = kappa_s
    kappa_ext1_p = kappa_ext1_s
    alpha_in1_p = alpha_p * np.sqrt((kappa_p**2/4+delta_p**2)/kappa_ext1_p)
    print(alpha_in1_p, alpha_in1_s)

    #alpha_in1_s = 1 * np.sqrt(kappa_ext1_s)
    def model(vars: list[np.complex128], t, *args):
        b_m, a_s, a_p = vars
        delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s = args
        # mechanical mode
        Gamma_m = -1j*delta_m + gamma_m/2
        Force_m = 0
        dbm_dt = -Gamma_m*b_m + Force_m - 1j*G_0*a_s.conjugate()*a_p
        # pump field
        Gamma_p = 1j*delta_p + kappa_p/2
        Force_p = np.sqrt(kappa_ext1_p)*alpha_in1_p
        dap_dt = -Gamma_p*a_p + Force_p - 1j*G_0*a_s*b_m
        # Stokes field
        Gamma_s = -1j*(delta_s) + kappa_s/2
        Force_s = np.sqrt(kappa_ext1_s)*alpha_in1_s
        das_dt = -Gamma_s*a_s + Force_s - 1j*G_0*a_p*b_m.conjugate()
        #print(b_m, a_s, a_p, dbm_dt, das_dt, dap_dt)
        return [dbm_dt, das_dt, dap_dt]
    print(delta_s, delta_m)

    if calculate_time_evolution:
        # time evolution, the time array length considers the decay rate of the cavity to know when we reach the staedy state
        t = np.linspace(0, max(15/kappa_s,15/gamma_m), 1200)
        # init fields
        init_fields = np.complex128([0, 0, alpha_p])
        args = (delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s)
        # solve time evolution with langevin equations
        solution, infodict = odeintw(model, init_fields, t, args=args, full_output=True)
        b_m_le, a_s_le, a_p_le = solution[:, 0], solution[:, 1], solution[:, 2]
        num_b_m_le, num_a_s_le, num_a_p_le = np.float64(np.abs(b_m_le)**2), np.float64(np.abs(a_s_le)**2), np.float64(np.abs(a_p_le)**2)
        if False:            
            # You have to check that the time evolution is converging to the steady state
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(1,2,figsize=(10,5), constrained_layout=True)
            ax[0].plot(t, num_b_m_le, label='num_b_m')
            ax[0].plot(t, num_a_s_le, label='num_a_s')
            ax[0].plot(0, 0, label='num_a_p')
            ax[0].grid()
            ax[0].set_xlabel('Time [s]')
            ax[0].set_ylabel('b_m a_s Number')
            #ax[0].legend()
            ax0twin = ax[0].twinx()       
            ax0twin.plot(0,0, label='num_b_m')
            ax0twin.plot(0,0, label='num_a_s')
            ax0twin.plot(t, num_a_p_le, label='num_a_p')
            ax0twin.set_ylabel('a_p Number')
            #plt.plot(t, [num_a_ss]*len(t))
            #plt.plot(t, [num_b_ss]*len(t))
            ax[1].plot(b_m_le.real, b_m_le.imag, label='b_m')
            ax[1].plot(a_s_le.real, a_s_le.imag, label='a_s')
            ax[1].plot(0,0, label='a_p')
            # simmetric plot axis range with 0 0 in the center
            max_x, max_y = 1.1*max(max(abs(b_m_le.real)),max(abs(a_s_le.real))), 1.1*max(max(abs(b_m_le.imag)),max(abs(a_s_le.imag)))
            ax[1].set_xlim(-max_x, max_x)
            ax[1].set_ylim(-max_y, max_y)
            ax[1].set_xlabel('b_m a_s x')
            ax[1].set_ylabel('b_m a_s p')
            ax[1].grid()
            #ax[1].legend()
            #twin axis
            ax2=fig.add_subplot(122, frameon=False)
            ax2.plot(0,0, label="b_m")
            ax2.plot(0,0, label="a_s")
            ax2.plot(a_p_le.real, a_p_le.imag, label='a_p')
            max_x, max_y = 1.1*max(abs(a_p_le.real)), 1.1*max(abs(a_p_le.imag))
            ax2.set_xlim(-max_x, max_x)
            ax2.set_ylim(-max_y, max_y)
            ax2.xaxis.tick_top()
            ax2.yaxis.tick_right()
            ax2.set_xlabel('a_p x')
            ax2.set_ylabel('a_p p')
            ax2.xaxis.set_label_position('top')
            ax2.yaxis.set_label_position('right')
            plt.show()
            #camera.snap()
        b_m_ss, a_s_ss, a_p_ss = np.mean(b_m_le[-10:]), np.mean(a_s_le[-10:]), np.mean(a_p_le[-10:])
        print(a_p_ss)
        num_b_m_ss, num_a_s_ss, num_a_p_ss = np.mean(num_b_m_le[-10:]), np.mean(num_a_s_le[-10:]), np.mean(num_a_p_le[-10:])
    else:
        # calculate the steady state solution of the Lindbladian problem and the expectations
        init_fields = np.complex128([0, 0, alpha_p])
        args = (delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s)
        # solve time evolution with langevin equations
        solution = fsolvew(model, init_fields, t=0, args=args)        
        b_m_ss, a_s_ss, a_p_ss = solution[0], solution[1], solution[2]
        num_b_m_ss, num_a_s_ss, num_a_p_ss = np.abs(b_m_ss)**2, np.abs(a_s_ss)**2, np.abs(a_p_ss)**2
    print("Steady state cavity field: %.4e pump photons, %.4e photons and %.4e phonons" % (num_a_p_ss, num_a_s_ss, num_b_m_ss))
    return a_s_ss , a_p_ss


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
    omega_s = omega_p + (-1 if is_sideband_stokes else 1) * 12.000e9 #+ np.linspace(-8e6, 8e6, 10).reshape(-1,1)
    omega_in1_s = omega_s + np.linspace(-1e6, 1e6, 101)
    alpha_p = 7e3*(1 if is_sideband_stokes else 3) #* np.linspace(0,1.2,6).reshape(-1,1)
    G_0 = 100
    Omega_m = 12e9
    gamma_m = 1e6
    #from celluloid import Camera
    #fig,ax=plt.subplots(1,2,figsize=(10,5), constrained_layout=True)
    #camera = Camera(fig)
    r=reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, 
                               is_sideband_stokes, N=5, N_m=5)
    t, a_p=transmittivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, 
                                 is_sideband_stokes, N=5, N_m=5)
    print((omega_in1_s.T-omega_p)[np.where(t==np.max(t))])
    #anim = camera.animate(blit=True)
    #anim.save('scatter.gif')
    plt.plot(omega_in1_s.T-omega_p, r.T, "--",label='Reflectivity')
    plt.plot(omega_in1_s.T-omega_p, t.T, label='Transmissivity')
    plt.ylim(-0.1,4.1)
    plt.xticks(*get_axis_values(omega_in1_s.T-omega_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.grid()
    plt.show()
    plt.plot(omega_in1_s.T-omega_p, np.abs(a_p.T)**2-alpha_p**2, label='Pump photons')
    plt.ylabel("Pump photons difference")
    plt.twinx()
    plt.plot(omega_in1_s.T-omega_p, np.angle(a_p.T), "g",label='Pump photons')
    plt.ylabel("Pump photons phase")
    plt.xticks(*get_axis_values(omega_in1_s.T-omega_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.show()
