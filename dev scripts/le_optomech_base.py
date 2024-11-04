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
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time
import sys
#sys.stdout = open('output.txt','wt')

counter=0
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

def reflectivity_ss_sideband(omega_in1_s, omega_s, kappa_ext1_s, kappa_s, alpha_in1_s,
                             omega_in1_p, omega_p, kappa_ext1_p, kappa_p, alpha_in1_p,
                             G_0, Omega_m, gamma_m, is_sideband_stokes=True):
    """
    Calculate the reflectivity spectrum of an optomechanical cavity with the Master Equation Solver.
    ```
            | alpha_in1_s - sqrt(kappa_ext1_s) * <a>  |^2
        R = | ----------------------------------------|     with <a> = expectation of the cavity field
            |               alpha_in1_s               |
    ```

    Parameters
    ----------
    omega_in1_s: (float or np.ndarray)
        frequency of the stokes input field [Hz]
    omega_s: (float)
        frequency of the stokes cavity field [Hz]
    kappa_ext1_s:
        external loss rate of the cavity stokes field = coupling of stokes port 1 field inside the cavity [Hz]
    kappa_s: (float)
        total loss rate of the cavity stokes field [Hz]
    alpha_in1_s: (complex)
        complex amplitude of the stokes input field [sqrt(Hz)]
        it is the "amplitude" of the travelling field derived from the input power in photon/s
    omega_in1_p: (float or np.ndarray)
        frequency of the pump input field [Hz]
    omega_p: (float)
        frequency of the pump cavity field [Hz]
    kappa_ext1_p: (float)
        external loss rate of the cavity pump field = coupling of pump port 1 field inside the cavity [Hz]
    kappa_p: (float)
        total loss rate of the cavity pump field [Hz]
    alpha_in1_p: (complex)
        complex amplitude of the pump input field [sqrt(Hz)]
        it is the "amplitude" of the travelling field derived from the input power in photon/s
    G_0: (float)
        single-photon optomechanical coupling strength [Hz]
    Omega_m: (float)
        frequency of the mechanical cavity field [Hz]
    gamma_m: (float)
        total loss rate of the mechanical cavity field [Hz]
    is_sideband_stokes : bool, optional
        If True, the sideband is the stokes field, otherwise it is the anti-stokes field

    Returns
    -------
    reflectivity: (float or np.ndarray)
        reflectivity of the stokes field of the cavity
    """
    alpha_in1_s = np.sqrt(kappa_ext1_s)
    alpha_out1_s = np.zeros_like(omega_in1_s, np.complex128)
    a_p = np.zeros_like(omega_in1_s, np.complex128)
    for i,o_in in enumerate(omega_in1_s):
        a_s, a_p[i] = get_steady_state_field_optomechanical_cavity(omega_s-o_in, kappa_ext1_s, kappa_s, alpha_in1_s,
                                                              omega_p-omega_in1_p, kappa_ext1_p, kappa_p, alpha_in1_p, 
                                                              G_0, (omega_in1_p-o_in)+(-1 if is_sideband_stokes else 1)*Omega_m, gamma_m,
                                                              is_sideband_stokes=is_sideband_stokes, calculate_time_evolution=False)
        alpha_out1_s[i] = alpha_in1_s - np.sqrt(kappa_ext1_s) * a_s
    return np.abs(alpha_out1_s/alpha_in1_s) ** 2, a_p


def transmittivity_ss_sideband(omega_in1_s, omega_s, kappa_ext1_s, kappa_ext2_s, kappa_s, alpha_in1_s,
                               omega_in1_p, omega_p, kappa_ext1_p, kappa_p, alpha_in1_p,
                               G_0, Omega_m, gamma_m, is_sideband_stokes=True):
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
    kappa_s: (float)
        total loss rate of the cavity stokes field [Hz]
    alpha_in1_s: (complex)
        complex amplitude of the stokes input field [sqrt(Hz)]
        it is the "amplitude" of the travelling field derived from the input power in photon/s
    omega_in1_p: (float or np.ndarray)
        frequency of the pump input field [Hz]
    omega_p: (float)
        frequency of the pump cavity field [Hz]
    kappa_ext1_p: (float)
        external loss rate of the cavity pump field = coupling of pump port 1 field inside the cavity [Hz]
    kappa_p: (float)
        total loss rate of the cavity pump field [Hz]
    alpha_in1_p: (complex)
        complex amplitude of the pump input field [sqrt(Hz)]
        it is the "amplitude" of the travelling field derived from the input power in photon/s
    G_0: (float)
        single-photon optomechanical coupling strength [Hz]
    Omega_m: (float)
        frequency of the mechanical cavity field [Hz]
    gamma_m: (float)
        total loss rate of the mechanical cavity field [Hz]
    is_sideband_stokes : bool, optional
        If True, the sideband is the stokes field, otherwise it is the anti-stokes field
    

    Returns
    -------
    transmissivity: (float or np.ndarray)
        transmissivity of the stokes field of the cavity
    """
    alpha_in1_s = np.sqrt(kappa_ext1_s)
    alpha_out1_s = np.zeros_like(omega_in1_s, np.complex128)
    a_p = np.zeros_like(omega_in1_s, np.complex128)
    for i,o_in in enumerate(omega_in1_s):
        a_s, a_p[i] = get_steady_state_field_optomechanical_cavity(omega_s-o_in, kappa_ext1_s, kappa_s, alpha_in1_s,
                                                                   omega_p-omega_in1_p, kappa_ext1_p, kappa_p, alpha_in1_p,
                                                                   G_0, (omega_in1_p-o_in)+(-1 if is_sideband_stokes else 1)*Omega_m, gamma_m,
                                                                   is_sideband_stokes=is_sideband_stokes, calculate_time_evolution=True)
        alpha_out1_s[i] = np.sqrt(kappa_ext2_s) * a_s
    return np.abs(alpha_out1_s/alpha_in1_s) ** 2, a_p


def get_steady_state_field_optomechanical_cavity(delta_s, kappa_ext1_s, kappa_s, alpha_in1_s,
                                                 delta_p, kappa_ext1_p, kappa_p, alpha_in1_p,
                                                 G_0, delta_m, gamma_m,
                                                 is_sideband_stokes=True, calculate_time_evolution=True):
    """
    Calculate the steady state field of an optomechanical cavity with the Langevin equations.
    ```
        TODO: add equations
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
    delta_p: float
        detuning of the cavity pump field [Hz]
    kappa_ext1_p: float
        external loss rate of the cavity pump field = coupling of pump port 1 field inside the cavity [Hz]
    kappa_p: float
        total loss rate of the cavity pump field [Hz]
    alpha_in1_p: complex
        complex amplitude of the pump input field [sqrt(Hz)]
        it is the "amplitude" of the travelling field derived from the input power in photon/s
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

    def model_stokes(vars: list[np.complex128], t, *args):
        b_m, a_s, a_p = vars
        delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s, is_sideband_stokes = args
        # mechanical mode
        Gamma_m = -1j*delta_m + gamma_m/2
        Force_m = 0
        dbm_dt = -Gamma_m*b_m + Force_m - 1j*G_0*(a_s.conjugate()*a_p if is_sideband_stokes else a_s*a_p.conjugate())
        # Stokes field
        Gamma_s = -1j*delta_s + kappa_s/2
        Force_s = np.sqrt(kappa_ext1_s)*alpha_in1_s
        das_dt = -Gamma_s*a_s + Force_s - 1j*G_0*(a_p*b_m.conjugate() if is_sideband_stokes else a_p.conjugate()*b_m)
        # pump field
        Gamma_p = 1j*delta_p + kappa_p/2
        Force_p = np.sqrt(kappa_ext1_p)*alpha_in1_p
        dap_dt = -Gamma_p*a_p + Force_p - 1j*G_0*(a_s*b_m if is_sideband_stokes else a_s.conjugate()*b_m.conjugate())
        #print(b_m, a_s, a_p, dbm_dt, das_dt, dap_dt)
        #print("%.2e %.2e %.2e %.2e %.2e %.2e"%(dbm_dt.real, dbm_dt.imag, das_dt.real, das_dt.imag, dap_dt.real, dap_dt.imag))
        return [dbm_dt, das_dt, dap_dt]
    
    def model_anti_stokes(vars: list[np.complex128], t, *args):
        b_m, a_s, a_p = vars
        delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s, is_sideband_stokes = args
        # mechanical mode
        Gamma_m = -1j*delta_m + gamma_m/2
        Force_m = 0
        dbm_dt = -Gamma_m*b_m + Force_m - 1j*G_0*a_s*a_p.conjugate()
    
    def jacobian_n_constant(vars: list[np.complex128], t, *args):
        b_m, a_s, a_p = vars
        delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s, is_sideband_stokes = args
        # mechanical mode
        Gamma_m = -1j*delta_m + gamma_m/2
        Force_m = 0
        dbm = [-Gamma_m,
               0,
               -0.5*1j*G_0*a_s.conjugate(),
               0,
               -0.5*1j*G_0*a_p,
               0]
        dbm_c = [0,
                 0.5*1j*G_0*a_p.conjugate(),
                 0,
                 -Gamma_m.conjugate(),
                 0,
                 0.5*1j*G_0*a_s]
        # Stokes field
        Gamma_s = -1j*(delta_s) + kappa_s/2
        Force_s = np.sqrt(kappa_ext1_s)*alpha_in1_s
        das = [0,
                -Gamma_s,
                -0.5*1j*G_0*b_m.conjugate(),
                -0.5*1j*G_0*a_p,
                0,
                0]
        das_c = [0.5*1j*G_0*a_p.conjugate(),
                 0,
                 0,
                 0,
                 -Gamma_s.conjugate(),
                 0.5*1j*G_0*b_m]
        # pump field
        Gamma_p = 1j*delta_p + kappa_p/2
        Force_p = np.sqrt(kappa_ext1_p)*alpha_in1_p
        dap = [-0.5*1j*G_0*a_s,
               -0.5*1j*G_0*b_m,
                -Gamma_p,
                0,
                0,
                0]
        dap_c = [0,
                 0,
                 0,
                 0.5*1j*G_0*a_s.conjugate(),
                 0.5*1j*G_0*b_m.conjugate(),
                -Gamma_p.conjugate()]
        jacobian = np.array([dbm, das, dap, dbm_c, das_c, dap_c])
        constant = np.array([Force_m, Force_s, Force_p, Force_m.conjugate(), Force_s.conjugate(), Force_p.conjugate()])
        return jacobian, constant
    
    def model_ivp(t, vars, *args):
        if False:
            jacobian, constant = jacobian_n_constant(vars, t, *args)
            dvars_dt_c = np.dot(jacobian, np.append(vars, np.conj(vars))) + constant
            dvars_dt = dvars_dt_c[:3]
            #print(dvars_dt-dvars_dt_c[3:].conjugate())
        elif False:
            b_m, a_s, a_p = vars
            nvars = [b_m, a_s, a_p]
            dnvars_dt = model_stokes(nvars, t, *args)
            d_bm, d_as, d_ap = dnvars_dt
            dn_as = d_as.conjugate()*a_s+a_s.conjugate()*d_as
            print(t, (d_as.conjugate()*a_s).real, (d_as.conjugate()*a_s).imag)
            print(t, (d_ap.conjugate()*a_p).real**2+(d_bm.conjugate()*b_m).real**2-(d_as.conjugate()*a_s).real**2)
            dvars_dt = [d_bm, d_as, d_ap]
        else:
            dvars_dt = model_stokes(vars, t, *args)
        return dvars_dt
    
    def detect_steady_state(t, vars, *args):
        delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s, is_sideband_stokes = args
        threshold = 0.3*G_0
        eps = 3e-2
        #b_m, a_s, a_p = np.array([vars[0].real, vars[0].imag]), np.array([vars[1].real, vars[1].imag]), np.array([vars[2].real, vars[2].imag])
        #b_m, a_s, a_p = b_m/np.linalg.norm(b_m), a_s/np.linalg.norm(a_s), a_p/np.linalg.norm(a_p)
        deriv = model_ivp(t, vars, *args)
        #curl_bm,curl_as,curl_ap=np.array([deriv[0].real, deriv[0].imag])@b_m, np.array([deriv[1].real, deriv[1].imag])@a_s, np.array([deriv[2].real, deriv[2].imag])@a_p
        curl_bm = (vars[0].real*deriv[0].real+vars[0].imag*deriv[0].imag)/max(eps,np.abs(vars[0])**2)
        curl_as = (vars[1].real*deriv[1].real+vars[1].imag*deriv[1].imag)/max(eps,np.abs(vars[1])**2)
        curl_ap = (vars[2].real*deriv[2].real+vars[2].imag*deriv[2].imag)/max(eps,np.abs(vars[2])**2)#pump
        #print("%.2e %.2e %.2e"%(curl_bm, curl_as, curl_ap))
        return (np.abs(curl_bm) + np.abs(curl_as) + np.abs(curl_ap)) - threshold

    print(delta_p, delta_s, delta_m)
    # approximate value for the pump field, we use it for initialization of the solver
    alpha_p = alpha_in1_p / (kappa_p/2+1j*delta_p)*np.sqrt(kappa_ext1_p)

    if calculate_time_evolution:
        # time evolution, the time array length considers the decay rate of the cavity to know when we reach the staedy state
        t = np.linspace(0, 20*G_0*max(1/kappa_s,1/gamma_m), 20*G_0*10)
        # init fields
        init_fields = np.complex128([0, 0, alpha_p])
        step=0.2*min(1/kappa_s,1/gamma_m)
        args = (delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s, is_sideband_stokes)
        # solve time evolution with langevin equations
        t0=time.time()
        detect_ss = detect_steady_state
        detect_ss.terminal = True
        detect_ss.direction = -1
        sol = solve_ivp(model_ivp, [t[0], t[-1]], init_fields, args=args, events=detect_ss,
                        method='RK45', first_step=step*0.1, rtol=1e-9, atol=1e-9, dense_output=True)
        print("Elapsed time %.2f"%(time.time()-t0))
        #print(sol)
        if len(sol.t_events[0])>0:
            print("End time/max time: %.2e/%.2e"%(sol.t_events[0], t[-1]))
            #print(model_ivp(sol.t_events[0][0], sol.y_events[0][0], *args))
            t = np.arange(0, sol.t_events[0][0], step)
            #print(len(t))
        else:
            print("No steady state found after max time %.2e"%(t[-1]))
        solution = sol.sol(t).T
        #solution, infodict = odeintw(model, init_fields, t, args=args, full_output=True)
        b_m_le, a_s_le, a_p_le = solution[:, 0], solution[:, 1], solution[:, 2]
        #test= solution[:, 3]
        num_b_m_le, num_a_s_le, num_a_p_le = np.float64(np.abs(b_m_le)**2), np.float64(np.abs(a_s_le)**2), np.float64(np.abs(a_p_le)**2)
        #print(num_b_m_le[-1]+ num_a_s_le[-1]+ num_a_p_le[-1])
        if True:
            # You have to check that the time evolution is converging to the steady state
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(1,2,figsize=(10,5), constrained_layout=True)
            ax[0].plot(t, num_b_m_le, label='num_b_m')
            ax[0].plot(t, num_a_s_le, label='num_a_s')
            ax[0].plot(0, 0, label='num_a_p')
            #ax[0].plot(t, test, label='test')
            ax[0].grid()
            #ax[0].set_ylim(min(num_b_m_le[-1000:]), max(num_b_m_le[-1000:]))
            ax[0].set_xlabel('Time [s]')
            ax[0].set_ylabel('b_m a_s Number')
            ax[0].legend()
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
            ax[1].legend()
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
            global counter
            counter+=1
            #plt.savefig("results\le_scan\%03i.png"%counter)
            plt.show()
            time.sleep(0.1)
            #camera.snap()
        b_m_ss, a_s_ss, a_p_ss = np.mean(b_m_le[-10:]), np.mean(a_s_le[-10:]), np.mean(a_p_le[-10:])
        num_b_m_ss, num_a_s_ss, num_a_p_ss = np.mean(num_b_m_le[-10:]), np.mean(num_a_s_le[-10:]), np.mean(num_a_p_le[-10:])
    else:
        def model_n(vars, *args):
            #eps=1e-10
            b_m, a_s, a_p = vars[0]+1j*vars[1], vars[2]+1j*vars[3], vars[4]+1j*vars[5]
            dbm_dt, das_dt, dap_dt = model_stokes([b_m, a_s, a_p],0, *args)
            #dbm_dt, das_dt, dap_dt = dbm_dt/np.abs(b_m)**2, das_dt/np.abs(a_s)**2, dap_dt/np.abs(a_p)**2
            #bm_dot,as_dot,ap_dot = (b_m.real*dbm_dt.real+b_m.imag*dbm_dt.imag)/(eps+np.abs(b_m)**2), (a_s.real*das_dt.real+a_s.imag*das_dt.imag)/(eps+np.abs(a_s)**2), (a_p.real*dap_dt.real+a_p.imag*dap_dt.imag)/(eps+np.abs(a_p)**2)
            #print("%.2e %.2e %.2e" % (bm_dot, as_dot, ap_dot))
            #print("%.2e %.2e %.2e %.2e %.2e %.2e"%(b_m.real, b_m.imag, a_s.real, a_s.imag, a_p.real, a_p.imag))
            return np.abs(dbm_dt)**2 + np.abs(das_dt)**2 + np.abs(dap_dt)**2
            return np.abs(bm_dot)**2 + np.abs(as_dot)**2 + np.abs(ap_dot)**2 +\
                np.abs(dbm_dt)**2 + np.abs(das_dt)**2 + np.abs(dap_dt)**2
        
        # calculate the steady state solution of the Lindbladian problem and the expectations
        time0 = time.time()
        init_fields = np.float64([0,0, 0,0, alpha_p.real,alpha_p.imag])        
        args = (delta_m, gamma_m, G_0, delta_p, kappa_p, kappa_ext1_p, alpha_in1_p, delta_s, kappa_s, kappa_ext1_s, alpha_in1_s, is_sideband_stokes)
        solution = minimize(model_n, init_fields, args=args, method="powell", options={'gtol':1e-10,'maxiter':1e5})
        # solve time evolution with langevin equations
        print("Elapsed time %.2f"%(time.time()-time0))
        print(solution.message)
        print(solution.x)
        b_m_ss, a_s_ss, a_p_ss = solution.x[0]+1j*solution.x[1], solution.x[2]+1j*solution.x[3], solution.x[4]+1j*solution.x[5]
        num_b_m_ss, num_a_s_ss, num_a_p_ss = np.abs(b_m_ss)**2, np.abs(a_s_ss)**2, np.abs(a_p_ss)**2
    print("Steady state cavity field: %.4e pump photons, %.4e photons and %.4e phonons" % (num_a_p_ss, num_a_s_ss, num_b_m_ss))
    return a_s_ss , a_p_ss


if __name__=="__main__":
    import matplotlib.pyplot as plt
    def get_axis_values(values, n=5):
        return np.linspace(min(values), max(values), n), ["%.4f"%(i/1e9) for i in np.linspace(min(values), max(values), n)]
    # Test the analytical model
    is_sideband_stokes = False
    lambda_to_omega = lambda l: 2 * np.pi * 3e8 / l
    kappa_ext1_s = 1e6
    kappa_ext1_p = 2e6
    kappa_ext2_s = 1e6
    kappa_ext2_p = 1e6
    kappa_s = kappa_ext1_s + kappa_ext2_s + 1e6
    kappa_p = kappa_ext1_p + kappa_ext2_p + 1e6
    omega_p = lambda_to_omega(1550e-9)
    omega_in1_p = omega_p + (-1 if is_sideband_stokes else 1) * 3e5
    omega_s = omega_p + (-1 if is_sideband_stokes else 1) * 12.0012e9 #+ np.linspace(-8e6, 8e6, 10).reshape(-1,1)
    omega_in1_s = omega_s + (-1 if is_sideband_stokes else 3) * np.linspace(-3e6, 1e6, 401)
    alpha_p = 7e3*(1 if is_sideband_stokes else 3) #* np.linspace(0,1.2,6).reshape(-1,1)
    alpha_in1_p = alpha_p * (kappa_p/2+1j*(omega_p-omega_in1_p))/np.sqrt(kappa_ext1_p)
    alpha_in1_s = np.sqrt(kappa_ext1_s)
    G_0 = 100
    Omega_m = 12e9
    gamma_m = 1e5
    #from celluloid import Camera
    #fig,ax=plt.subplots(1,2,figsize=(10,5), constrained_layout=True)
    #camera = Camera(fig)
    r, a_p=reflectivity_ss_sideband(omega_in1_s, omega_s, kappa_ext1_s, kappa_s, alpha_in1_s,
                               omega_in1_p, omega_p, kappa_ext1_p, kappa_p, alpha_in1_p,
                               G_0, Omega_m, gamma_m, is_sideband_stokes)
    t, a_p2=transmittivity_ss_sideband(omega_in1_s, omega_s, kappa_ext1_s, kappa_ext2_s, kappa_s, alpha_in1_s,
                                      omega_in1_p, omega_p, kappa_ext1_p, kappa_p, alpha_in1_p,
                                      G_0, Omega_m, gamma_m, is_sideband_stokes)
    print((omega_in1_s.T-omega_in1_p)[np.where(t==np.max(t))])
    #anim = camera.animate(blit=True)
    #anim.save('scatter.gif')
    plt.plot(omega_in1_s.T-omega_in1_p, r.T, "--",label='Reflectivity')
    plt.plot(omega_in1_s.T-omega_in1_p, t.T, label='Transmissivity')
    plt.ylim(-0.1,4.1)
    plt.xticks(*get_axis_values(omega_in1_s.T-omega_in1_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.grid()
    plt.show()
    plt.plot(omega_in1_s.T-omega_in1_p, r.T, "--",label='Reflectivity')
    plt.plot(omega_in1_s.T-omega_in1_p, t.T, label='Transmissivity')
    plt.xticks(*get_axis_values(omega_in1_s.T-omega_in1_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.grid()
    plt.show()
    plt.plot(omega_in1_s.T-omega_in1_p, np.abs(a_p.T)**2-np.abs(alpha_p)**2, "b",label='Pump photons')
    plt.plot(omega_in1_s.T-omega_in1_p, np.abs(a_p2.T)**2-np.abs(alpha_p)**2, "b--")
    plt.ylabel("Pump photons difference")
    plt.twinx()
    plt.plot(omega_in1_s.T-omega_in1_p, np.angle(a_p.T), "g",label='Pump photons')
    plt.plot(omega_in1_s.T-omega_in1_p, np.angle(a_p2.T), "g--")
    plt.ylabel("Pump photons phase")
    plt.xticks(*get_axis_values(omega_in1_s.T-omega_in1_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.show()
