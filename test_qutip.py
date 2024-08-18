"""
Testing some functions of qutip to understand how to implement an optical cavity.
Developed from
- https://github.com/qutip/qutip-notebooks/blob/master/examples/optomechanical-steadystate.ipynb
- https://github.com/qutip/qutip-notebooks/blob/master/examples/piqs_steadystate_superradiance.ipynb
- https://nbviewer.org/github/qutip/qutip-notebooks/blob/master/examples/piqs-overview.ipynb
Author: D. Tomasella

"""
# In[]
import numpy as np
from qutip import *

# still using:
# resonant pump
# rotating wave approximation (only for the sideband)
# classical approximation for the pump
# undepleted pump approximation


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time

    # simple cavity system: a_in and a
    N_in = 4 # number of modes for the input field
    N = 24 # number of modes for the cavity field
    a_in = tensor(destroy(N_in), qeye(N))
    a = tensor(qeye(N_in), destroy(N))
    num_a_in = a_in.dag()*a_in
    num_a = a.dag()*a

    # Parameters definition
    lambda_to_omega = lambda l: 2 * np.pi * 3e8 / l
    omega = lambda_to_omega(1550e-9)
    omega_in = omega + 1e1
    kappa_ext1_s = 1e6
    kappa_ext2_s = 1e6
    kappa_s = kappa_ext1_s + kappa_ext2_s + 1e6
    kappa_minus_in = kappa_s-kappa_ext1_s
    alpha = 0.03

    # Hamiltonian in rotating wave approximation
    #------------
    Hamiltonian = (omega-omega_in)*num_a+kappa_ext1_s*(a.dag()*a_in+a*a_in.dag())+1j*alpha*kappa_ext2_s*(a.dag()-a)
    # Collapse operators
    #-------------------
    c_a = np.sqrt(kappa_minus_in)*a
    n_th = 1 # a way to simulate a continuous flux of photons in the input field
    coupling = 0.4e7
    cm_ain = np.sqrt(coupling*(1.0 + n_th))*a_in
    cp_ain = np.sqrt(coupling*n_th)*a_in.dag()
    collapse_operators = [c_a,cm_ain]#,cp_ain,cm_ain]

    if True:
        start = time()
        rho_ss = steadystate(Hamiltonian, collapse_operators, method="direct", return_info=True)
        n_a_in_ss = expect(num_a_in, rho_ss)
        n_a_ss = expect(num_a, rho_ss)
        print("Time elapsed SS: ",time()-start)
    print("Steady state: ",n_a_in_ss,n_a_ss)

    rho_0 = tensor(basis(N_in,0),basis(N,0))
    start = time()
    result_me = mesolve(Hamiltonian, rho_0, np.linspace(0, 1e-5, 100), c_ops=collapse_operators, e_ops=[a_in,a],options={"store_states":True})
    a_in_me, a_me = result_me.expect
    print("Time elapsed ME: ",time()-start)

    ntraj = 30
    start = time()
    result_mc = mcsolve(Hamiltonian, rho_0, np.linspace(0, 1e-5, 300), c_ops=collapse_operators, e_ops=[a_in,a], ntraj=ntraj,options={"store_states":True})
    a_in_mc, a_mc = result_mc.expect
    print("Time elapsed MC: ",time()-start)
    plt.plot(result_me.times, np.ones_like(result_me.times)*abs(alpha)**2, label='input (coupling %.2f)'%(1-kappa_ext2_s/kappa_s))
    plt.plot(result_me.times, np.ones_like(result_me.times)*n_a_in_ss, ":", label='SS n_a out')
    plt.plot(result_me.times, np.ones_like(result_me.times)*n_a_ss, ":", label='SS n_a')
    plt.plot(result_me.times, np.abs(a_in_me)**2, label='ME n_a out')
    plt.plot(result_me.times, np.abs(a_me)**2, label='ME n_a')
    plt.plot(result_mc.times, np.abs(a_in_mc)**2, "--", label='MC n_a out')
    plt.plot(result_mc.times, np.abs(a_mc)**2, "--", label='MC n_a')
    plt.legend(loc="center right")

# In[]
    def get_axis_values(values, n=7):
        return np.linspace(min(values), max(values), n), ["%.2f"%(i/1e6) for i in np.linspace(min(values), max(values), n)]
    
    omega_in = omega + np.linspace(-4e6,4e6,31)
    n_a_spec = np.zeros_like(omega_in)
    for i, o_in in enumerate(omega_in):
        Hamiltonian = (omega-o_in)*num_a+kappa_ext1_s*(a.dag()*a_in+a*a_in.dag())+1j*alpha*kappa_ext2_s*(a.dag()-a)
        result_me = mesolve(Hamiltonian, rho_0, np.linspace(0, 1e-5, 100), c_ops=collapse_operators, e_ops=[num_a],options={"store_states":True})
        n_a_spec[i] = result_me.expect[0][-1]
        print("Cavity photons with delta=%.0fHz: %f"%(o_in-omega,n_a_spec[i]))
    plt.plot(omega_in-omega,n_a_spec,label='Cavity photons')
    plt.xticks(*get_axis_values(omega_in-omega))
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("Cavity photons")