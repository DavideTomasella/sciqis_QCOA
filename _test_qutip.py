"""
Numerical solution of the Lindbladian steady state equation for Brillouin scattering in an optical cavity
Developed from https://github.com/qutip/qutip-notebooks/blob/master/examples/optomechanical-steadystate.ipynb
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

    # simple cavity system: a_in and a
    N_in = 10 # number of modes for the input field
    N = 10 # number of modes for the cavity field
    a_in = tensor(destroy(N_in), qeye(N))
    a = tensor(qeye(N_in), destroy(N))
    num_a_in = a_in.dag()*a_in
    num_a = a.dag()*a

    # Parameters definition
    lambda_to_omega = lambda l: 2 * np.pi * 3e8 / l
    omega_in = lambda_to_omega(1550e-9)
    omega = omega_in + 1e1
    kappa_ext1_s = 1e6
    kappa_ext2_s = 1e6
    kappa_s = kappa_ext1_s + kappa_ext2_s + 1e6
    kappa_minus_in = kappa_s-kappa_ext1_s
    alpha = 1

    # Hamiltonian in rotating wave approximation
    #------------
    Hamiltonian = (omega-omega_in)*num_a+kappa_ext1_s*(a.dag()*a_in+a*a_in.dag())+alpha*kappa_ext2_s*(a.dag()+a)
    # Collapse operators
    #-------------------
    c_a = np.sqrt(kappa_minus_in)*a
    n_th = 1 # a way to simulate a continuous flux of photons in the input field
    coupling = 1e7
    cm_ain = np.sqrt(coupling*(1.0 + n_th))*a_in
    cp_ain = np.sqrt(coupling*n_th)*a_in.dag()
    collapse_operators = [c_a,cm_ain]#,cp_ain,cm_ain]

    if True:
        rho_ss = steadystate(Hamiltonian, collapse_operators, return_info=True)
    n_a_in_ss = expect(num_a_in, rho_ss)
    n_a_ss = expect(num_a, rho_ss)
    print("Steady state: ",n_a_in_ss,n_a_ss)

    rho_0 = tensor(basis(N_in,0),basis(N,0))
    result_me = mesolve(Hamiltonian, rho_0, np.linspace(0, 1e-5, 100), c_ops=collapse_operators, e_ops=[num_a_in,num_a],options={"store_states":True})
    n_a_in_me, n_a_me = result_me.expect[0], result_me.expect[1]
    plt.plot(result_me.times, n_a_in_me, label='n_a_in')
    plt.plot(result_me.times, n_a_me, label='n_a')
    plt.legend()