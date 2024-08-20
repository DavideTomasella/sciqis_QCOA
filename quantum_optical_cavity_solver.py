"""
Analytical model for cavity solvers
"""
from cavity_solver import BaseCavitySolver
import numpy as np
from typing import Union
from qutip import *

class QuantumOpticalCavitySolver(BaseCavitySolver):

    def __init__(self):
        super().__init__()

    def configure(self, **kwargs):
        """ configure all the solver parameters before running the simulation

        Parameters
        ----------
        is_sideband_stokes: (bool)
            if True, the sideband is the stokes field, otherwise it is the anti-stokes field
        is_optomechanical: (bool)
            if True, the model includes the optomechanical coupling
        scan_FSR_detuning: (bool)
            if True, the model scans the detuning of the cavity FSR or the resonance frequency if only one optical field is considered
        scan_pump_power: (bool)
            if True, the model scans the pump power
        omega_p: (float)
            frequency of the pump cavity field [Hz]
        kappa_ext1_s: (float)
            external loss rate of the cavity optical field = coupling of port 1 field inside the cavity [Hz]
        kappa_ext2_s: (float)
            external loss rate of the cavity optical field = coupling of port 2 field inside the cavity [Hz]
        kappa_0_s: (float)
            intrinsic loss rate of the cavity optical field [Hz]
        Omega_m: (float)
            frequency of the mechanical cavity field [Hz]
        gamma_m: (float)
            total loss rate of the mechanical cavity field [Hz]
        G_0: (float)
            single-photon optomechanical coupling strength [Hz]
        FSR_s: (float)
            frequency difference of the considered cavity resonance compared to the pump field [Hz] (is more optical fields are considered, this is the FSR of the cavity)
        detuning_s_0: (float)
            initial detuning of the cavity resonance frequency compared to omega_p ± FSR_s [Hz] (± depends on stokes or anti-stokes sideband)
        detuning_s_1: (float)
            final detuning of the cavity resonance frequency compared to omega_p ± FSR_s [Hz] (± depends on stokes or anti-stokes sideband)
        alpha_p: (complex)
            complex amplitude of the pump cavity field
        alpha_p_0: (float)
            initial pump complex amplitude
        alpha_p_1: (float)
            final pump complex amplitude
        
        """
        self._solver = self.__class__.__name__
        self._kappa_ext1_s = kwargs.get("kappa_ext1_s", 1e6)
        self._kappa_ext2_s = kwargs.get("kappa_ext2_s", 1e6)
        self._kappa_s = self._kappa_ext1_s + self._kappa_ext2_s + kwargs.get("kappa_0_s", 1e6)
        self._omega_p = kwargs.get("omega_p", 200e12)
        self._omega_s = self._omega_p
        # define the scan range for the plot.
        range_plot = 3*self._kappa_s
        self._omega_in1_s = np.linspace(self._omega_s - range_plot, self._omega_s + range_plot, 21)
        # improve visualization with small mechanical linewidth
        self._omega_in1_s = np.unique(np.sort(np.concatenate((self._omega_in1_s, 
                                        np.linspace(self._omega_s - 0.4*self._kappa_s, self._omega_s + 0.4*self._kappa_s, 11)))))
        self._alpha_in1_s = np.sqrt(2*np.pi*self._kappa_ext1_s) # for numerial stability
        self._delta_s = self._omega_s - self._omega_in1_s
        #speed up the calculation far from the optical and mechanical resonance
        close_s = np.abs(self._delta_s) < (self._kappa_s)
        self._N = np.int32(4 + 2*close_s)
        self._N_m = np.int32(4 + 2*close_s)
        if kwargs.get("use_time_evolution", False):
            self._max_t_evolution = 2/self._kappa_s
        else:
            self._max_t_evolution = 10/self._kappa_s
        self._configured = True

    def _calculate_cavity_field(self) -> Union[float, np.ndarray]:
        """ 
        Calculate the steady state field of an optomechanical cavity with the Master Equation Solver.
        ```
            H = delta_s * a.dag() * a + 1j * sqrt(kappa_ext1_s) * alpha_in1_s * (a.dag() - a)
            L = [sqrt(kappa_s) * a]
        from the previous rotating wave approximation:
            delta_s = omega_s - omega_in1_s
            delta_m = (omega_p - omega_in1_s) ∓ Omega_m depending if it is stokes or anti-stokes sideband
        thus the steady state for the input field with amplitude is simply alpha_in1_s
        ```

        Returns
        -----------
        alpha_s: (complex or np.ndarray)
            complex amplitude of the sideband field of the cavity
        """

        t = np.linspace(0, self._max_t_evolution, 800)
        a_ss = np.zeros_like(self._omega_in1_s, np.complex128)
        for i in range(len(self._omega_in1_s)):
            # init fields
            a = tensor(destroy(self._N[i]))
            n_a = a.dag()*a

            # Rotating wave approximation with the input field that correspond to the frequency we are probing
            # delta_s = omega-omega_in1
            free_evolution = 2*np.pi*self._delta_s[i]*n_a
            incoupling_fields = 1j*np.sqrt(2*np.pi*self._kappa_ext1_s)*self._alpha_in1_s*(a.dag()-a)
            #interaction = GO*abs(alpha_p)*(a.dag()+a)*(b.dag()+b)
            decay_channel_a = np.sqrt(2*np.pi*self._kappa_s)*a
            
            Hamiltonian = free_evolution + incoupling_fields
            collapse_operators = [decay_channel_a]

            # init vacuum state
            rho_0 = tensor(coherent(self._N[i],0))
            # solve time evolution with master equation
            result = mesolve(Hamiltonian, rho_0, t, collapse_operators, [a,n_a])
            a_me,n_a_me = result.expect
            a_ss[i] = np.mean(a_me[-10:])
            n_a_ss = np.mean(n_a_me[-10:])
            print(np.max(n_a_me))
            print(f"MESolver({self._N[i]},{self._N_m[i]}) {i+1}/{len(self._omega_in1_s)}: n_a={n_a_ss}")

        return a_ss
    
    def _calculate_time_evolution(self) -> tuple[Union[float, np.ndarray], tuple[Union[float, np.ndarray]]]:
        """ 
        Get the cavity field time evolution from the cavity paramters and the model
         ```
            H = delta_s * a.dag() * a + 1j * sqrt(kappa_ext1_s) * alpha_in1_s * (a.dag() - a)
            L = [sqrt(kappa_s) * a]
        from the previous rotating wave approximation:
            delta_s = omega_s - omega_in1_s
            delta_m = (omega_p - omega_in1_s) ∓ Omega_m depending if it is stokes or anti-stokes sideband
        thus the steady state for the input field with amplitude is simply alpha_in1_s
        ```
        
        Returns
        --------
        t: (np.ndarray)
            time array for the time evolution of the system
        populations: (tuple(np.ndarray))
            tuple of the different mode expected population over time. Each element is a 2d np.ndarray (frequencies x time)
        """
        
        t = np.linspace(0, self._max_t_evolution, 800)
        list_n_a_me = np.zeros([len(self._omega_in1_s),len(t)], np.float32)
        for i in range(len(self._omega_in1_s)):
            # init fields
            a = tensor(destroy(self._N[i]))
            n_a = a.dag()*a

            # Rotating wave approximation with the input field that correspond to the frequency we are probing
            # delta_s = omega-omega_in1
            free_evolution = 2*np.pi*self._delta_s[i]*n_a
            incoupling_fields = 1j*np.sqrt(2*np.pi*self._kappa_ext1_s)*self._alpha_in1_s*(a.dag()-a)
            #interaction = GO*abs(alpha_p)*(a.dag()+a)*(b.dag()+b)
            decay_channel_a = np.sqrt(2*np.pi*self._kappa_s)*a
            
            Hamiltonian = free_evolution + incoupling_fields
            collapse_operators = [decay_channel_a]

            # init vacuum state
            rho_0 = tensor(coherent(self._N[i],0))
            # solve time evolution with master equation
            result = mesolve(Hamiltonian, rho_0, t, collapse_operators, [a,n_a])
            a_me,n_a_me = result.expect
            list_n_a_me[i,:] = n_a_me
            n_a_ss = np.mean(n_a_me[-10:])
            print(np.max(n_a_me))
            print(f"MESolver({self._N[i]},{self._N_m[i]}) {i+1}/{len(self._omega_in1_s)}: n_a={n_a_ss}")

        return t, list_n_a_me

    def solve_cavity_RT(self):
        """ Solve the cavity steady state solution
        Given the steady state solution for the cavity field amplitude <a_s>=get_cavity_field(), 
        - we derive the reflectivity by calculating the output stokes field alpha_out1_s
          (this is the total field outside the port 1 of the cavity!).
          The reflectivity is |alpha_out1_s/alpha_in1_s|^2.
        ```
                | alpha_in1_s - sqrt(kappa_ext1_s) * <a_s> |^2
            R = | -----------------------------------------|  
                |                alpha_in1_s               |
        ```
        - we derive the transmissivity by calculating the output field alpha_out2_s,
          (this is the total field outside the port 2 of the cavity!).
          The transmissivity is |alpha_out2_s/alpha_in1_s|^2.
        ```
                | sqrt(kappa_ext2_s) * <a_s> |^2
            T = | ---------------------------| 
                |        alpha_in1_s         |
        ```

        Returns
        -------
        R : float or np.ndarray
            The reflectivity of the cavity. 
            In the case of scanning the input field frequency, this is a 1D array.
            If we are scanning some parameter (e.g, the cavity FSR or the pump power), this is a 2D array.
        T : float or np.ndarray
            In the case of scanning the input field frequency, this is a 1D array.
            If we are scanning some parameter (e.g, the cavity FSR or the pump power), this is a 2D array.
        """
        return super().solve_cavity_RT()
    
    def solve_cavity_time_evolution(self) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """ Solve the cavity steady state solution while calculating the time evolution of the cavity field"""
        return super().solve_cavity_time_evolution()
    
    def get_current_configuration(self):
        """ Get the current configuration of the solver"""
        if not self._configured:
            raise Exception("Solver not configured")
        config = {
            "solver": self._solver,
            "omega_s": self._omega_s,
            "omega_p": self._omega_p,
            "omega_in1_s": self._omega_in1_s,
            "kappa_ext1_s": self._kappa_ext1_s,
            "kappa_ext2_s": self._kappa_ext2_s,
            "kappa_s": self._kappa_s,
            "delta_s": self._delta_s,
            "max_t_evolution": self._max_t_evolution,
            "N": self._N,
        }
        return config