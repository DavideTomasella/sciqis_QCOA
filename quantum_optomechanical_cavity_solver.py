"""
Analytical model for cavity solvers
"""
from cavity_solver import BaseCavitySolver
import numpy as np
from typing import Union
from qutip import *

class QuantumOptomechanicalCavitySolver(BaseCavitySolver):

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
        self._is_sideband_stokes = kwargs.get("is_sideband_stokes", False)
        # optomechanical parameters
        self._G0 = kwargs.get("G0", 100)
        self._omega_s = self._omega_p + (-1 if self._is_sideband_stokes else 1)*kwargs.get("FSR_s", 12e9)
        self._Omega_m = kwargs.get("Omega_m", 12e9)
        self._gamma_m = kwargs.get("gamma_m", 1e6)
        if kwargs.get("use_bath", False):
            self._n_bath_m = 1 / (6.626e-34 * self._omega_p / 1.38e-23 / kwargs.get("bath_T",0))
        else:
            self._n_bath_m = 0
        # scan the pump power
        # photons inside a cavity
        #              k_ext1                           k_ext1         P_in1
        #    N  = --------------- |alpha_in1|^2 = ---------------*---------------
        #          k^2/4 + Delta                   k^2/4 + Delta   h_bar * omega
        power_to_alpha = lambda x: np.sqrt(x * 2*np.pi*self._kappa_ext1_s / ((2*np.pi*self._kappa_s)**2 / 4) / (6.626e-34 * self._omega_p))
        self._alpha_p = power_to_alpha(kwargs.get("power_p", 1e-3))
        # define the scan range for the plot.
        range_plot = max(3*self._kappa_s, 2*self._gamma_m)
        self._omega_in1_s = np.linspace(self._omega_s - range_plot, self._omega_s + range_plot, 11)
        # improve visualization with small mechanical linewidth
        if self._gamma_m < 0.2*self._kappa_s:
            self._omega_in1_s = np.unique(np.sort(np.concatenate((self._omega_in1_s, 
                                            np.linspace(self._omega_s - 1.5*self._kappa_s, self._omega_s + 1.5*self._kappa_s, 21),
                                                                  self._omega_p + (-1 if self._is_sideband_stokes else 1) *\
                                  np.linspace(self._Omega_m - 3*self._gamma_m, self._Omega_m + 3*self._gamma_m, 21)))))
        
        self._alpha_in1_s = np.sqrt(2*np.pi*self._kappa_ext1_s) # for numerial stability
        self._delta_s = self._omega_s - self._omega_in1_s
        self._delta_m = (self._omega_p - self._omega_in1_s) + (-1 if self._is_sideband_stokes else 1)*self._Omega_m
        #speed up the calculation far from the optical and mechanical resonance
        close_m = np.abs(self._delta_m) < min(1.2*self._gamma_m , 0.2*self._kappa_s)
        close_s = np.abs(self._delta_s) < max(self._kappa_s, self._gamma_m)
        self._N = np.int32(4 + 2*close_s + 2*close_m)
        self._N_m = np.int32(np.clip(4 + 2*close_s + 4*close_m + 6*close_m*self._is_sideband_stokes, a_min=np.round(0.5+2*self._n_bath_m), a_max=None))
        self._max_t_evolution = max(15/self._kappa_s, 15/self._gamma_m)
        self._configured = True

    def _calculate_cavity_field(self) -> Union[float, np.ndarray]:
        """ 
        Get the steady-state solution for cavity standing wave field from the cavity parameters and the model.
        I don't care about the rotating wave approximation used because I have delta_s and delta_m that represent the difference compare to the cavity modes, 
        i.e., I'm deriving the spectrum of the envelope of the cavity field.
        The analytical model is based on Kharel et al.'s  paper doi:10.1126/sciadv.aav0582.
        ```
                        sqrt(kappa_in1_s) * alpha_in1_s  
                -------------------------------------------------
        <a_s> =                             G0^2 * |alpha_p|^2     with ∓ depending if it is stokes or anti-stokes sideband
                 i*delta_s + kappa_s/2 + -----------------------
                                          i*delta_m ∓ gamma_m/2
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

        a_ss = np.zeros_like(self._omega_in1_s, np.complex128)
        for i in range(len(self._omega_in1_s)):
            # init fields
            a = tensor(destroy(self._N[i]), qeye(self._N_m[i]))
            b = tensor(qeye(self._N[i]), destroy(self._N_m[i]))
            n_a = a.dag()*a
            n_b = b.dag()*b

            # Rotating wave approximation with the input field that correspond to the frequency we are probing
            # and for the mechanical mode, to the difference between sideband and mechanical frequency
            # delta_s = omega-omega_in1
            # delta_m = omega_m - (1 if is_sideband_stokes else -1)*(omega_p-omega_in1_s)
            free_evolution = 2*np.pi*self._delta_s[i]*n_a + 2*np.pi*self._delta_m[i]*n_b
            incoupling_fields = 1j*np.sqrt(2*np.pi*self._kappa_ext1_s)*self._alpha_in1_s*(a.dag()-a)
            #interaction = GO*abs(alpha_p)*(a.dag()+a)*(b.dag()+b)
            interaction = -2*np.pi*self._G0*abs(self._alpha_p)*(a.dag()*b.dag() + a*b if self._is_sideband_stokes else a.dag()*b + a*b.dag())
            decay_channel_a = np.sqrt(2*np.pi*self._kappa_s)*a
            decay_channel_b = np.sqrt(2*np.pi*self._gamma_m*(self._n_bath_m+1))*b
            decay_channel_b_dag = np.sqrt(2*np.pi*self._gamma_m*self._n_bath_m)*b.dag()

            Hamiltonian = free_evolution + incoupling_fields + interaction
            collapse_operators = [decay_channel_a,decay_channel_b,decay_channel_b_dag]

            t = np.linspace(0, self._max_t_evolution, 1200)
            # init vacuum state
            rho_0 = tensor(coherent(self._N[i],0),coherent(self._N_m[i],self._n_bath_m))
            # solve time evolution with master equation
            result = mesolve(Hamiltonian, rho_0, t, collapse_operators, [a,n_a,n_b])
            a_me,n_a_me,n_b_me = result.expect
            a_ss[i] = np.mean(a_me[-10:])
            n_a_ss = np.mean(n_a_me[-10:])
            n_b_ss = np.mean(n_b_me[-10:])
            print(np.max(n_a_me),np.max(n_b_me))
            print(f"MESolver({self._N[i]},{self._N_m[i]}) {i+1}/{len(self._omega_in1_s)}: n_a={n_a_ss}, n_b={n_b_ss}")

        return a_ss
    
    def _calculate_time_evolution(self) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """ Get the cavity field time evolution from the cavity paramters and the model"""
        raise NotImplementedError

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
            "is_sideband_stokes": self._is_sideband_stokes,
            "omega_p": self._omega_p,
            "omega_s": self._omega_s,
            "omega_in1_s": self._omega_in1_s,
            "kappa_ext1_s": self._kappa_ext1_s,
            "kappa_ext2_s": self._kappa_ext2_s,
            "kappa_s": self._kappa_s,
            "delta_s": self._delta_s,
            "alpha_p": self._alpha_p,
            "Omega_m": self._Omega_m,
            "gamma_m": self._gamma_m,
            "delta_m": self._delta_m,
            "G0": self._G0,
            "n_bath_m": self._n_bath_m,
        }
        return config