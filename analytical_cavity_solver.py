"""
Analytical model for cavity solvers
"""
from cavity_solver import BaseCavitySolver
import numpy as np
from typing import Union

class AnalyticalCavitySolver(BaseCavitySolver):

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
        self._solver = "AnalyticalCavitySolver"
        self._kappa_ext1_s = kwargs.get("kappa_ext1_s", 1e6)
        self._kappa_ext2_s = kwargs.get("kappa_ext2_s", 1e6)
        self._kappa_s = self._kappa_ext1_s + self._kappa_ext2_s + kwargs.get("kappa_0_s", 1e6)
        self._omega_p = kwargs.get("omega_p", 200e12)
        self._is_sideband_stokes = kwargs.get("is_sideband_stokes", False)
        scan_points = 7
        # optical vs mechanical mode model
        if kwargs.get("is_optomechanical", False):
            self._G0 = kwargs.get("G0", 100)
            self._omega_s = self._omega_p + (-1 if self._is_sideband_stokes else 1)*kwargs.get("FSR_s", 12e9)
            self._Omega_m = kwargs.get("Omega_m", 12e9)
            self._gamma_m = kwargs.get("gamma_m", 1e6)
            # scan the pump power
            # photons inside a cavity
            #              k_ext1                           k_ext1         P_in1
            #    N  = --------------- |alpha_in1|^2 = ---------------*---------------
            #          k^2/4 + Delta                   k^2/4 + Delta   h_bar * omega
            power_to_alpha = lambda x: np.sqrt(x * 2*np.pi*self._kappa_ext1_s / ((2*np.pi*self._kappa_s)**2 / 4) / (6.626e-34 *self._omega_p))
            if kwargs.get("scan_pump_power", False):
                self._alpha_p = np.linspace(power_to_alpha(kwargs.get("power_p_0", 0)), power_to_alpha(kwargs.get("power_p_1", 1e-1)), scan_points).reshape(-1,1)
            else:
                self._alpha_p = power_to_alpha(kwargs.get("power_p", 1e-3))
        else:
            self._G0 = 0
            self._omega_s = self._omega_p
            self._Omega_m = 0
            self._gamma_m = 0
            self._alpha_p = 0
        # define the scan range for the plot. If we are scanning the central frequency of the response, we need to take it into account
        if kwargs.get("scan_FSR_detuning", False):
            range_plot = max(3*self._kappa_s, 2*self._gamma_m)
            det_m = kwargs.get("detuning_s_0", -range_plot/2)
            det_p = max(det_m, kwargs.get("detuning_s_1", range_plot/2))
            self._omega_in1_s = np.linspace(self._omega_s + min(-range_plot,det_m), self._omega_s + max(range_plot,det_p), 301)
            self._omega_s = np.linspace(self._omega_s + det_m, self._omega_s + det_p, scan_points).reshape(-1,1)
        else:
            range_plot = max(3*self._kappa_s, 2*self._gamma_m)
            self._omega_in1_s = np.linspace(self._omega_s - range_plot, self._omega_s + range_plot, 301)
        # improve visualization with small mechanical linewidth
        if self._gamma_m < 0.2*self._kappa_s:
            self._omega_in1_s = np.unique(np.sort(np.concatenate((self._omega_in1_s, 
                                                                  self._omega_p + (-1 if self._is_sideband_stokes else 1) *\
                                  np.linspace(self._Omega_m - self._gamma_m, self._Omega_m + self._gamma_m, 201)))))
        
        self._alpha_in1_s = np.sqrt(2*np.pi*self._kappa_ext1_s) # for numerial stability
        self._delta_s = self._omega_s - self._omega_in1_s
        self._delta_m = (self._omega_p - self._omega_in1_s) + (-1 if self._is_sideband_stokes else 1)*self._Omega_m
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

        P = np.sqrt(2*np.pi*self._kappa_ext1_s) / (\
            1j * 2*np.pi*self._delta_s + 2*np.pi*self._kappa_s / 2 +\
            (2*np.pi*self._G0) ** 2 * np.abs(self._alpha_p) ** 2 / \
                (1j * 2*np.pi*self._delta_m + (-1 if self._is_sideband_stokes else 1) * 2*np.pi*self._gamma_m / 2)\
        ) * self._alpha_in1_s
        return P
    
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
        }
        return config