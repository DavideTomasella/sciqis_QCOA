"""
Analytical model for cavity solvers
"""
from cavity_solver import BaseCavitySolver
import numpy as np
from typing import Union

class AnalyticalCavitySolver(BaseCavitySolver):

    def __init__(self):
        self.__configured = False

    def configure(self, **kwargs):
        """ configure all the solver parameters before running the simulation"""
        # TODO decide if we need parameters as "use time evolution"
        self.kappa_ext1_s = kwargs.get("kappa_ext1_s", 1)
        self.kappa_ext2_s = kwargs.get("kappa_ext2_s", 1)
        self.kappa_s = self.kappa_ext1_s + self.kappa_ext2_s + kwargs.get("kappa_0_s", 1)
        self.alpha_in1_s = 1
        self.__configured = True

    def __calculate_cavity_field(self) -> Union[float, np.ndarray]:
        """ Get the cavity standing wave field from the cavity paramters and the model"""
        return 0
    
    def __calculate_time_evolution(self) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """ Get the cavity field time evolution from the cavity paramters and the model"""
        return 0, 0

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
        config = {
            "alpha_in1_s": self.alpha_in1_s,
            "kappa_ext1_s": self.kappa_ext1_s,
            "kappa_ext2_s": self.kappa_ext2_s,
            "kappa_s": self.kappa_s
        }
        return config