"""
Abstract class for cavity solvers
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Union

class BaseCavitySolver(ABC):

    def __init__(self):
        self._configured = False

    @abstractmethod
    def configure(self, **kwargs):
        """ configure all the solver parameters before running the simulation"""
        # TODO decide if we need parameters as "use time evolution"
        self._solver = "BaseCavitySolver"
        self._kappa_ext1_s = kwargs.get("kappa_ext1_s", 1)
        self._kappa_ext2_s = kwargs.get("kappa_ext2_s", 1)
        self._kappa_s = self._kappa_ext1_s + self._kappa_ext2_s + kwargs.get("kappa_0_s", 1)
        self._omega_p = kwargs.get("omega_p", 200e12)
        self._omega_s = self._omega_p
        self._omega_in1_s = np.linspace(self._omega_s - 3*self._kappa_s, self._omega_s + 3*self._kappa_s, 301)
        self._alpha_in1_s = np.sqrt(2*np.pi*self._kappa_ext1_s)
        self._delta_s = self._omega_s - self._omega_in1_s
        self._configured = True

    @abstractmethod
    def _calculate_cavity_field(self) -> Union[float, np.ndarray]:
        """ 
        Get the cavity standing wave field from the cavity paramters and the model
        
        Returns
        -------
        a_s : float or np.ndarray
            The steady-state solution for the cavity field amplitude.
        """
        return 0
    
    @abstractmethod
    def _calculate_time_evolution(self) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """ 
        Get the cavity field time evolution from the cavity paramters and the model
        
        Returns
        -------
        na_s : float or np.ndarray
            The expectation of the number of photons in the cavity mode a_s.
        nb_m : float or np.ndarray
            The expectation of the number of phonons in the mechanical mode b_m.
        """
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
        delta_s : float or np.ndarray
            The probing field frequency compared to the central frequency of the optical response.
        R : float or np.ndarray
            The reflectivity of the cavity. 
            In the case of scanning the input field frequency, this is a 1D array.
            If we are scanning some parameter (e.g, the cavity FSR or the pump power), this is a 2D array.
        T : float or np.ndarray
            In the case of scanning the input field frequency, this is a 1D array.
            If we are scanning some parameter (e.g, the cavity FSR or the pump power), this is a 2D array.
        """
        if not self._configured:
            raise Exception("Solver not configured")
        
        a_s = self._calculate_cavity_field()
        alpha_out1_s = self._alpha_in1_s - np.sqrt(2*np.pi*self._kappa_ext1_s) * a_s
        alpha_out2_s = np.sqrt(2*np.pi*self._kappa_ext2_s) * a_s
        R = np.abs(alpha_out1_s/self._alpha_in1_s) ** 2
        T = np.abs(alpha_out2_s/self._alpha_in1_s) ** 2
        return self._omega_in1_s-self._omega_p, R, T
    
    def solve_cavity_time_evolution(self) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """ 
        Solve the cavity steady state solution while calculating the time evolution of the cavity field
        
        Returns
        -------
        delta_s : float or np.ndarray
            The probing field frequency compared to the central frequency of the optical response.
        na_s : float or np.ndarray
            The expectation of the number of photons in the cavity mode a_s.
        nb_m : float or np.ndarray
            The expectation of the number of phonons in the mechanical mode b_m.
        """
        if not self._configured:
            raise Exception("Solver not configured")
        na_s, nb_m = self._calculate_time_evolution()
        # the number of photons and phonons is normalized to 1 photon coupled to the cavity per second
        return self._delta_s, na_s, nb_m
    
    @abstractmethod
    def get_current_configuration(self):
        """ Get the current configuration of the solver"""
        if not self._configured:
            raise Exception("Solver not configured")
        config = {
            "solver": self._solver,
            "omega_s": self._omega_s,
            "omega_p": self._omega_p,
            "omega_in1_s": self._omega_in1_s,
            "alpha_in1_s": self._alpha_in1_s,
            "kappa_ext1_s": self._kappa_ext1_s,
            "kappa_ext2_s": self._kappa_ext2_s,
            "kappa_s": self._kappa_s,
            "delta_s": self._delta_s,
        }
        return config
    