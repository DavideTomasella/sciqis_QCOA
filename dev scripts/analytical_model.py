"""
Analytical model for Brillouin scattering in an optical cavity (classical approximations)
Author: D. Tomasella

"""
# In[]
import numpy as np


def reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, is_sideband_stokes=True):
    """
        Calculate the reflectivity of the stokes field of the cavity in the rotating wave approximation for the input field alpha_in1_s (omega_in1_s) 
        and for the mechanical field Omega_m (omega_in1_p-omega_in1_s=omega_p-omega_in1_s because we are resonantly pumping omega_p)
        given the paramters of the cavity fields (stokes, pump, and mechanics) and the input field alpha_in1_s.
        The pump field is approximated as a classical field and in the undepleted regime.
        Notes:
        - The rotation frequency of the input stokes field is omega_in1_s-omega_in1_s=0 (probe) (rotating wave approximation)
        - The rotation frequency of the stokes cavity field is omega_s-omega_in1_s
        - The rotation frequency of the pump field is omega_p-omega_in1_s but we don't care (classical field approximation)
        - The rotation frequency of the mechanical field is (omega_p-omega_in1_s)∓Omega_m when we are considering the stokes and antis-stokes sideband.
        
        Given the steady state solution for the stokes cavity field cavity_ss_sideband(), we derive the reflectivity by calculating the output stokes field 
        alpha_out1_s=alpha_in1_s-sqrt(kappa_ext1_s)*cavity_ss_sideband(alpha_in1_s) (this is the total field outside the port 1 of the cavity!).
        The reflectivity is |alpha_out1_s/alpha_in1_s|^2.
        ```
                | alpha_in1_s - sqrt(kappa_ext1_s) * <a_s> |^2
            R = | -----------------------------------------|     with <a_s> = expectation of the sideband cavity field
                |                alpha_in1_s               |
        ```

        Parameters
        ---------
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
        is_sideband_stokes: (bool)
            if True, the sideband is the stokes field, otherwise it is the anti-stokes field

        Returns
        --------
        reflectivity: (float or np.ndarray)
            reflectivity of the stokes field of the cavity
    """    
    alpha_in1_s = 1
    alpha_out1_s = alpha_in1_s - np.sqrt(kappa_ext1_s)  * cavity_ss_sideband(omega_s-omega_in1_s, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, 
                                                                             (omega_p-omega_in1_s)+(-1 if is_sideband_stokes else 1)*Omega_m, gamma_m,
                                                                             is_sideband_stokes)
    
    return np.abs(alpha_out1_s/alpha_in1_s) ** 2


def transmissivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, is_sideband_stokes=True):
    """
        Calculate the transmissivity of the stokes field of the cavity in the rotating wave approximation for the input field alpha_in1_s (omega_in1_s)
        and for the mechanical field Omega_m (omega_in1_p-omega_in1_s=omega_p-omega_in1_s because we are resonantly pumping omega_p)
        given the paramters of the cavity fields (stokes, pump, and mechanics) and the input field alpha_in1_s.
        The pump field is approximated as a classical field and in the undepleted regime.
        Notes:
        - The rotation frequency of the input stokes field is omega_in1_s-omega_in1_s=0 (probe) (rotating wave approximation)
        - The rotation frequency of the stokes cavity field is omega_s-omega_in1_s
        - The rotation frequency of the pump field is omega_p-omega_in1_s but we don't care (classical field approximation)
        - The rotation frequency of the mechanical field is (omega_p-omega_in1_s)∓Omega_m when we are considering the stokes and antis-stokes sideband.

        Given the steady state solution for the stokes cavity field cavity_ss_stokes(), we derive the transmissivity by calculating the output stokes field
        alpha_out2_s=sqrt(kappa_ext2_s)*cavity_ss_stokes(alpha_in1_s) (this is the total field outside the port 2 of the cavity!).
        The transmissivity is |alpha_out2_s/alpha_in1_s|^2.
        ```
                | sqrt(kappa_ext2_s) * <a_s> |^2
            T = | ---------------------------|     with <a_s> = expectation of the sideband cavity field
                |        alpha_in1_s         |
        ```

        Parameters
        ----------
        omega_in1_s: (float or np.ndarray)
            frequency of the stokes input field [Hz]
        kappa_ext1_s: (float)
            external loss rate of the cavity stokes field = coupling of stokes port 1 field inside the cavity [Hz]
        kappa_ext2_s: (float)
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
        is_sideband_stokes: (bool)
            if True, the sideband is the stokes field, otherwise it is the anti-stokes field
        
        Returns
        -----
        transmissivity: (float or np.ndarray)
            transmissivity of the stokes field of the cavity
    """

    alpha_in1_s = 1
    alpha_out2_s = np.sqrt(kappa_ext2_s) * cavity_ss_sideband(omega_s-omega_in1_s, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, 
                                                              (omega_p-omega_in1_s)+(-1 if is_sideband_stokes else 1)*Omega_m, gamma_m,
                                                              is_sideband_stokes)
    
    return np.abs(alpha_out2_s/alpha_in1_s) ** 2


def cavity_ss_sideband(delta_s, kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0, delta_m, gamma_m, is_sideband_stokes=True):
    """
        Calculate the steady state solution for the stokes field of the cavity. 
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

        Parameters
        -----------
        delta_s: (float or np.ndarray)
            detuning of test frequency compared to the stokes field of the cavity [Hz]
        kappa_ext1_s: (float)
            external loss rate of the cavity stokes field = coupling of stokes port 1 field inside the cavity [Hz]
        kappa_s: (float)
            total loss rate of the cavity stokes field [Hz]
        alpha_in1_s: (complex)
            complex amplitude of the input stokes field
        alpha_p: (complex)
            complex amplitude of the pump cavity field
        G_0: (float)
            single-photon optomechanical coupling strength [Hz]
        delta_m: (float or np.ndarray)
            detuning of the excited mechanical field (i.e, difference between pump and stokes probe) compared to the mechanical frequency [Hz]
        gamma_m: (float)
            total loss rate of the mechanical cavity field [Hz]
        is_sideband_stokes: (bool)
            if True, the sideband is the stokes field, otherwise it is the anti-stokes field

        Returns
        -----------
        alpha_s: (complex or np.ndarray)
            complex amplitude of the stokes field of the cavity
    """

    P = np.sqrt(kappa_ext1_s) / (\
            1j * delta_s + kappa_s / 2 +\
            G_0 ** 2 * np.abs(alpha_p) ** 2 / (1j * delta_m + (-1 if is_sideband_stokes else 1) *gamma_m / 2)\
        ) * alpha_in1_s
    
    return P


if __name__ == "__main__":
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
    omega_in1_s = omega_s + np.linspace(-1e7, 1e7, 1000)
    alpha_p = 7e3*(1 if is_sideband_stokes else 3) * np.linspace(0,1.2,6).reshape(-1,1)
    G_0 = 100
    Omega_m = 12e9
    gamma_m = 1e6

    r=reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, is_sideband_stokes)
    t=transmissivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, is_sideband_stokes)  

    plt.plot(omega_in1_s.T-omega_p, r.T, "--",label='Reflectivity')
    plt.plot(omega_in1_s.T-omega_p, t.T, label='Transmissivity')
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.ylim(-0.1,2.1)
    plt.xticks(*get_axis_values(omega_in1_s.T-omega_p))
    plt.grid()
    plt.show()
    
    omega_s = omega_p + (-1 if is_sideband_stokes else 1) * 12.0008e9 + np.linspace(-6e6, 6e6, 10).reshape(-1,1)
    alpha_p = 7e3*(1 if is_sideband_stokes else 3) #* np.linspace(0,1.2,6).reshape(-1,1)

    r=reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, is_sideband_stokes)
    t=transmissivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s, omega_p, alpha_p, G_0, Omega_m, gamma_m, is_sideband_stokes)  

    plt.plot(omega_in1_s.T-omega_p, r.T, "--",label='Reflectivity')
    plt.plot(omega_in1_s.T-omega_p, t.T, label='Transmissivity')
    plt.ylim(-0.1,2.1)
    plt.xticks(*get_axis_values(omega_in1_s.T-omega_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.grid()
    plt.show()