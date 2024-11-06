"""
Analytical model for Brillouin scattering in an optical cavity (classical approximations)
Author: D. Tomasella

"""
# In[]
import numpy as np


def reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s,
                             omega_p, alpha_p, 
                             G_0_mm, Omega_m_mm, gamma_m_mm, 
                             is_sideband_stokes=True):
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
    alpha_out1_s = alpha_in1_s - np.sqrt(kappa_ext1_s)  * cavity_ss_sideband((omega_s-omega_in1_s).reshape(np.size(omega_s),-1,1), kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0_mm.reshape(1,1,-1), 
                                                                             (omega_p-omega_in1_s).reshape(1,-1,1)+(-1 if is_sideband_stokes else 1)*Omega_m_mm.reshape(1,1,-1),
                                                                              gamma_m_mm.reshape(1,1,-1), is_sideband_stokes).reshape(np.size(omega_s)*np.size(alpha_p),-1)
    
    return np.abs(alpha_out1_s/alpha_in1_s) ** 2


def transmissivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s,
                                omega_p, alpha_p,
                                  G_0_mm, Omega_m_mm, gamma_m_mm,
                                    is_sideband_stokes=True):
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
    alpha_out2_s = np.sqrt(kappa_ext2_s) * cavity_ss_sideband((omega_s-omega_in1_s).reshape(np.size(omega_s),-1,1), kappa_ext1_s, kappa_s, alpha_in1_s, alpha_p, G_0_mm.reshape(1,1,-1), 
                                                              (omega_p-omega_in1_s).reshape(1,-1,1)+(-1 if is_sideband_stokes else 1)*Omega_m_mm.reshape(1,1,-1),
                                                               gamma_m_mm.reshape(1,1,-1), is_sideband_stokes).reshape(np.size(omega_s)*np.size(alpha_p),-1)
    
    return np.abs(alpha_out2_s/alpha_in1_s) ** 2


def cavity_ss_sideband(delta_s, kappa_ext1_s, kappa_s, alpha_in1_s,
                        alpha_p, G_0_mm, delta_m_mm, gamma_m_mm,
                          is_sideband_stokes=True):
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
            np.abs(alpha_p) ** 2 * np.sum(G_0_mm ** 2 / (1j * delta_m_mm + (-1 if is_sideband_stokes else 1) *gamma_m_mm / 2), axis=2, keepdims=True)\
        ) * alpha_in1_s
    
    return P


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def get_axis_values(values, n=5):
        return np.linspace(min(values), max(values), n), ["%.4f"%(i/1e9) for i in np.linspace(min(values), max(values), n)]
    # Test the analytical model
    is_sideband_stokes = True
    lambda_to_omega = lambda l: 3e8 / l
    kappa_ext1_s = 2.5e6
    kappa_ext1_p = 2e6
    kappa_ext2_s = 5e5
    kappa_ext2_p = 5e5
    kappa_s = kappa_ext1_s + kappa_ext2_s + 3e5
    kappa_p = kappa_ext1_p + kappa_ext2_p + 3e5

    # approximate frequency to calculate the mechanical response
    omega_lowf = lambda_to_omega(1553.3e-9)
    omega_highf = omega_lowf + 12.7e9 #+ np.linspace(-8e6, 8e6, 10).reshape(-1,1)
    omega_s = omega_lowf if is_sideband_stokes else omega_highf
    omega_p = omega_highf if is_sideband_stokes else omega_lowf

    # mechanical parameters
    G_0 = 15
    v_a=6360#m/s
    v_0=3e8/1.55#m/s
    Omega_0 = 2 * omega_p * v_a / v_0
    Omega_m = Omega_0 / (1 + (1 if is_sideband_stokes else -1) * v_a / v_0)
    gamma_m = 800#Hz
    
    # new tuned optical FSR to show some peaks in the cavity response
    omega_highf = omega_lowf + Omega_0 -10e5
    omega_s = omega_lowf if is_sideband_stokes else omega_highf
    omega_p = omega_highf if is_sideband_stokes else omega_lowf

    #multimode: fundamental modes
    N=3
    L=1e-2
    zero_order_fsr = v_a/L #should be 720kHz with pi*v_a/L, but it isn't
    number_of_oscillations_central_mode = (Omega_m // zero_order_fsr)
    Omega_m_zeros = Omega_m + zero_order_fsr * np.arange(-np.floor(N/2),np.ceil(N/2))
    modular_Omega_m_zeros = Omega_m_zeros - number_of_oscillations_central_mode * zero_order_fsr
    G_0_zeros = G_0*np.sinc(modular_Omega_m_zeros / 2 / zero_order_fsr)**2
    gamma_m_zeros = gamma_m*np.ones_like(Omega_m_zeros)

    #multimode: high order modes
    M = 10
    high_order_fsr = 45e3
    # data for dx=100um, dy=100um, angX=1mrad, angY=1mrad
    distribution_highorder_g0 = np.array([55,123,380,485,667,380,342,164,108,32])/4000
    Omega_m_highorder = np.repeat(Omega_m_zeros,M) + high_order_fsr * np.tile(np.arange(0,M),N)
    modular_Omega_m_highorder = Omega_m_highorder - number_of_oscillations_central_mode * zero_order_fsr
    G_0_highorder = G_0 * np.sinc(modular_Omega_m_highorder / 2 / zero_order_fsr)**2\
                         * np.tile(distribution_highorder_g0, N)
    gamma_m_highorder = gamma_m*np.ones_like(Omega_m_highorder)
    
    print(Omega_m_zeros, G_0_zeros)
    print(Omega_m_highorder, G_0_highorder)
    frequency_range_mech_modes = Omega_m * (-1 if is_sideband_stokes else 1) + np.array([-N/2*zero_order_fsr, N/2*zero_order_fsr+M*high_order_fsr])

    # input fields definition
    omega_in1_p = omega_p + (-1 if is_sideband_stokes else 1) * 1e5
    omega_in1_s = omega_s + np.linspace(-4e6, 4e6, 4001)
    power_in1_p = 2.7e-5*(1 if is_sideband_stokes else 3)
    power_in1_p = 400e-6
    alpha_in1_p = np.sqrt(power_in1_p / (6.626e-34 *omega_p))        
    alpha_p = alpha_in1_p * np.sqrt(kappa_ext1_p)/(kappa_p/2+1j*(omega_p-omega_in1_p))
    #alpha_p = 7e3*(1 if is_sideband_stokes else 3) #* np.linspace(0,1.2,6).reshape(-1,1)

    r=reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s,
                                omega_p, alpha_p, G_0_highorder, Omega_m_highorder, gamma_m_highorder,
                                  is_sideband_stokes)
    t=transmissivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s,
                                  omega_p, alpha_p, G_0_highorder, Omega_m_highorder, gamma_m_highorder,
                                    is_sideband_stokes)  

    plt.plot(omega_in1_s.T-omega_p, r.T, "--",label='Reflectivity')
    plt.plot(omega_in1_s.T-omega_p, t.T, label='Transmissivity')
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")

    plt.xlim(frequency_range_mech_modes)
    plt.xticks(*get_axis_values(frequency_range_mech_modes))
    plt.ylim(-0.1,1.1)
    #plt.xticks(*get_axis_values(omega_in1_s.T-omega_p))
    plt.grid()
    plt.show()
    omega_s = omega_s #+ np.linspace(-1e6, 1e6, 3).reshape(-1,1)
    alpha_p = alpha_p * np.linspace(0,1.2,6).reshape(-1,1,1)

    r=reflectivity_ss_sideband(omega_in1_s, kappa_ext1_s, omega_s, kappa_s, 
                               omega_p, alpha_p, G_0_highorder, Omega_m_highorder, gamma_m_highorder,
                                 is_sideband_stokes)
    t=transmissivity_ss_sideband(omega_in1_s, kappa_ext1_s, kappa_ext2_s, omega_s, kappa_s, 
                                 omega_p, alpha_p, G_0_highorder, Omega_m_highorder, gamma_m_highorder,
                                   is_sideband_stokes)  

    plt.plot((omega_in1_s.T-omega_p).T, r.T, "--",label='Reflectivity')
    plt.plot((omega_in1_s.T-omega_p).T, t.T, label='Transmissivity')
    plt.ylim(-0.1,1.1)
    #plt.xlim([-5e6,+5e6])
    #plt.xticks(*get_axis_values(omega_in1_s.T-omega_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.grid()
    plt.show()

    plt.plot((omega_in1_s.T-omega_s).T, r.T, "--",label='Reflectivity')
    plt.plot((omega_in1_s.T-omega_s).T, t.T, label='Transmissivity')
    plt.ylim(-0.1,1.1)
    plt.xlim([-N/2*zero_order_fsr,+N/2*(zero_order_fsr+M*high_order_fsr)])
    #plt.xticks(*get_axis_values(omega_in1_s.T-omega_p))
    plt.xlabel("Sideband relative frequency [GHz]")
    plt.ylabel("Cavity response")
    plt.grid()
    plt.show()
# %%
