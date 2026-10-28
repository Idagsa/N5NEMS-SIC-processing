#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""---------------------------------------------------------------------------
AMSR Ocean Algorithm; Frank J. Wentz, Thomas Meissner; Remote
Sensing Systems; Version 2; November 2; 2000. Modified to NIMBUS 6 SCAMS geometry.
NIMBUS 6 SCAMS Scanning Microwave Spectrometer 145 km spatial resolution 2400 km swath
Tb=scams(V,W,L,Ta,Ts,theta,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh)
V: columnar water vapor [mm]
W: windspeed over water [m/s]
L: columnar cloud liquid water [mm]
Ts: sea surface temperature [K]
theta: incidence angle [deg]
Ti: ice surface temperature [K]
c_ice: ice concentration [0-1]
e_ice: ice emissivity [0-1]
-------------------------------------------------------------------------------"""

import numpy as np
from numpy import lib
import sys, os, string, math, cmath
import pickle

def NEMSv3(V,W,L,Ta,Ts,c_ice):
    
    # NEMS freqensies
    frequencies = np.array([22.235, 31.400, 53.650, 54.900, 58.800])

    # Model Coefficients for the Atmosphere pr frequencies FOR AMSR!!!
    b0 = [241.69, 239.45, 245.87, 245.87, 245.87]
    b1 = [3.1032, 2.5441, 2.5061, 2.5061, 2.5061]
    b2 = [-0.081429, -0.051284, -0.062789, -0.062789, -0.062789]
    b3 = [0.00099893, 0.00045202, 0.00075962, 0.00075962, 0.00075962]
    b4 = [-4.837e-06, -1.436e-06, -3.606e-06, -3.606e-06, -3.606e-06]
    b5 = [0.20, 0.58, 0.53, 0.53, 0.53]
    b6 = [-0.20, -0.57, -12.52, -12.52, -12.52]
    b7 = [-0.0521, -0.0238, -0.2326, -0.2326, -0.2326]
    
    # RMS Error in Oxygen and Water Vapor Absorption Approximation FOR AMSR!!!
    ao1 = [0.01575, 0.04006, 1.13176, 1.13176, 1.13176]
    ao2 = [-0.000087, -0.000200, -0.000226, -0.000226, -0.000226]
    av1 = [0.00514, 0.00188, 0.00317, 0.00317, 0.00317]
    av2 = [0.19e-5, 0.09e-5, 0.27e-5, 0.27e-5, 0.27e-5]

    # Coefficients for Rayleigh Absorption and Mie Scattering FOR AMSR
    aL1 = [0.0891, 0.2027, 0.4021, 0.4021, 0.4021]
    aL2 = [0.0281, 0.0261, 0.0231, 0.0231, 0.0231]

    # Model Coefficients for Geometric Optics for AMSR
    r0v = [-0.00063, -0.00101, -0.00123, -0.00123, -0.00123]
    r0h = [0.00139, 0.00191, 0.00197, 0.00197, 0.00197]
    r1v = [-0.000070, -0.000105, -0.000113, -0.000113, -0.000113]
    r1h = [0.000085, 0.000112, 0.000119, 0.000119, 0.000119]
    r2v = [-2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05]
    r2h = [-4.195e-05, -5.451e-05, -5.5e-05, -5.5e-05, -5.5e-05]
    r3v = [0.41e-06, 0.45e-06, 0.32e-06, 0.32e-06, 0.32e-06]
    r3h = [-0.20e-06, -0.36e-06, -0.44e-06, -0.44e-06, -0.44e-06]
    
        
    # The coefficients m1 and m2. Units are s/m. 
    m1v = [0.00178, 0.00257, 0.00260, 0.00260, 0.00260]
    m1h = [0.00308, 0.00329, 0.00330, 0.00330, 0.00330]
    m2v = [0.00730, 0.00701, 0.00700, 0.00700, 0.00700]
    m2h = [0.00660, 0.00660, 0.00660, 0.00660, 0.00660]
    
    # Ice temperatures and inv angle for freqs
    Ti_amsrv = np.full(9, 260)
    Ti_amsrh = np.full(9, 260)
    theta = 0
    e_icev = np.full(9, 0.9)
    e_iceh = np.full(9, 0.9)
    
    # --------------------------------------
    # Here the real simulation begins
    TD=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TU=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AO=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AL=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    tau=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TBU=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TBD=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    llambda=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    epsilon=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    rho_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    rho_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    R_0H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_0V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_geoH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_geoV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    F_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    F_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    OmegaH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    OmegaV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    T_BOmegaH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    T_BOmegaV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    emissivityh=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    emissivityv=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    term=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Delta_S2=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Tv=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Th=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    T_C=2.7

    if c_ice < 0.0: c_ice=0.0
    if c_ice > 1.0: c_ice=1.0

    Ts_mix=c_ice*Ta+(1.0-c_ice)*Ts

    Tl=(Ts_mix+273.0)/2.0

    #eq 27
    Tv=273.16+0.8337*V-3.029E-5*(V**3.33)
    if V > 48:  Tv=301.16
    G = 1.05*(Ts_mix-Tv)*(1-((Ts_mix-Tv)**2)/1200.0)
    if math.fabs(Ts_mix-Tv) > 20: G = (Ts_mix-Tv)*14/math.fabs(Ts_mix-Tv)

    epsilon_R=4.44 # this value is from wentz and meisner, 2000, p. 28
    s=35.0
    ny=0.012 # Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    light_speed=3.00E10
    free_space_permittivity=8.854E-12
    #eq 43
    epsilon_S = (87.90*math.exp(-0.004585*(Ts-273.15)))*(math.exp(-3.45E-3*s+4.69E-6*s**2+1.36E-5*s*(Ts-273.15)))
    #eq 44
    lambda_R = (3.30*math.exp(-0.0346*(Ts-273.15)+0.00017*(Ts-273.15)**2))-(6.54E-3*(1-3.06E-2*(Ts-273.15)+2.0E-4*(Ts-273.15)**2)*s)
    #eq 41
    C=0.5536*s
    #eq 42
    delta_t=25.0-(Ts-273.15)
    #eq 40
    qsi=2.03E-2+1.27E-4*delta_t+2.46E-6*delta_t**2-C*(3.34E-5-4.60E-7*delta_t+4.60E-8*delta_t**2)
    #eq 39
    sigma=3.39E9*(C**0.892)*math.exp(-delta_t*qsi)


    for i in range(0,5):
        #eq26
        TD[i]=b0[i]+b1[i]*V+b2[i]*V**2+b3[i]*V**3+b4[i]*V**4+b5[i]*G
        TU[i]=TD[i]+b6[i]+b7[i]*V
        #eq 28
        AO[i]=ao1[i]+ao2[i]*(TD[i]-270.0)
        #eq 29
        AV[i]=av1[i]*V+av2[i]*V**2
        #eq 33
        AL[i]=aL1[i]*(1.0-aL2[i]*(Tl-283.0))*L
        #eq 22
        tau[i]=math.exp((-1.0/math.cos(math.radians(theta)))*(AO[i]+AV[i]+AL[i])) 
        #eq 24
        TBU[i]=TU[i]*(1.0-tau[i])
        TBD[i]=TD[i]*(1.0-tau[i])

        llambda[i]=(light_speed/(frequencies[i]*1E9))

        #eq 35
        epsilon[i]=epsilon_R+((epsilon_S-epsilon_R)/(1.0+((cmath.sqrt(-1)*lambda_R)/llambda[i])**(1.0-ny)))-((2.0*cmath.sqrt(-1)*sigma*llambda[i])/light_speed)
        #eq.45
        rho_H[i]=(math.cos(math.radians(theta))-cmath.sqrt(epsilon[i]-math.sin(math.radians(theta))**2))/\
                 (math.cos(math.radians(theta))+cmath.sqrt(epsilon[i]-math.sin(math.radians(theta))**2))
        rho_V[i]=(epsilon[i]*math.cos(math.radians(theta))-cmath.sqrt(epsilon[i]-math.sin(math.radians(theta))**2))/\
                 (epsilon[i]*math.cos(math.radians(theta))+cmath.sqrt(epsilon[i]-math.sin(math.radians(theta))**2))
        #eq46
        R_0H[i]=np.absolute(rho_H[i])**2
        R_0V[i]=np.absolute(rho_V[i])**2+(4.887E-8-6.108E-8*(Ts-273.0)**3)

        #eq 57
        R_geoH[i]=R_0H[i]-(r0h[i]+r1h[i]*(theta-53.0)+r2h[i]*(Ts-288.0)+r3h[i]*(theta-53.0)*(Ts-288.0))*W
        R_geoV[i]=R_0V[i]-(r0v[i]+r1v[i]*(theta-53.0)+r2v[i]*(Ts-288.0)+r3v[i]*(theta-53.0)*(Ts-288.0))*W
        #eq.60
        W_1=7.0
        W_2=12.0
        if W<W_1: F_H[i]=m1h[i]*W
        elif W_1<W<W_2: F_H[i]=m1h[i]*W+0.5*(m2h[i]-m1h[i])*((W-W_1)**2)/(W_2-W_1)
        else: F_H[i]=m2h[i]*W-0.5*(m2h[i]-m1h[i])*(W_2+W_1)
        W_1=3.0
        W_2=12.0
        if W<W_1: F_V[i]=m1v[i]*W
        elif W_1<W<W_2: F_V[i]=m1v[i]*W+0.5*(m2v[i]-m1v[i])*((W-W_1)**2)/(W_2-W_1)
        else: F_V[i]=m2v[i]*W-0.5*(m2v[i]-m1v[i])*(W_2+W_1)

        R_H[i]=(1-F_H[i])*R_geoH[i]
        R_V[i]=(1-F_V[i])*R_geoV[i]

        emissivityh[i]=1-R_H[i]
        emissivityv[i]=1-R_V[i]
        
        # Something is going on here!! 
        if i >= 2:  # Adapted to NEMS now, was 4 before
            Delta_S2[i]=5.22E-3*W
        else:
            Delta_S2[i]=5.22E-3*(1-0.00748*(37.0-frequencies[i])**1.3)*W
        if Delta_S2[i]>0.069: 
            Delta_S2[i]=0.069
        
        #eq.62
        term[i]=Delta_S2[i]-70.0*Delta_S2[i]**3
        OmegaH[i]=(6.2-0.001*(37.0-frequencies[i])**2)*term[i]*tau[i]**2.0
        OmegaV[i]=(2.5+0.018*(37.0-frequencies[i]))*term[i]*tau[i]**3.4
        #eq.61
        T_BOmegaH[i]=((1+OmegaH[i])*(1-tau[i])*(TD[i]-T_C)+T_C)*R_H[i] 
        T_BOmegaV[i]=((1+OmegaV[i])*(1-tau[i])*(TD[i]-T_C)+T_C)*R_V[i]
        
    #ch1
    Th22=TBU[0]+tau[0]*((1.0 - c_ice)*emissivityh[0]*Ts+c_ice*e_iceh[0]*Ti_amsrh[0]+(1.0 - c_ice)*(1.0 - emissivityh[0])*\
             (T_BOmegaH[0]+tau[0]*T_C)+c_ice*(1.0 - e_iceh[0])*(TBD[0]+tau[0]*T_C))

    Tv22=TBU[0]+tau[0]*((1.0 - c_ice)*emissivityv[0]*Ts+c_ice*e_icev[0]*Ti_amsrv[0]+(1.0 - c_ice)*(1.0 - emissivityv[0])*\
         (T_BOmegaV[0]+tau[0]*T_C)+c_ice*(1.0 - e_icev[0])*(TBD[0]+tau[0]*T_C))
    
    T22=Tv22*(math.sin(math.radians(theta)))**2 + Th22*(math.cos(math.radians(theta)))**2
        
        
    # ch2
    Th31=TBU[1]+tau[1]*((1.0 - c_ice)*emissivityh[1]*Ts+c_ice*e_iceh[1]*Ti_amsrh[1]+(1.0 - c_ice)*(1.0 - emissivityh[1])*\
         (T_BOmegaH[1]+tau[1]*T_C)+c_ice*(1.0 - e_iceh[1])*(TBD[1]+tau[1]*T_C))

    Tv31=TBU[1]+tau[1]*((1.0 - c_ice)*emissivityv[1]*Ts+c_ice*e_icev[1]*Ti_amsrv[1]+(1.0 - c_ice)*(1.0 - emissivityv[1])*\
         (T_BOmegaV[1]+tau[1]*T_C)+c_ice*(1.0 - e_icev[1])*(TBD[1]+tau[1]*T_C))
        
    T31=Tv31*(math.sin(math.radians(theta)))**2 + Th31*(math.cos(math.radians(theta)))**2

    Tb=np.array([T22, T31])
    
    return Tb


class RadiativeTransferModel:
    """
    AMSR Ocean Algorithm RTM with integrated XGBoost bias correction
    Frank J. Wentz, Thomas Meissner; Remote Sensing Systems; Version 2
    Modified to NIMBUS 6 SCAMS geometry
    """
    
    def __init__(self, apply_correction=True, region='arctic'):
        self.frequencies = np.array([22.235, 31.400, 53.650, 54.900, 58.800])
        self.T_C = 2.7
        self._init_coefficients()
        
        # Store bias correction models - initially None
        self.arctic_model_22 = None
        self.arctic_model_31 = None
        self.antarctic_model_22 = None
        self.antarctic_model_31 = None
        
        self.apply_correction = apply_correction
        self.region = region  # 'arctic' or 'antarctic'
    
    def set_bias_models(self, arctic_22_path, arctic_31_path, 
                        antarctic_22_path, antarctic_31_path):
        """
        Load XGBoost bias correction models from pickle files.
        
        FIX: Actually load the pickle files instead of storing paths as strings!
        
        Args:
            arctic_22_path: Path to Arctic 22 GHz model pickle file
            arctic_31_path: Path to Arctic 31 GHz model pickle file
            antarctic_22_path: Path to Antarctic 22 GHz model pickle file
            antarctic_31_path: Path to Antarctic 31 GHz model pickle file
        """
        print("Loading XGBoost bias correction models...")

        def _load(path, label):
            """Load one model and record how many features it actually
            expects, so a schema mismatch is caught here -- once, loudly --
            instead of silently inside every simulate() call."""
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                print(f"  \u2717 Failed to load {label}: {e}")
                return None

            n_expected = getattr(model, 'n_features_in_', None)
            names = getattr(model, 'feature_names_in_', None)
            if n_expected is None and hasattr(model, 'get_booster'):
                n_expected = model.get_booster().num_features()
                names = model.get_booster().feature_names
            print(f"  \u2713 {label} loaded ({type(model).__name__}), "
                  f"expects {n_expected} feature(s)"
                  + (f": {list(names)}" if names else ""))
            model._expected_n_features = n_expected  # stashed for the runtime check below
            return model

        self.arctic_model_22 = _load(arctic_22_path, "Arctic 22 GHz model")
        self.arctic_model_31 = _load(arctic_31_path, "Arctic 31 GHz model")
        self.antarctic_model_22 = _load(antarctic_22_path, "Antarctic 22 GHz model")
        self.antarctic_model_31 = _load(antarctic_31_path, "Antarctic 31 GHz model")

        for label, model in (("Arctic 22 GHz", self.arctic_model_22),
                             ("Arctic 31 GHz", self.arctic_model_31),
                             ("Antarctic 22 GHz", self.antarctic_model_22),
                             ("Antarctic 31 GHz", self.antarctic_model_31)):
            if model is not None and getattr(model, '_expected_n_features', None) != 6:
                print(f"  \u26a0 {label} expects "
                      f"{getattr(model, '_expected_n_features', 'unknown')} features, "
                      f"but simulate() currently builds 6 ([V, W, L, Ta, Ts, c_ice]). "
                      f"Bias correction WILL fail for this model until the feature "
                      f"vector in _apply_bias_correction() is rebuilt to match -- "
                      f"see inspect_bias_models.py.")

        print()
    
    def set_region(self, region):
        """Set region for bias correction: 'arctic' or 'antarctic'"""
        if region not in ['arctic', 'antarctic']:
            raise ValueError("Region must be 'arctic' or 'antarctic'")
        self.region = region
    
    def _init_coefficients(self):
        """Initialize model coefficients"""
        self.b0 = [241.69, 239.45, 245.87, 245.87, 245.87]
        self.b1 = [3.1032, 2.5441, 2.5061, 2.5061, 2.5061]
        self.b2 = [-0.081429, -0.051284, -0.062789, -0.062789, -0.062789]
        self.b3 = [0.00099893, 0.00045202, 0.00075962, 0.00075962, 0.00075962]
        self.b4 = [-4.837e-06, -1.436e-06, -3.606e-06, -3.606e-06, -3.606e-06]
        self.b5 = [0.20, 0.58, 0.53, 0.53, 0.53]
        self.b6 = [-0.20, -0.57, -12.52, -12.52, -12.52]
        self.b7 = [-0.0521, -0.0238, -0.2326, -0.2326, -0.2326]
        
        self.ao1 = [0.01575, 0.04006, 1.13176, 1.13176, 1.13176]
        self.ao2 = [-0.000087, -0.000200, -0.000226, -0.000226, -0.000226]
        self.av1 = [0.00514, 0.00188, 0.00317, 0.00317, 0.00317]
        self.av2 = [0.19e-5, 0.09e-5, 0.27e-5, 0.27e-5, 0.27e-5]
        
        self.aL1 = [0.0891, 0.2027, 0.4021, 0.4021, 0.4021]
        self.aL2 = [0.0281, 0.0261, 0.0231, 0.0231, 0.0231]
        
        self.r0v = [-0.00063, -0.00101, -0.00123, -0.00123, -0.00123]
        self.r0h = [0.00139, 0.00191, 0.00197, 0.00197, 0.00197]
        self.r1v = [-0.000070, -0.000105, -0.000113, -0.000113, -0.000113]
        self.r1h = [0.000085, 0.000112, 0.000119, 0.000119, 0.000119]
        self.r2v = [-2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05]
        self.r2h = [-4.195e-05, -5.451e-05, -5.5e-05, -5.5e-05, -5.5e-05]
        self.r3v = [0.41e-06, 0.45e-06, 0.32e-06, 0.32e-06, 0.32e-06]
        self.r3h = [-0.20e-06, -0.36e-06, -0.44e-06, -0.44e-06, -0.44e-06]
        
        self.m1v = [0.00178, 0.00257, 0.00260, 0.00260, 0.00260]
        self.m1h = [0.00308, 0.00329, 0.00330, 0.00330, 0.00330]
        self.m2v = [0.00730, 0.00701, 0.00700, 0.00700, 0.00700]
        self.m2h = [0.00660, 0.00660, 0.00660, 0.00660, 0.00660]
    
    def _apply_bias_correction(self, Tb_22, Tb_31, V, W, L, Ta, Ts, c_ice):
        """
        Apply XGBoost bias correction to brightness temperatures.
        
        The models predict bias values that are subtracted from simulated TB.
        
        Args:
            Tb_22, Tb_31: Simulated brightness temperatures (K)
            V, W, L, Ta, Ts, c_ice: Input atmospheric/surface parameters
            
        Returns:
            Tuple of (Tb_22_corrected, Tb_31_corrected)
        """
        # If corrections disabled, return uncorrected values
        if not self.apply_correction:
            return Tb_22, Tb_31
        
        # Select models based on region
        if self.region == 'arctic':
            model_22 = self.arctic_model_22
            model_31 = self.arctic_model_31
        else:  # antarctic
            model_22 = self.antarctic_model_22
            model_31 = self.antarctic_model_31
        
        # If models not loaded, return uncorrected
        if model_22 is None or model_31 is None:
            return Tb_22, Tb_31
        
        # TODO: [V, W, L, Ta, Ts, c_ice] is 6 features. If the loaded model
        # expects a different count (see the load-time warning printed by
        # set_bias_models(), or run inspect_bias_models.py), this vector is
        # INCOMPLETE and needs the missing engineered features added here,
        # in the exact order the model was trained on -- guessing the order
        # or the missing columns would make predict() return a number
        # without erroring, which is worse than the loud failure below.
        features = np.array([[V, W, L, Ta, Ts, c_ice]])

        n_expected = getattr(model_22, '_expected_n_features', None)
        if n_expected is not None and features.shape[1] != n_expected:
            raise RuntimeError(
                f"Bias model for region='{self.region}' expects "
                f"{n_expected} features but simulate() built "
                f"{features.shape[1]} ({list(features[0])}). Fix the "
                f"feature vector in _apply_bias_correction() before "
                f"re-running -- silently falling back here would produce "
                f"output that looks bias-corrected but isn't, for every "
                f"pixel processed. See inspect_bias_models.py to recover "
                f"the model's expected feature order."
            )

        # Predict biases using XGBoost models
        bias_22 = model_22.predict(features)[0]
        bias_31 = model_31.predict(features)[0]

        # Apply bias correction (subtract predicted bias from simulated TB)
        Tb_22_corrected = Tb_22 - bias_22
        Tb_31_corrected = Tb_31 - bias_31

        return Tb_22_corrected, Tb_31_corrected
    
    def simulate(self, V, W, L, Ta, Ts, c_ice, theta=0,
                 Ti_amsrv=None, Ti_amsrh=None, e_icev=None, e_iceh=None):
        """
        Simulate brightness temperatures using AMSR RTM with optional bias correction.
        
        Args:
            V: Water vapor (mm)
            W: Wind speed (m/s)
            L: Liquid water (mm)
            Ta: Air temperature (K)
            Ts: Sea surface temperature (K)
            c_ice: Ice concentration (0-1)
            theta: Incidence angle (degrees, default 0)
            Ti_amsrv: Ice temperature vertical (K)
            Ti_amsrh: Ice temperature horizontal (K)
            e_icev: Ice emissivity vertical (0-1)
            e_iceh: Ice emissivity horizontal (0-1)
            
        Returns: 
            Tuple (Tb_22, Tb_31) in Kelvin
            Bias corrections applied if models loaded and enabled
        """
        if Ti_amsrv is None:
            Ti_amsrv = np.full(5, 260)
        if Ti_amsrh is None:
            Ti_amsrh = np.full(5, 260)
        if e_icev is None:
            e_icev = np.full(5, 0.9)
        if e_iceh is None:
            e_iceh = np.full(5, 0.9)
        
        # Initialize arrays
        n_freq = 5
        TD = np.zeros(n_freq)
        TU = np.zeros(n_freq)
        AO = np.zeros(n_freq)
        AV = np.zeros(n_freq)
        AL = np.zeros(n_freq)
        tau = np.zeros(n_freq)
        TBU = np.zeros(n_freq)
        TBD = np.zeros(n_freq)
        
        # Clamp ice concentration
        c_ice = np.clip(c_ice, 0.0, 1.0)
        
        # Mix surface temperature based on ice concentration
        Ts_mix = c_ice * Ta + (1.0 - c_ice) * Ts
        Tl = (Ts_mix + 273.0) / 2.0
        
        # Equation 27
        Tv = 273.16 + 0.8337 * V - 3.029e-5 * (V**3.33)
        if V > 48:
            Tv = 301.16
        G = 1.05 * (Ts_mix - Tv) * (1 - ((Ts_mix - Tv)**2) / 1200.0)
        if abs(Ts_mix - Tv) > 20:
            G = (Ts_mix - Tv) * 14 / abs(Ts_mix - Tv)
        
        # Seawater dielectric properties
        epsilon_R = 4.44
        s = 35.0
        ny = 0.012
        light_speed = 3.00e10
        
        epsilon_S = (87.90 * math.exp(-0.004585 * (Ts - 273.15))) * \
                   (math.exp(-3.45e-3 * s + 4.69e-6 * s**2 + 1.36e-5 * s * (Ts - 273.15)))
        
        lambda_R = (3.30 * math.exp(-0.0346 * (Ts - 273.15) + 0.00017 * (Ts - 273.15)**2)) - \
                  (6.54e-3 * (1 - 3.06e-2 * (Ts - 273.15) + 2.0e-4 * (Ts - 273.15)**2) * s)
        
        C = 0.5536 * s
        delta_t = 25.0 - (Ts - 273.15)
        qsi = 2.03e-2 + 1.27e-4 * delta_t + 2.46e-6 * delta_t**2 - \
              C * (3.34e-5 - 4.60e-7 * delta_t + 4.60e-8 * delta_t**2)
        sigma = 3.39e9 * (C**0.892) * math.exp(-delta_t * qsi)
        
        # Compute for each frequency
        Tb_list = []
        for i in range(2):  # Only first 2 frequencies (22.235, 31.400 GHz)
            TD[i] = self.b0[i] + self.b1[i]*V + self.b2[i]*V**2 + self.b3[i]*V**3 + \
                   self.b4[i]*V**4 + self.b5[i]*G
            TU[i] = TD[i] + self.b6[i] + self.b7[i]*V
            
            AO[i] = self.ao1[i] + self.ao2[i] * (TD[i] - 270.0)
            AV[i] = self.av1[i] * V + self.av2[i] * V**2
            AL[i] = self.aL1[i] * (1.0 - self.aL2[i] * (Tl - 283.0)) * L
            
            tau[i] = math.exp((-1.0 / math.cos(math.radians(theta))) * (AO[i] + AV[i] + AL[i]))
            TBU[i] = TU[i] * (1.0 - tau[i])
            TBD[i] = TD[i] * (1.0 - tau[i])
            
            llambda = light_speed / (self.frequencies[i] * 1e9)
            
            epsilon = epsilon_R + ((epsilon_S - epsilon_R) / 
                     (1.0 + ((cmath.sqrt(-1) * lambda_R) / llambda)**(1.0 - ny))) - \
                     ((2.0 * cmath.sqrt(-1) * sigma * llambda) / light_speed)
            
            rho_H = (math.cos(math.radians(theta)) - cmath.sqrt(epsilon - math.sin(math.radians(theta))**2)) / \
                   (math.cos(math.radians(theta)) + cmath.sqrt(epsilon - math.sin(math.radians(theta))**2))
            rho_V = (epsilon * math.cos(math.radians(theta)) - cmath.sqrt(epsilon - math.sin(math.radians(theta))**2)) / \
                   (epsilon * math.cos(math.radians(theta)) + cmath.sqrt(epsilon - math.sin(math.radians(theta))**2))
            
            R_0H = abs(rho_H)**2
            R_0V = abs(rho_V)**2 + (4.887e-8 - 6.108e-8 * (Ts - 273.0)**3)
            
            R_geoH = R_0H - (self.r0h[i] + self.r1h[i]*(theta-53.0) + 
                           self.r2h[i]*(Ts-288.0) + self.r3h[i]*(theta-53.0)*(Ts-288.0)) * W
            R_geoV = R_0V - (self.r0v[i] + self.r1v[i]*(theta-53.0) + 
                           self.r2v[i]*(Ts-288.0) + self.r3v[i]*(theta-53.0)*(Ts-288.0)) * W
            
            # Wind modulation
            if W < 7.0:
                F_H = self.m1h[i] * W
                F_V = self.m1v[i] * W
            elif 7.0 <= W <= 12.0:
                F_H = self.m1h[i]*W + 0.5*(self.m2h[i]-self.m1h[i])*((W-7.0)**2)/5.0
                F_V = self.m1v[i]*W + 0.5*(self.m2v[i]-self.m1v[i])*((W-3.0)**2)/9.0
            else:
                F_H = self.m2h[i]*W - 0.5*(self.m2h[i]-self.m1h[i])*19.0
                F_V = self.m2v[i]*W - 0.5*(self.m2v[i]-self.m1v[i])*15.0
            
            R_H = (1 - F_H) * R_geoH
            R_V = (1 - F_V) * R_geoV
            
            emissivityh = 1 - R_H
            emissivityv = 1 - R_V
            
            Delta_S2 = 5.22e-3 * W if i >= 2 else 5.22e-3 * (1 - 0.00748*(37.0-self.frequencies[i])**1.3) * W
            Delta_S2 = min(Delta_S2, 0.069)
            
            term = Delta_S2 - 70.0 * Delta_S2**3
            OmegaH = (6.2 - 0.001*(37.0-self.frequencies[i])**2) * term * tau[i]**2.0
            OmegaV = (2.5 + 0.018*(37.0-self.frequencies[i])) * term * tau[i]**3.4
            
            T_BOmegaH = ((1 + OmegaH) * (1 - tau[i]) * (TD[i] - self.T_C) + self.T_C) * R_H
            T_BOmegaV = ((1 + OmegaV) * (1 - tau[i]) * (TD[i] - self.T_C) + self.T_C) * R_V
            
            Th = TBU[i] + tau[i] * ((1.0 - c_ice) * emissivityh * Ts + c_ice * e_iceh[i] * Ti_amsrh[i] + 
                                   (1.0 - c_ice) * (1.0 - emissivityh) * (T_BOmegaH + tau[i] * self.T_C) +
                                   c_ice * (1.0 - e_iceh[i]) * (TBD[i] + tau[i] * self.T_C))
            
            Tv = TBU[i] + tau[i] * ((1.0 - c_ice) * emissivityv * Ts + c_ice * e_icev[i] * Ti_amsrv[i] +
                                   (1.0 - c_ice) * (1.0 - emissivityv) * (T_BOmegaV + tau[i] * self.T_C) +
                                   c_ice * (1.0 - e_icev[i]) * (TBD[i] + tau[i] * self.T_C))
            
            T = Tv * (math.sin(math.radians(theta)))**2 + Th * (math.cos(math.radians(theta)))**2
            Tb_list.append(T)
        
        # Apply bias correction if models are available and enabled
        Tb_22_corrected, Tb_31_corrected = self._apply_bias_correction(
            Tb_list[0], Tb_list[1], V, W, L, Ta, Ts, c_ice
        )
        
        # Return as tuple (not array)
        return Tb_22_corrected, Tb_31_corrected