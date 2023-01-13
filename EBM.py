"""
Definition of the 0-D EBM that is featured in our thesis work.
"""
##############################################################################
##### Imports
##############################################################################

import numpy as np

##############################################################################
##### Standard Parameter Choices
##############################################################################

TSI = 1367 # TSI in units of WM-2
OLR = 239 # OLR from space in WM-2
global_mean_temp = 288 # K
boltzman = 5.67 * 10**-8
emissitivity = OLR/(boltzman * global_mean_temp **4) # Computing emissitivity factor as a model fit
capacity = 105 * 10**5 # From Dijkstra 'Nonlinear Climate Dynamics' page 267, units are Jm-2K-2
a0 = 0.5 # Lower albedo is a0 - a1/2
a1 = 0.4 # Upper albedo is a0 + a1/2
T_ref = 270

##############################################################################
##### Model Definition
##############################################################################

# Outgoing Radiation Parameterised via Stefan-Boltzman Law with greenhouse included
def R_O(T, emissitivity=emissitivity):
    return emissitivity * boltzman * T**4

# Incoming Radiation caputres Ice-Albedo Effect
def albedo(T, a0=a0, a1=a1, T_ref=T_ref):
    return a0 - a1/2 * np.tanh(T - T_ref)

def R_I(T, TSI=TSI, a0=a0, a1=a1, T_ref=T_ref):
    return TSI/4 * (1 - albedo(T, a0=a0, a1=a1, T_ref=T_ref))

# Heat Balance
def ebm_rhs(t, T, TSI=TSI, a0=a0, a1=a1, T_ref=T_ref, emissitivity=emissitivity):
    return R_I(T, TSI=TSI, a0=a0, a1=a1, T_ref=T_ref) - R_O(T, emissitivity=emissitivity)

# Potential of System
def potential(T, TSI=TSI, a0=a0, a1=a1, T_ref=T_ref, emissitivity=emissitivity):
    R_i_term = TSI/4 * (T - a0 *T + a1/2 * np.log( np.cosh(T - T_ref) ) )
    R_o_term = emissitivity * boltzman * T**5 /5
    return (R_i_term - R_o_term)