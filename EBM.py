"""
Definition of the 0-D EBM that is featured in our thesis work.
"""
import numpy as np
import xarray as xr
from tqdm import tqdm

# Standard Parameter Choices

TSI = 1367  # TSI in units of WM-2
OLR = 239  # OLR from space in WM-2
global_mean_temp = 288  # K
boltzman = 5.67 * 10**-8
emissitivity = OLR / (
    boltzman * global_mean_temp**4
)  # Computing emissitivity factor as a model fit
capacity = (
    105 * 10**5
)  # From Dijkstra 'Nonlinear Climate Dynamics' page 267, units are Jm-2K-2
a0 = 0.5  # Lower albedo is a0 - a1/2
a1 = 0.4  # Upper albedo is a0 + a1/2
T_ref = 270

# Model Definition


# Outgoing Radiation via Stefan-Boltzman Law with greenhouse included
def R_O(T, emissitivity=emissitivity):
    return emissitivity * boltzman * T**4


# Incoming Radiation caputres Ice-Albedo Effect
def albedo(T, a0=a0, a1=a1, T_ref=T_ref):
    return a0 - a1 / 2 * np.tanh(T - T_ref)


def R_I(T, TSI=TSI, a0=a0, a1=a1, T_ref=T_ref):
    return TSI / 4 * (1 - albedo(T, a0=a0, a1=a1, T_ref=T_ref))


# Heat Balance
def ebm_rhs(T, TSI=TSI, a0=a0, a1=a1, T_ref=T_ref, emissitivity=emissitivity):
    return R_I(T, TSI=TSI, a0=a0, a1=a1, T_ref=T_ref) - R_O(
        T, emissitivity=emissitivity
    )


# Potential of System
def potential(T, TSI=TSI, a0=a0, a1=a1, T_ref=T_ref, emissitivity=emissitivity):
    R_i_term = TSI / 4 * (T - a0 * T + a1 / 2 * np.log(np.cosh(T - T_ref)))
    R_o_term = emissitivity * boltzman * T**5 / 5
    return R_i_term - R_o_term


class EBMTrajectoryObserver:
    """Observes the trajectory of EBM. Dumps to netcdf."""

    def __init__(self, integrator, name="OD_EBM"):
        """param, integrator: integrator being observed."""

        # Needed knowledge of the integrator
        self._parameters = integrator.parameters

        # Trajectory Observation logs
        self.time_obs = []  # Times we've made observations
        self.T_obs = []
        self.dump_count = 0

    def look(self, integrator):
        """Observes trajectory"""

        # Note the time
        self.time_obs.append(integrator.time)

        # Making Observations
        self.T_obs.append(integrator.state[0].copy())
        return

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if len(self.T_obs) == 0:
            print("I have no observations! :(")
            return

        dic = {}
        _time = self.time_obs
        dic["Temp"] = xr.DataArray(
            self.T_obs, dims=["time"], name="Temp", coords={"time": _time}
        )
        return xr.Dataset(dic, attrs=self._parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.T_obs = []

    def dump(self, save_name):
        """Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if len(self.T_obs) == 0:
            print("I have no observations! :(")
            return

        self.observations.to_netcdf(save_name)
        print(f"Observations written to {save_name}. Erasing personal log.\n")
        self.wipe()
        self.dump_count += 1


def make_observations(runner, looker, obs_num, obs_freq, noprog=True):
    """Makes observations given runner and looker.
    runner, integrator object.
    looker, observer object.
    obs_num, how many observations you want.
    obs_freq, adimensional time between observations"""
    looker.look(runner)  # Look at IC
    for step in tqdm(np.repeat(obs_freq, obs_num), disable=noprog):
        runner.run(obs_freq)
        looker.look(runner)
