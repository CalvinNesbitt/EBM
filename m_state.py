# Imports

from chaos_explorer.integrator import OdeIntegrator
from chaos_explorer.m_state.bisectionAlgorithm import BisectionAlgorithm
from EBM import emissitivity, a0, a1, ebm_rhs
import numpy as np
from tqdm import tqdm

# Functions needed for m_state algorithm


def check_if_ebm_ic_cold(ic, integrator):
    """
    Checks whether a given ic ends up at the cold point.
    """

    integrator.state = ic
    tau = 0.1  # How long we integrate between checks, will effect how efficient we are

    for i in range(1000):  # How many checks we make
        integrator.run(tau)
        if (
            integrator.state[0] < 260
        ):  # Threshold for being cold, ensure cold ic matches this
            return True
        elif integrator.state[0] > 280:  # Threshold for being hot
            return False
    return None


def heat_up(x):
    x += 0.01


def cool_down(x):
    x -= 0.01


# Functions for finding m_states


def find_m_state(TSI):

    # Standard Parameter choice
    EBM_parameters = {"TSI": TSI, "emissitivity": emissitivity, "a0": a0, "a1": a1}

    # Setup integrator
    ebm_integrator = OdeIntegrator(ebm_rhs, np.array([280]), EBM_parameters)

    # Setup mstate algorithm
    mstate = BisectionAlgorithm(
        ebm_integrator,
        check_if_ebm_ic_cold,
        cool_down,
        heat_up,
        0.001,
        np.array([[200], [300]]),
    )
    mstate.run(200)
    return mstate.midpoint


def find_m_states(TSIs, timer=True):
    return (TSIs, [find_m_state(x)[0] for x in tqdm(TSIs, disable=not timer)])
