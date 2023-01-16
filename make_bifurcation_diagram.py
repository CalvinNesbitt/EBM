# Imports
from EBM import ebm_rhs, emissitivity, a0, a1, make_observations, EBMTrajectoryObserver
from plotting import init_2d_fax
from detDyn.dynamics.integrator import odeIntegrator
from m_state import find_m_states

from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm

# Function for Finding Attractors


def bistable(TSI):
    return len(find_attractors(TSI)) == 2


def find_attractors(TSI):
    "For a specified TSI, finds the attractors of the 0D EBM."

    # Standard Parameter choice
    EBM_parameters = {"TSI": TSI, "emissitivity": emissitivity, "a0": a0, "a1": a1}

    # Setup integrators
    sb_integrator = odeIntegrator(ebm_rhs, np.array([200]), EBM_parameters)
    w_integrator = odeIntegrator(ebm_rhs, np.array([300]), EBM_parameters)

    # Run Integrations to find attractors
    attractor_list = []
    for integrator in [sb_integrator, w_integrator]:
        looker = EBMTrajectoryObserver(integrator)
        make_observations(integrator, looker, 500, 0.1, noprog=True)
        attractor_list.append(looker.observations.Temp.isel(time=-1).item())
    return np.unique(np.array(attractor_list).round(decimals=3))


def attractor_TSI_search(TSIs, timer=True):
    "Search for attractors for different TSI Values"

    # Initialise Lists
    sb_TSIs = []
    sb_attractors = []
    w_TSIs = []
    w_attractors = []

    for TSI in tqdm(TSIs, disable=not timer):
        attractors = find_attractors(TSI)
        if len(attractors) == 2:  # Bistable case
            sb_TSIs.append(TSI)
            sb_attractors.append(np.min(attractors))
            w_TSIs.append(TSI)
            w_attractors.append(np.max(attractors))
        elif attractors[0] < 270:
            sb_TSIs.append(TSI)
            sb_attractors.append(np.min(attractors))
        elif attractors[0] > 270:
            w_TSIs.append(TSI)
            w_attractors.append(np.min(attractors))

    return [(sb_TSIs, sb_attractors), (w_TSIs, w_attractors)]


def plot_bifurcation_diagram(attractor_info, fax=None):

    # Unpack attractor info
    sb_TSIs, sb_attractors = attractor_info[0]
    w_TSIs, w_attractors = attractor_info[1]
    m_TSIs, m_states = attractor_info[2]

    # Interpolate on to grid
    w_TSI_grid = np.linspace(min(w_TSIs), max(w_TSIs), 100)
    sb_TSI_grid = np.linspace(min(sb_TSIs), max(sb_TSIs), 100)
    m_TSI_grid = np.linspace(min(m_TSIs), max(m_TSIs), 100)
    w_function = interp1d(w_TSIs, w_attractors, kind="cubic")
    sb_function = interp1d(sb_TSIs, sb_attractors, kind="cubic")
    m_function = interp1d(m_TSIs, m_states, kind="cubic")

    # Make Plot
    if fax is None:
        fax = init_2d_fax()
    fig, ax = fax

    # Extend M-state line join up diagram
    joined_m_TSI_grid = np.concatenate([[min(w_TSIs)], m_TSI_grid, [max(sb_TSIs)]])
    joined_m_TSI_values = np.concatenate(
        [[w_function(min(w_TSIs))], m_function(m_TSI_grid), [sb_function(max(sb_TSIs))]]
    )
    ax.plot(joined_m_TSI_grid / 1367, joined_m_TSI_values, c="g", ls="--")

    # Plot Attractors
    ax.plot(w_TSI_grid / 1367, w_function(w_TSI_grid), c="r")
    ax.plot(sb_TSI_grid / 1367, sb_function(sb_TSI_grid), c="b")
    return fax


def make_bifrucation_diagram(TSIs, fax=None, timer=True):
    "Computes attractors and plots bifurcation diagram."
    print("Finding attractors.\n")
    attractor_info = attractor_TSI_search(TSIs, timer=timer)
    print("Finding m_states.\n")
    bistable_TSIs = np.linspace(1250, 2100, 5)
    m_state_info = find_m_states(bistable_TSIs, timer=timer)
    return plot_bifurcation_diagram(attractor_info + [m_state_info], fax=fax)


if __name__ == "__main__":
    # Search for attractors for different TSI Values
    TSIs = 1367 * np.linspace(0.5, 2, 10)
    fig, ax = make_bifrucation_diagram(TSIs)
    ax.set_xlabel("$\\frac{TSI}{TSI_{0}}$")
    ax.set_ylabel("$T$")
    fig

# %%
