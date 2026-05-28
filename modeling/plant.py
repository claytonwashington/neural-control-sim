"""Cleo E/I plant for digital twin modeling.

Builds a 3D LIF excitatory/inhibitory network with:
- ProportionalCurrentOpsin (control-affine)
- 2 OpticFiber light sources in opposing quadrants
- 50-channel MultiUnitActivity probe

Usage:
    from modeling.plant import build_plant
    sim, devices = build_plant()
    sim.run(100 * ms)
"""

from __future__ import annotations

import numpy as np
import brian2.only as b2
from brian2 import (
    ms,
    mV,
    Mohm,
    pA,
    nA,
    mm,
    um,
    mwatt,
    nmeter,
)

import cleo
import cleo.coords
import cleo.ephys
import cleo.light
import cleo.opto


def _make_probe_coords(n_channels: int, volume_um: float) -> np.ndarray:
    """Generate 3D probe coordinates spanning the volume.

    Places channels in a grid pattern across the XZ plane at Y=0,
    roughly mimicking a planar MEA.

    Parameters
    ----------
    n_channels : int
        Number of recording channels
    volume_um : float
        Side length of the cubic volume in microns

    Returns
    -------
    coords : ndarray, shape (n_channels, 3)
        Channel coordinates in microns
    """
    half = volume_um / 2
    # Grid along X and Z, Y=0 (planar MEA)
    n_side = int(np.ceil(np.sqrt(n_channels)))
    xs = np.linspace(-half * 0.8, half * 0.8, n_side)
    zs = np.linspace(-half * 0.8, half * 0.8, n_side)
    xx, zz = np.meshgrid(xs, zs)
    coords = np.column_stack([
        xx.ravel()[:n_channels],
        np.zeros(n_channels),
        zz.ravel()[:n_channels],
    ])
    return coords


def build_plant(
    n_exc: int = 800,
    n_inh: int = 200,
    n_channels: int = 50,
    volume_um: float = 500.0,
    I_per_Irr_val: float = 50e-12,  # 50 pA per mW/mm^2
    p_connect: float = 0.1,
    seed: int = 42,
) -> tuple[cleo.CLSimulator, dict]:
    """Build the Cleo E/I plant for digital twin experiments.

    Parameters
    ----------
    n_exc : int
        Number of excitatory neurons
    n_inh : int
        Number of inhibitory neurons
    n_channels : int
        Number of MUA probe channels
    volume_um : float
        Side length of cubic volume in microns
    I_per_Irr_val : float
        Current per unit irradiance in amps / (mW/mm^2).
        Default 50 pA/(mW/mm^2) gives ~50-500 pA at 1-10 mW/mm^2.
    p_connect : float
        Connection probability for recurrent synapses
    seed : int
        Random seed

    Returns
    -------
    sim : cleo.CLSimulator
        The configured simulator (no IOProcessor set yet)
    devices : dict
        References to key devices: 'light', 'opsin', 'probe', 'mua',
        'ng', 'exc_syn', 'inh_syn'
    """
    np.random.seed(seed)

    n_total = n_exc + n_inh
    half = volume_um / 2

    # -------------------------------------------------------------------------
    # Network: LIF E/I population
    # -------------------------------------------------------------------------
    b2.start_scope()

    ng = b2.NeuronGroup(
        n_total,
        """
        dv/dt = (-(v - E_L) + Rm * (I_syn + I_opto + I_bg)) / tau_m : volt
        I_syn : amp
        I_opto : amp
        """,
        threshold="v > v_th",
        reset="v = E_L",
        refractory=2 * ms,
        method="euler",
        namespace={
            "tau_m": 20 * ms,
            "Rm": 500 * Mohm,
            "E_L": -70 * mV,
            "v_th": -50 * mV,
            "I_bg": 60 * pA,
        },
    )
    ng.v = "E_L + rand() * (v_th - E_L)"  # random initial voltages

    # Excitatory synapses (first n_exc neurons -> all)
    exc_syn = b2.Synapses(
        ng[:n_exc],
        ng,
        on_pre="I_syn_post += w_exc",
        namespace={"w_exc": 0.1 * mV / (500 * Mohm)},  # ~0.2 pA
    )
    exc_syn.connect(p=p_connect)

    # Inhibitory synapses (last n_inh neurons -> all)
    inh_syn = b2.Synapses(
        ng[n_exc:],
        ng,
        on_pre="I_syn_post -= w_inh",
        namespace={"w_inh": 0.4 * mV / (500 * Mohm)},  # ~0.8 pA, 4x excitatory
    )
    inh_syn.connect(p=p_connect)

    # Build network
    net = b2.Network(ng, exc_syn, inh_syn)

    # -------------------------------------------------------------------------
    # 3D spatial layout
    # -------------------------------------------------------------------------
    cleo.coords.assign_coords_rand_rect_prism(
        ng,
        xlim=(-half, half),
        ylim=(-half, half),
        zlim=(-half, half),
        unit=um,
    )

    # -------------------------------------------------------------------------
    # Simulator + devices
    # -------------------------------------------------------------------------
    sim = cleo.CLSimulator(net)

    # Light: 2 fibers in opposing quadrants
    quarter = half / 2
    light = cleo.light.Light(
        name="fibers",
        light_model=cleo.light.fiber473nm(),
        coords=np.array([[-quarter, -quarter, -half], [quarter, quarter, -half]]) * um,
        direction=np.array([[0, 0, 1], [0, 0, 1]]),
        wavelength=473 * nmeter,
    )
    sim.inject(light, ng)

    # Opsin: ProportionalCurrentOpsin (control-affine)
    opsin = cleo.opto.ProportionalCurrentOpsin(
        name="opto",
        I_per_Irr=I_per_Irr_val * nA / (mwatt / mm**2) * 1e9,
        # normalize: I_per_Irr_val is in amps, need units of amp / (mW/mm^2)
    )
    # Actually, let's be explicit about units:
    # I_per_Irr should have units of amp / (mW/mm^2)
    # So for 50 pA per mW/mm^2:
    opsin = cleo.opto.ProportionalCurrentOpsin(
        name="opto",
        I_per_Irr=I_per_Irr_val * b2.amp / (mwatt / mm**2),
    )
    sim.inject(opsin, ng, Iopto_var_name="I_opto")

    # Probe: 50-channel MultiUnitActivity
    mua = cleo.ephys.MultiUnitActivity(name="mua")
    probe_coords = _make_probe_coords(n_channels, volume_um)
    probe = cleo.ephys.Probe(
        name="probe",
        coords=probe_coords * um,
        signals=[mua],
        save_history=True,
    )
    sim.inject(probe, ng)

    devices = {
        "light": light,
        "opsin": opsin,
        "probe": probe,
        "mua": mua,
        "ng": ng,
        "exc_syn": exc_syn,
        "inh_syn": inh_syn,
    }

    return sim, devices


if __name__ == "__main__":
    # Smoke test
    print("Building plant...")
    sim, devices = build_plant()
    print(f"  Neurons: {len(devices['ng'])}")
    print(f"  Channels: {len(devices['probe'].coords)}")
    print(f"  Fibers: {len(devices['light'].coords)}")
    print("Running 100ms smoke test...")
    sim.run(100 * ms)
    print("Done.")
