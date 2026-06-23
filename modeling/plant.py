"""Cleo E/I plant for digital twin modeling — bidirectional variant.

Builds a 3D LIF excitatory/inhibitory network with:
- ChrimsonR (excitatory opsin, peak ~590nm) driven by red fiber
- GtACR2 (inhibitory opsin, peak ~470nm) driven by blue fiber
- 50-channel MultiUnitActivity probe

The input u is 2-dimensional: u[0] = red (excitatory), u[1] = blue (inhibitory).

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
    um,
    nmeter,
)

import cleo
import cleo.coords
import cleo.ephys
import cleo.light
from cleo.opto.opsin_library import chrimson_4s, gtacr2_4s


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
    p_connect: float = 0.1,
    seed: int = 42,
) -> tuple[cleo.CLSimulator, dict]:
    """Build the Cleo E/I plant with bidirectional optogenetic control.

    Uses ChrimsonR (red-shifted excitatory cation channel) and
    GtACR2 (blue-shifted inhibitory anion channel) for independent
    excitation and inhibition.

    Brian2 does not allow two (summed) synapses to target the same
    variable, so we use separate variables I_exc_opto and I_inh_opto
    and combine them into I_opto in the neuron equations.

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
    p_connect : float
        Connection probability for recurrent synapses
    seed : int
        Random seed

    Returns
    -------
    sim : cleo.CLSimulator
        The configured simulator (no IOProcessor set yet)
    devices : dict
        References to key devices
    """
    np.random.seed(seed)

    n_total = n_exc + n_inh
    half = volume_um / 2

    # -------------------------------------------------------------------------
    # Network: LIF E/I population
    # -------------------------------------------------------------------------
    b2.start_scope()

    # Two separate opsin current variables to avoid the Brian2
    # "multiple summed variables" limitation.
    ng = b2.NeuronGroup(
        n_total,
        """
        dv/dt = (-(v - E_L) + Rm * (I_syn + I_exc_opto + I_inh_opto + I_bg)) / tau_m : volt
        I_syn : amp
        I_exc_opto : amp
        I_inh_opto : amp
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

    quarter = half / 2

    # Light 1: Red fiber for ChrimsonR (~590nm excitation)
    light_red = cleo.light.Light(
        name="fiber_red",
        light_model=cleo.light.fiber473nm(),  # same spatial propagation model
        coords=np.array([[-quarter, -quarter, -half]]) * um,
        direction=np.array([[0, 0, 1]]),
        wavelength=590 * nmeter,
    )
    sim.inject(light_red, ng)

    # Light 2: Blue fiber for GtACR2 (~470nm inhibition)
    light_blue = cleo.light.Light(
        name="fiber_blue",
        light_model=cleo.light.fiber473nm(),
        coords=np.array([[quarter, quarter, -half]]) * um,
        direction=np.array([[0, 0, 1]]),
        wavelength=470 * nmeter,
    )
    sim.inject(light_blue, ng)

    # Opsin 1: ChrimsonR — excitatory cation channel (E=0mV)
    # All neurons express it; driven primarily by the red fiber.
    # Maps to I_exc_opto in the neuron model.
    opsin_exc = chrimson_4s()
    sim.inject(opsin_exc, ng, Iopto_var_name="I_exc_opto")

    # Opsin 2: GtACR2 — inhibitory anion channel (E=-69.5mV)
    # All neurons express it; driven primarily by the blue fiber.
    # Maps to I_inh_opto in the neuron model.
    opsin_inh = gtacr2_4s()
    sim.inject(opsin_inh, ng, Iopto_var_name="I_inh_opto")

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
        "light_red": light_red,
        "light_blue": light_blue,
        "opsin_exc": opsin_exc,
        "opsin_inh": opsin_inh,
        "probe": probe,
        "mua": mua,
        "ng": ng,
        "exc_syn": exc_syn,
        "inh_syn": inh_syn,
    }

    return sim, devices


def build_plant_v2(
    n_exc: int = 800,
    n_inh: int = 200,
    n_channels: int = 50,
    volume_um: float = 500.0,
    p_connect: float = 0.1,
    seed: int = 42,
) -> tuple[cleo.CLSimulator, dict]:
    """Build the Cleo E/I plant with ChR2(H134R) + eNpHR3.0 (bidirectional v2).

    The current primary bidirectional plant. Uses ChR2(H134R) (excitatory
    cation channel, peak ~450nm) driven by a blue fiber and eNpHR3.0
    (inhibitory chloride *pump*, peak ~590nm, E=-400mV) driven by a yellow
    fiber. Unlike GtACR2 (E=-69.5mV, ~0.5mV drive), eNpHR3.0 provides strong
    hyperpolarizing drive (~330mV), enabling genuine inhibition below baseline.

    Produced ``data/training_trials_bidir_v2.h5``. Ported verbatim from the
    canonical ``feature/bidir-v2-plant`` branch.

    Parameters mirror :func:`build_plant`.

    Returns
    -------
    sim : cleo.CLSimulator
        The configured simulator (no IOProcessor set yet).
    devices : dict
        References: ``light_exc`` (blue), ``light_inh`` (yellow), ``opsin_exc``,
        ``opsin_inh``, ``probe``, ``mua``, ``ng``, ``exc_syn``, ``inh_syn``.
    """
    from cleo.opto.opsin_library import chr2_h134r_4s, enphr3_3s

    np.random.seed(seed)

    n_total = n_exc + n_inh
    half = volume_um / 2

    # -------------------------------------------------------------------------
    # Network: LIF E/I population (identical to v1)
    # -------------------------------------------------------------------------
    b2.start_scope()

    ng = b2.NeuronGroup(
        n_total,
        """
        dv/dt = (-(v - E_L) + Rm * (I_syn + I_exc_opto + I_inh_opto + I_bg)) / tau_m : volt
        I_syn : amp
        I_exc_opto : amp
        I_inh_opto : amp
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
    ng.v = "E_L + rand() * (v_th - E_L)"

    exc_syn = b2.Synapses(
        ng[:n_exc],
        ng,
        on_pre="I_syn_post += w_exc",
        namespace={"w_exc": 0.1 * mV / (500 * Mohm)},
    )
    exc_syn.connect(p=p_connect)

    inh_syn = b2.Synapses(
        ng[n_exc:],
        ng,
        on_pre="I_syn_post -= w_inh",
        namespace={"w_inh": 0.4 * mV / (500 * Mohm)},
    )
    inh_syn.connect(p=p_connect)

    net = b2.Network(ng, exc_syn, inh_syn)

    cleo.coords.assign_coords_rand_rect_prism(
        ng,
        xlim=(-half, half),
        ylim=(-half, half),
        zlim=(-half, half),
        unit=um,
    )

    sim = cleo.CLSimulator(net)

    quarter = half / 2

    # Light 1: Blue fiber for ChR2(H134R) (~450nm excitation)
    light_exc = cleo.light.Light(
        name="fiber_exc",
        light_model=cleo.light.fiber473nm(),
        coords=np.array([[-quarter, -quarter, -half]]) * um,
        direction=np.array([[0, 0, 1]]),
        wavelength=450 * nmeter,
    )
    sim.inject(light_exc, ng)

    # Light 2: Yellow fiber for eNpHR3.0 (~590nm inhibition)
    light_inh = cleo.light.Light(
        name="fiber_inh",
        light_model=cleo.light.fiber473nm(),
        coords=np.array([[quarter, quarter, -half]]) * um,
        direction=np.array([[0, 0, 1]]),
        wavelength=590 * nmeter,
    )
    sim.inject(light_inh, ng)

    # Opsin 1: ChR2(H134R) — excitatory cation channel (E=0mV)
    opsin_exc = chr2_h134r_4s()
    sim.inject(opsin_exc, ng, Iopto_var_name="I_exc_opto")

    # Opsin 2: eNpHR3.0 — inhibitory chloride pump (E=-400mV)
    opsin_inh = enphr3_3s()
    sim.inject(opsin_inh, ng, Iopto_var_name="I_inh_opto")

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
        "light_exc": light_exc,
        "light_inh": light_inh,
        "opsin_exc": opsin_exc,
        "opsin_inh": opsin_inh,
        "probe": probe,
        "mua": mua,
        "ng": ng,
        "exc_syn": exc_syn,
        "inh_syn": inh_syn,
    }

    return sim, devices


if __name__ == "__main__":
    # Smoke test
    print("Building bidirectional plant...")
    sim, devices = build_plant()
    print(f"  Neurons: {len(devices['ng'])}")
    print(f"  Channels: {len(devices['probe'].coords)}")
    print(f"  Red fiber sources: {len(devices['light_red'].coords)}")
    print(f"  Blue fiber sources: {len(devices['light_blue'].coords)}")
    print(f"  Opsins: {devices['opsin_exc'].name} (exc) + {devices['opsin_inh'].name} (inh)")
    print("Running 100ms smoke test...")
    sim.run(100 * ms)
    print("Done.")


def build_plant_v2(
    n_exc: int = 800,
    n_inh: int = 200,
    n_channels: int = 50,
    volume_um: float = 500.0,
    p_connect: float = 0.1,
    seed: int = 42,
) -> tuple[cleo.CLSimulator, dict]:
    """Build the Cleo E/I plant with ChR2(H134R) + eNpHR3.0.

    Uses ChR2(H134R) (excitatory cation channel, peak ~450nm) driven by
    a blue fiber and eNpHR3.0 (inhibitory chloride pump, peak ~590nm,
    E=-400mV) driven by a yellow fiber. Unlike GtACR2 (E=-69.5mV),
    eNpHR3.0 is a pump with strong hyperpolarizing drive, enabling
    genuine inhibition below baseline.

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
    p_connect : float
        Connection probability for recurrent synapses
    seed : int
        Random seed

    Returns
    -------
    sim : cleo.CLSimulator
        The configured simulator (no IOProcessor set yet)
    devices : dict
        References to key devices
    """
    from cleo.opto.opsin_library import chr2_h134r_4s, enphr3_3s

    np.random.seed(seed)

    n_total = n_exc + n_inh
    half = volume_um / 2

    # -------------------------------------------------------------------------
    # Network: LIF E/I population (identical to v1)
    # -------------------------------------------------------------------------
    b2.start_scope()

    ng = b2.NeuronGroup(
        n_total,
        """
        dv/dt = (-(v - E_L) + Rm * (I_syn + I_exc_opto + I_inh_opto + I_bg)) / tau_m : volt
        I_syn : amp
        I_exc_opto : amp
        I_inh_opto : amp
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
    ng.v = "E_L + rand() * (v_th - E_L)"

    exc_syn = b2.Synapses(
        ng[:n_exc],
        ng,
        on_pre="I_syn_post += w_exc",
        namespace={"w_exc": 0.1 * mV / (500 * Mohm)},
    )
    exc_syn.connect(p=p_connect)

    inh_syn = b2.Synapses(
        ng[n_exc:],
        ng,
        on_pre="I_syn_post -= w_inh",
        namespace={"w_inh": 0.4 * mV / (500 * Mohm)},
    )
    inh_syn.connect(p=p_connect)

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

    quarter = half / 2

    # Light 1: Blue fiber for ChR2(H134R) (~450nm excitation)
    light_exc = cleo.light.Light(
        name="fiber_exc",
        light_model=cleo.light.fiber473nm(),
        coords=np.array([[-quarter, -quarter, -half]]) * um,
        direction=np.array([[0, 0, 1]]),
        wavelength=450 * nmeter,
    )
    sim.inject(light_exc, ng)

    # Light 2: Yellow fiber for eNpHR3.0 (~590nm inhibition)
    light_inh = cleo.light.Light(
        name="fiber_inh",
        light_model=cleo.light.fiber473nm(),
        coords=np.array([[quarter, quarter, -half]]) * um,
        direction=np.array([[0, 0, 1]]),
        wavelength=590 * nmeter,
    )
    sim.inject(light_inh, ng)

    # Opsin 1: ChR2(H134R) — excitatory cation channel (E=0mV)
    opsin_exc = chr2_h134r_4s()
    sim.inject(opsin_exc, ng, Iopto_var_name="I_exc_opto")

    # Opsin 2: eNpHR3.0 — inhibitory chloride pump (E=-400mV)
    opsin_inh = enphr3_3s()
    sim.inject(opsin_inh, ng, Iopto_var_name="I_inh_opto")

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
        "light_exc": light_exc,
        "light_inh": light_inh,
        "opsin_exc": opsin_exc,
        "opsin_inh": opsin_inh,
        "probe": probe,
        "mua": mua,
        "ng": ng,
        "exc_syn": exc_syn,
        "inh_syn": inh_syn,
    }

    return sim, devices
