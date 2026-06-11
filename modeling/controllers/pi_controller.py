"""PI (Proportional-Integral) controller for Cleo optoclamp experiments.

Simple feedback controller that adjusts stimulation based on the error
between measured and target firing rates. No model needed.

Usage:
    from modeling.controllers.pi_controller import PIController
    ctrl = PIController(
        target_rate=target,   # (n_channels,) or scalar
        Kp=0.5, Ki=0.01,
        sample_period_ms=1.0,
    )
    sim.set_io_processor(ctrl)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import brian2.only as b2
from brian2 import ms, mwatt, mm, Hz

from cleo.ioproc import LatencyIOProcessor, exp_firing_rate_estimate


class PIController(LatencyIOProcessor):
    """Proportional-Integral controller for rate clamping.

    Computes u = Kp * error + Ki * ∫error dt, where
    error = target_rate - measured_rate (population mean).

    Parameters
    ----------
    target_rate : float
        Target firing rate in Hz (population mean)
    Kp : float
        Proportional gain (mW/mm² per Hz of error)
    Ki : float
        Integral gain (mW/mm² per Hz·ms of error)
    n_u : int
        Number of stimulation channels
    sample_period_ms : float
        How often process() is called
    compute_delay_ms : float
        Simulated processing latency
    tau_rate_ms : float
        Exponential rate estimation time constant
    u_max : float
        Maximum irradiance in mW/mm²
    warmup_steps : int
        Steps before control starts
    light_name, probe_name, mua_name : str
        Cleo device names
    """

    def __init__(
        self,
        target_rate: float,
        Kp: float = 0.5,
        Ki: float = 0.01,
        n_u: int = 2,
        sample_period_ms: float = 1.0,
        compute_delay_ms: float = 0.0,
        tau_rate_ms: float = 20.0,
        u_max: float = 50.0,
        warmup_steps: int = 200,
        light_name: str = "fibers",
        probe_name: str = "probe",
        mua_name: str = "mua",
    ):
        super().__init__(sample_period=sample_period_ms * ms)
        self.target_rate = target_rate
        self.Kp = Kp
        self.Ki = Ki
        self.n_u = n_u
        self.compute_delay = compute_delay_ms * ms
        self.tau_rate = tau_rate_ms * ms
        self.u_max = u_max
        self.warmup_steps = warmup_steps
        self.light_name = light_name
        self.probe_name = probe_name
        self.mua_name = mua_name

        # State
        self._rates = None
        self._integ_error = 0.0
        self._step = 0

        # Logging
        self.rate_log: list[np.ndarray] = []
        self.u_log: list[np.ndarray] = []
        self.error_log: list[float] = []

    def process(self, state_dict: dict, t_samp) -> Tuple[dict, float]:
        # 1. Extract spike counts
        i_chan, t_spikes, counts = state_dict[self.probe_name][self.mua_name]
        counts = np.asarray(counts, dtype=float)

        # 2. Initialize rates
        if self._rates is None:
            self._rates = np.zeros(len(counts)) * Hz

        # 3. Estimate firing rates
        self._rates = exp_firing_rate_estimate(
            counts, self.sample_period, self._rates, self.tau_rate
        )
        rate_hz = np.array(self._rates, dtype=np.float64)
        self.rate_log.append(rate_hz.copy())
        self._step += 1

        # 4. Warmup: no control
        if self._step < self.warmup_steps:
            u_out = np.zeros(self.n_u)
            self.u_log.append(u_out)
            return {self.light_name: u_out * mwatt / mm**2}, t_samp

        # 5. Compute population mean rate
        mean_rate = np.mean(rate_hz)

        # 6. PI control
        error = self.target_rate - mean_rate
        self._integ_error += error * (self.sample_period / ms)  # integrate in ms
        self.error_log.append(error)

        u_scalar = self.Kp * error + self.Ki * self._integ_error
        u_scalar = np.clip(u_scalar, 0, self.u_max)

        # Apply same signal to all fibers
        u_out = np.full(self.n_u, u_scalar)
        self.u_log.append(u_out.copy())

        return (
            {self.light_name: u_out * mwatt / mm**2},
            t_samp + self.compute_delay,
        )

    def get_results(self) -> dict:
        return {
            "rates": np.array(self.rate_log) if self.rate_log else np.array([]),
            "u": np.array(self.u_log) if self.u_log else np.array([]),
            "errors": np.array(self.error_log) if self.error_log else np.array([]),
        }

    def _base_reset(self):
        super()._base_reset()
        self._rates = None
        self._integ_error = 0.0
        self._step = 0
        self.rate_log.clear()
        self.u_log.clear()
        self.error_log.clear()

class BidirectionalPIController(LatencyIOProcessor):
    """PI controller with sign-splitting for bidirectional (exc + inh) control.

    The v1 plant sent the same scalar to both exc and inh fibers, which is
    self-defeating. Sign-splitting routes positive error (rate too low) to
    excitation and negative error (rate too high) to inhibition.

    The PI loop computes a scalar u that can go negative:
        error = target_rate - measured_rate
        u = Kp * error + Ki * ∫error dt

    Then the control is split by sign:
        u_exc = clip(max(0,  u), 0, u_max)   →  fiber_exc (ChR2-H134R, blue)
        u_inh = clip(max(0, -u), 0, u_max)   →  fiber_inh (eNpHR3.0, yellow)

    Parameters
    ----------
    target_rate : float
        Target firing rate in Hz (population mean)
    Kp, Ki : float
        Proportional / integral gains
    sample_period_ms : float
        Controller sample period
    compute_delay_ms : float
        Simulated processing latency
    tau_rate_ms : float
        Exponential rate estimation time constant
    u_max : float
        Maximum irradiance per channel in mW/mm²
    warmup_steps : int
        Steps before control starts
    light_name_exc, light_name_inh : str
        Cleo Light device names for the excitatory / inhibitory fibers
    probe_name, mua_name : str
        Cleo Probe and MUA signal names
    """

    def __init__(
        self,
        target_rate: float,
        Kp: float = 0.5,
        Ki: float = 0.01,
        sample_period_ms: float = 1.0,
        compute_delay_ms: float = 0.0,
        tau_rate_ms: float = 20.0,
        u_max: float = 50.0,
        warmup_steps: int = 200,
        light_name_exc: str = "fiber_exc",
        light_name_inh: str = "fiber_inh",
        probe_name: str = "probe",
        mua_name: str = "mua",
    ):
        super().__init__(sample_period=sample_period_ms * ms)
        self.target_rate = target_rate
        self.Kp = Kp
        self.Ki = Ki
        self.compute_delay = compute_delay_ms * ms
        self.tau_rate = tau_rate_ms * ms
        self.u_max = u_max
        self.warmup_steps = warmup_steps
        self.light_name_exc = light_name_exc
        self.light_name_inh = light_name_inh
        self.probe_name = probe_name
        self.mua_name = mua_name

        # State
        self._rates = None
        self._integ_error = 0.0
        self._step = 0

        # Logging
        self.rate_log: list[np.ndarray] = []
        self.u_exc_log: list[float] = []
        self.u_inh_log: list[float] = []
        self.error_log: list[float] = []

    def process(self, state_dict: dict, t_samp):
        # 1. Extract spike counts
        i_chan, t_spikes, counts = state_dict[self.probe_name][self.mua_name]
        counts = np.asarray(counts, dtype=float)

        # 2. Initialize rates
        if self._rates is None:
            self._rates = np.zeros(len(counts)) * Hz

        # 3. Estimate firing rates
        self._rates = exp_firing_rate_estimate(
            counts, self.sample_period, self._rates, self.tau_rate
        )
        rate_hz = np.array(self._rates, dtype=np.float64)
        self.rate_log.append(rate_hz.copy())
        self._step += 1

        # 4. Warmup: no control
        if self._step < self.warmup_steps:
            self.u_exc_log.append(0.0)
            self.u_inh_log.append(0.0)
            return {
                self.light_name_exc: np.zeros(1) * mwatt / mm**2,
                self.light_name_inh: np.zeros(1) * mwatt / mm**2,
            }, t_samp

        # 5. Compute population mean rate
        mean_rate = np.mean(rate_hz)

        # 6. PI control (scalar, can go negative)
        error = self.target_rate - mean_rate
        self._integ_error += error * (self.sample_period / ms)
        self.error_log.append(error)

        u_scalar = self.Kp * error + self.Ki * self._integ_error

        # ── Sign-splitting ──────────────────────────────────────────────
        # The v1 plant sent the same scalar to both exc and inh fibers,
        # which is self-defeating. Sign-splitting routes positive error
        # (rate too low) to excitation and negative error (rate too high)
        # to inhibition.
        u_exc = np.clip(max(0.0, u_scalar), 0, self.u_max)
        u_inh = np.clip(max(0.0, -u_scalar), 0, self.u_max)

        self.u_exc_log.append(float(u_exc))
        self.u_inh_log.append(float(u_inh))

        return (
            {
                self.light_name_exc: np.array([u_exc]) * mwatt / mm**2,
                self.light_name_inh: np.array([u_inh]) * mwatt / mm**2,
            },
            t_samp + self.compute_delay,
        )

    def get_results(self) -> dict:
        return {
            "rates": np.array(self.rate_log) if self.rate_log else np.array([]),
            "u_exc": np.array(self.u_exc_log) if self.u_exc_log else np.array([]),
            "u_inh": np.array(self.u_inh_log) if self.u_inh_log else np.array([]),
            "errors": np.array(self.error_log) if self.error_log else np.array([]),
        }

    def _base_reset(self):
        super()._base_reset()
        self._rates = None
        self._integ_error = 0.0
        self._step = 0
        self.rate_log.clear()
        self.u_exc_log.clear()
        self.u_inh_log.clear()
        self.error_log.clear()
