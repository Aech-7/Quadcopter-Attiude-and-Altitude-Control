"""
Controller ported from ControlAlgorithm.slx.

Structure (same as the Simulink diagram):

    z_ref (20) - z        -> PID_alt   -> throttle
    (optional) att_ref - theta -> *kp_att -> added to the rate setpoints below
    pitch_rate_ref - w[0] -> PID_pitch -> pitch cmd (clamped 0..100)
    roll_rate_ref  - w[1] -> PID_roll  -> roll cmd  (clamped 0..100)
    yaw_rate_ref   - w[2] -> PID_yaw   -> yaw cmd   (clamped 0..100)
    (throttle, pitch, roll, yaw) -> mixer() -> 4 motor voltages

Two ways to use it:
  * Pure/continuous:  voltages, ds = ctrl.command(z, omega, s)
        -> for ODE solvers; you integrate the 8-vector of PID states `s` yourself.
  * Stateful/discrete: voltages = ctrl.step(z, omega, dt)
        -> for fixed-step simulators (PyBullet, MuJoCo, Gazebo, ...). Uses forward Euler
           on the PID states, which is fine for dt <= ~2 ms given N = 100 rad/s.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class PIDGains:
    P: float
    I: float
    D: float
    N: float = 100.0              # derivative filter coefficient
    lower: float = -np.inf        # output saturation (Simulink "LimitOutput")
    upper: float = np.inf


class PID:
    """Simulink 'PID Controller' block: Parallel form, continuous time,
    filtered derivative (D*N*s/(s+N)), integrator with clamping anti-windup.

    State s = [integrator, derivative-filter].
    The I gain is applied *before* the integrator, as in Simulink.
    """

    def __init__(self, g: PIDGains):
        self.g = g
        self.s = np.zeros(2)

    def eval(self, e, s):
        g = self.g
        xi, xf = s
        d_term = g.D * g.N * (e - xf)
        u = g.P * e + xi + d_term
        u_sat = min(max(u, g.lower), g.upper)

        dxi = g.I * e
        # Clamping: stop integrating if saturated AND integrator input pushes further into saturation
        if u != u_sat and np.sign(u) == np.sign(dxi):
            dxi = 0.0
        dxf = g.N * (e - xf)
        return u_sat, np.array([dxi, dxf])

    def step(self, e, dt):
        u, ds = self.eval(e, self.s)
        self.s = self.s + dt * ds
        return u


def mix(throttle, pitch, roll, yaw):
    """'Control' MATLAB Function block: commands (0..100 scale) -> 4 voltages [V]."""
    s = pitch / 2 + roll / 2 + yaw / 2
    correct = 0.0
    if s > (100 - throttle):
        correct = s - (100 - throttle)
    if s > throttle:
        if correct < s - throttle:
            correct = s - throttle
    if correct != 0:
        pitch -= correct / 3 * 2
        roll -= correct / 3 * 2
        yaw -= correct / 3 * 2

    k = 11.4 / 100.0
    return np.array([
        (throttle - pitch / 2 - roll / 2 - yaw / 2) * k,
        (throttle - pitch / 2 + roll / 2 + yaw / 2) * k,
        (throttle + pitch / 2 + roll / 2 - yaw / 2) * k,
        (throttle + pitch / 2 - roll / 2 + yaw / 2) * k,
    ])


class QuadController:
    N_STATES = 8   # 4 PIDs x (integrator, filter)

    def __init__(self, z_ref=20.0, rate_ref=(0.0, 0.0, 0.0), tuned=True,
                 att_ref_deg=None, kp_att=(4.0, 4.0, 0.7), max_rate=3.0):
        """tuned=False reproduces ControlAlgorithm.slx exactly (unstable attitude at z_ref=20).
        tuned=True applies the fixes described in the README/chat:
          * throttle limited to 0..100 (stops the mixer's 'Correct' kick at t=0)
          * rate PIDs limited to -50..+50 instead of 0..100 (they can brake in both directions)
          * motor voltage clipped to 0..11.4 V
          * altitude D = 10 (was 0.01) and gains 8/1/10 -> no overshoot
          * yaw I = 2 (was 0.5) -> yaw rate settles in ~6 s instead of ~27 s
        """
        self.z_ref = z_ref
        self.rate_ref = np.asarray(rate_ref, dtype=float)
        self.tuned = tuned
        # Optional outer attitude loop (angle error -> rate setpoint). Disabled when att_ref_deg is None.
        # att_ref_deg = (pitch, roll, yaw) in degrees, same ordering as the model's theta.
        self.att_ref = None if att_ref_deg is None else np.radians(np.asarray(att_ref_deg, dtype=float))
        self.kp_att = np.asarray(kp_att, dtype=float)   # [1/s]
        self.max_rate = max_rate                        # rate-setpoint limit [rad/s]
        if tuned:
            self.alt = PID(PIDGains(P=8.0, I=1.0, D=10.0, N=100.0, lower=0.0, upper=100.0))
            self.pitch = PID(PIDGains(P=10.0, I=5.0, D=0.0, N=100.0, lower=-50.0, upper=50.0))
            self.roll = PID(PIDGains(P=10.0, I=5.0, D=0.0, N=100.0, lower=-50.0, upper=50.0))
            self.yaw = PID(PIDGains(P=5.0, I=2.0, D=0.0, N=100.0, lower=-50.0, upper=50.0))
        else:
            self.alt = PID(PIDGains(P=6.0, I=1.0, D=0.01, N=100.0))
            self.pitch = PID(PIDGains(P=10.0, I=5.0, D=0.0, N=100.0, lower=0.0, upper=100.0))
            self.roll = PID(PIDGains(P=10.0, I=5.0, D=0.0, N=100.0, lower=0.0, upper=100.0))
            self.yaw = PID(PIDGains(P=5.0, I=0.5, D=0.0, N=100.0, lower=0.0, upper=100.0))
        self._pids = (self.alt, self.pitch, self.roll, self.yaw)

    def rate_setpoint(self, theta=None):
        """Rate setpoint = constant rate_ref (+ outer attitude P-loop if att_ref is set)."""
        r = self.rate_ref.copy()
        if self.att_ref is not None and theta is not None:
            r = r + np.clip(self.kp_att * (self.att_ref - np.asarray(theta)),
                            -self.max_rate, self.max_rate)
        return r

    def command(self, z, omega, s, theta=None):
        """Pure function. omega = [pitch_rate, roll_rate, yaw_rate] (rad/s),
        theta = [pitch, roll, yaw] (rad, only needed for the attitude loop).
        s: array(8). Returns (voltages(4), ds(8))."""
        errs = (self.z_ref - z, *(self.rate_setpoint(theta) - np.asarray(omega)))
        u, ds = [], []
        for k, (pid, e) in enumerate(zip(self._pids, errs)):
            uk, dsk = pid.eval(e, s[2 * k: 2 * k + 2])
            u.append(uk)
            ds.append(dsk)
        v = mix(*u)
        if self.tuned:
            v = np.clip(v, 0.0, 11.4)
        return v, np.concatenate(ds)

    def step(self, z, omega, dt, theta=None):
        """Stateful fixed-step version (forward Euler on PID states)."""
        s = np.concatenate([p.s for p in self._pids])
        v, ds = self.command(z, omega, s, theta)
        for k, p in enumerate(self._pids):
            p.s = p.s + dt * ds[2 * k: 2 * k + 2]
        return v

    def reset(self):
        for p in self._pids:
            p.s = np.zeros(2)