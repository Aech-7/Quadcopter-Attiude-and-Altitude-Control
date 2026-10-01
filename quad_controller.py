"""
Controller ported from ControlAlgorithm.slx, extended with an XY position loop.

Structure (full cascaded hierarchy):

    [NEW] xy_ref - pos_xy  -> PID_x / PID_y  -> desired pitch / roll angle (rad)
                                                   |
    att_ref (optional manual override) ----------->|  (xy loop wins when xy_ref is set)
                                                   v
    att_ref_rad - theta    -> *kp_att             -> rate_setpoint [pitch_rate, roll_rate, yaw_rate]
    rate_setpoint - omega  -> PID_pitch / PID_roll / PID_yaw  -> cmd (clamped -50..+50)
    z_ref - z              -> PID_alt                         -> throttle (clamped 0..100)
    (throttle, pitch_cmd, roll_cmd, yaw_cmd) -> mixer() -> 4 motor voltages

Angle conventions (from the Simulink Stateflow model, do NOT change):
    theta = [pitch, roll, yaw]
    pitch (theta[0]) drives  Y-axis force:  f2 = sin(pitch)*cos(roll)*T
    roll  (theta[1]) drives  X-axis force:  f1 = sin(roll)*cos(pitch)*T
    => x_pid output  -> desired ROLL  (theta[1])
    => y_pid output  -> desired PITCH (theta[0])

State vector (N_STATES = 12):
    s[0:2]   altitude PID  [integrator, deriv-filter]
    s[2:4]   pitch rate PID
    s[4:6]   roll  rate PID
    s[6:8]   yaw   rate PID
    s[8:10]  x position PID   <- NEW
    s[10:12] y position PID   <- NEW

Two ways to use it:
  * Pure/continuous:  voltages, ds = ctrl.command(z, omega, s, theta, pos_xy)
        -> for ODE solvers; you integrate the 12-vector of PID states `s` yourself.
  * Stateful/discrete: voltages = ctrl.step(z, omega, dt, theta, pos_xy)
        -> for fixed-step simulators. Uses forward Euler on the PID states.
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
    # 4 inner PIDs × 2 states + 2 XY PIDs × 2 states
    N_STATES = 12

    def __init__(self, z_ref=20.0, rate_ref=(0.0, 0.0, 0.0), tuned=True,
                 att_ref_deg=None, kp_att=(4.0, 4.0, 0.7), max_rate=3.0,
                 xy_ref=None, max_tilt_deg=20.0, kv_xy=0.2):
        """
        Parameters
        ----------
        z_ref        : float   Altitude setpoint [m].
        rate_ref     : (3,)    Constant rate setpoint [rad/s] before attitude/xy override.
        tuned        : bool    True = improved gains; False = original Simulink gains.
        att_ref_deg  : (3,) | None
                       Static attitude target [pitch, roll, yaw] in degrees.
                       Ignored when xy_ref is set (xy loop takes over roll/pitch).
        kp_att       : (3,)   Outer attitude P-gain [1/s] (angle error -> rate setpoint).
        max_rate     : float   Clip on rate setpoints produced by attitude loop [rad/s].
        xy_ref       : (2,) | None
                       XY position setpoint [x_ref, y_ref] in metres.
                       When set, the XY PIDs compute desired roll/pitch angles
                       to drive the quad to that horizontal position.
        max_tilt_deg : float   Hard limit on roll/pitch angle demanded by the XY loop [deg].
        kv_xy        : float   Velocity damping gain [rad/(m/s)]. The current horizontal
                       velocity is subtracted from the tilt command directly, acting as a
                       true derivative brake on the state (not a filtered error signal).
                       Increase to reduce overshoot; decrease if response feels sluggish.
        """
        self.z_ref = z_ref
        self.rate_ref = np.asarray(rate_ref, dtype=float)
        self.tuned = tuned

        # Attitude setpoint (used only when xy_ref is None)
        self.att_ref = (None if att_ref_deg is None
                        else np.radians(np.asarray(att_ref_deg, dtype=float)))
        self.kp_att = np.asarray(kp_att, dtype=float)   # [1/s]
        self.max_rate = max_rate                          # rate-setpoint clip [rad/s]

        # XY position setpoint
        self.xy_ref = None if xy_ref is None else np.asarray(xy_ref, dtype=float)
        self.max_tilt = np.radians(max_tilt_deg)         # max tilt from XY loop [rad]
        self.kv_xy = float(kv_xy)                        # velocity damping [rad/(m/s)]


        # ---- Inner rate / altitude PIDs (unchanged from original) ----
        if tuned:
            self.alt   = PID(PIDGains(P=8.0,  I=1.0, D=10.0, N=100.0, lower=0.0,   upper=100.0))
            self.pitch = PID(PIDGains(P=10.0, I=5.0, D=0.0,  N=100.0, lower=-50.0, upper=50.0))
            self.roll  = PID(PIDGains(P=10.0, I=5.0, D=0.0,  N=100.0, lower=-50.0, upper=50.0))
            self.yaw   = PID(PIDGains(P=5.0,  I=2.0, D=0.0,  N=100.0, lower=-50.0, upper=50.0))
        else:
            self.alt   = PID(PIDGains(P=6.0,  I=1.0, D=0.01, N=100.0))
            self.pitch = PID(PIDGains(P=10.0, I=5.0, D=0.0,  N=100.0, lower=0.0, upper=100.0))
            self.roll  = PID(PIDGains(P=10.0, I=5.0, D=0.0,  N=100.0, lower=0.0, upper=100.0))
            self.yaw   = PID(PIDGains(P=5.0,  I=0.5, D=0.0,  N=100.0, lower=0.0, upper=100.0))
        self._inner_pids = (self.alt, self.pitch, self.roll, self.yaw)

        # ---- Outer XY position PIDs (pure PI — no D term) ----
        # Velocity damping is handled explicitly via kv_xy * velocity, not via D-on-error.
        # D-on-error is counter-productive: during the approach the error is large and
        # positive, so D *adds* to the tilt rather than braking it.
        tilt_lim = float(self.max_tilt)
        self.x_pid = PID(PIDGains(P=0.15, I=0.03, D=0.0,
                                  lower=-tilt_lim, upper=tilt_lim))
        self.y_pid = PID(PIDGains(P=0.15, I=0.03, D=0.0,
                                  lower=-tilt_lim, upper=tilt_lim))


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _att_ref_from_xy(self, pos_xy, vel_xy, s_xy):
        """XY PI + velocity damping -> desired [pitch, roll] angles (rad).

        Tilt command = clip(P*error + I_state - kv*velocity,  ±max_tilt)

        Using true state velocity as the damping signal (not a filtered error
        derivative) gives clean, noise-free deceleration.

        Sign convention (verified from linear_accel() in quad_model.py):
            +roll  (theta[1]) -> -x acceleration  => desired_roll  = -(x_pi - kv*vx)
            +pitch (theta[0]) -> +y acceleration  => desired_pitch =  (y_pi - kv*vy)

        Returns (att_ref_rad (3,), ds_xy (4,)).
        """
        ex = self.xy_ref[0] - pos_xy[0]
        ey = self.xy_ref[1] - pos_xy[1]
        vx, vy = float(vel_xy[0]), float(vel_xy[1])

        x_pi_out, ds_x = self.x_pid.eval(ex, s_xy[0:2])
        y_pi_out, ds_y = self.y_pid.eval(ey, s_xy[2:4])

        # Subtract velocity damping then clip to tilt limit
        raw_roll  = -(x_pi_out - self.kv_xy * vx)   # negate: +roll -> -x
        raw_pitch =  (y_pi_out - self.kv_xy * vy)   # same sign: +pitch -> +y
        desired_roll  = float(np.clip(raw_roll,  -self.max_tilt, self.max_tilt))
        desired_pitch = float(np.clip(raw_pitch, -self.max_tilt, self.max_tilt))

        att_ref = np.array([desired_pitch, desired_roll, 0.0])
        ds_xy   = np.concatenate([ds_x, ds_y])
        return att_ref, ds_xy


    def rate_setpoint(self, theta=None, att_ref_rad=None):
        """Rate setpoint = constant rate_ref + outer attitude P-loop contribution."""
        r = self.rate_ref.copy()
        ref = att_ref_rad if att_ref_rad is not None else self.att_ref
        if ref is not None and theta is not None:
            r = r + np.clip(self.kp_att * (ref - np.asarray(theta)),
                            -self.max_rate, self.max_rate)
        return r

    # ------------------------------------------------------------------
    # Pure (stateless) interface for the ODE solver
    # ------------------------------------------------------------------

    def command(self, z, omega, s, theta=None, pos_xy=None, vel_xy=None):
        """Pure function — no internal state is mutated.

        Parameters
        ----------
        z       : float   Current altitude [m].
        omega   : (3,)    Current angular rates [pitch_rate, roll_rate, yaw_rate] [rad/s].
        s       : (12,)   Full PID state vector.
                          s[0:8]  -> inner PIDs (alt, pitch, roll, yaw)
                          s[8:12] -> XY PIDs (x, y)  — only used when xy_ref is set.
        theta   : (3,) | None   Current [pitch, roll, yaw] [rad].
        pos_xy  : (2,) | None   Current [x, y] position [m].
        vel_xy  : (2,) | None   Current [vx, vy] velocity [m/s], used for damping.
                                If None, velocity damping is skipped (pure PI only).

        Returns
        -------
        voltages : (4,)   Motor voltages [V].
        ds       : (12,)  Derivative of full state vector.
        """
        s_inner = s[0:8]
        s_xy    = s[8:12]

        # --- Outer XY loop (produces att_ref_rad) ---
        att_ref_rad = None
        ds_xy = np.zeros(4)
        if self.xy_ref is not None and pos_xy is not None:
            _vel = np.asarray(vel_xy, dtype=float) if vel_xy is not None else np.zeros(2)
            att_ref_rad, ds_xy = self._att_ref_from_xy(np.asarray(pos_xy), _vel, s_xy)

        # --- Attitude P-loop (produces rate setpoint) ---
        w_ref = self.rate_setpoint(theta=theta, att_ref_rad=att_ref_rad)

        # --- Inner PIDs ---
        errs = (self.z_ref - z, *(w_ref - np.asarray(omega)))
        u, ds_inner = [], []
        for k, (pid, e) in enumerate(zip(self._inner_pids, errs)):
            uk, dsk = pid.eval(e, s_inner[2 * k: 2 * k + 2])
            u.append(uk)
            ds_inner.append(dsk)

        v = mix(*u)
        if self.tuned:
            v = np.clip(v, 0.0, 11.4)

        ds = np.concatenate([*ds_inner, ds_xy])
        return v, ds

    # ------------------------------------------------------------------
    # Stateful (discrete) interface for fixed-step simulators
    # ------------------------------------------------------------------

    def step(self, z, omega, dt, theta=None, pos_xy=None):
        """Stateful fixed-step version (forward Euler on PID states)."""
        s = np.concatenate([p.s for p in self._inner_pids]
                           + [self.x_pid.s, self.y_pid.s])
        v, ds = self.command(z, omega, s, theta, pos_xy)
        # Update inner PIDs
        for k, p in enumerate(self._inner_pids):
            p.s = p.s + dt * ds[2 * k: 2 * k + 2]
        # Update XY PIDs
        self.x_pid.s = self.x_pid.s + dt * ds[8:10]
        self.y_pid.s = self.y_pid.s + dt * ds[10:12]
        return v

    def reset(self):
        for p in self._inner_pids:
            p.s = np.zeros(2)
        self.x_pid.s = np.zeros(2)
        self.y_pid.s = np.zeros(2)