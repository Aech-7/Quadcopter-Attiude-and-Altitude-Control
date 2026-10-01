"""
Quadcopter plant model, ported 1:1 from the MATLAB Function (Stateflow) blocks
in ControlAlgorithm.slx.

Simulink block            -> function here
-------------------------------------------------------
Motors/Propellers         -> motor_model()
Rotational Dynamics       -> rotational_accel()
Linear Dynamics           -> linear_accel()
Disturbances due to Gusts -> gust_force()

Conventions kept from the Simulink model (do not "fix" these silently):
  * Motor order is 1..4 -> index 0..3.
  * Angles/rates are ordered [pitch, roll, yaw] (omega[0] is the "Pitch rate").
  * theta_dot = omega (small-angle kinematics, no Euler-rate transform).
  * Thrust is a 4-vector F [N], torque a 4-vector [N*m].
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class QuadParams:
    m: float = 0.743                 # mass [kg]
    g: float = 9.81                  # gravity [m/s^2]
    rho: float = 1.225               # air density [kg/m^3]
    Cd: float = 1.0                  # drag coefficient
    arm: float = 0.237               # moment arm constant used in the model [m]
    I: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.003, 0.007]))
    A_body: np.ndarray = field(default_factory=lambda: np.array([0.0197, 0.0197, 0.0512]))
    A_gust: np.ndarray = field(default_factory=lambda: np.array([0.0197, 0.0197, 0.0197 * 2 * 1.3]))
    prop_diameter: float = 0.2       # used in F = Ct*rho*n^2*D^4


# ----------------------------------------------------------------------------
# Motors / propellers
# ----------------------------------------------------------------------------
def motor_model(V, p: QuadParams = QuadParams()):
    """Voltage [V] (4,) -> (Torque (4,), F (4,), Current (4,))."""
    V = np.asarray(V, dtype=float)
    rpm = -2.6931 * V**3 + 1400.0 * V
    Ct = 2e-15 * rpm**3 - 4e-11 * rpm**2 + 3e-7 * rpm + 0.1013
    F = Ct * p.rho * (rpm / 60.0) ** 2 * p.prop_diameter**4
    torque = 4e-14 * rpm**3 + 8e-12 * rpm**2 + 3e-6 * rpm
    current = 1400.0 * torque
    return torque, F, current


# ----------------------------------------------------------------------------
# Rotational dynamics
# ----------------------------------------------------------------------------
def rotational_accel(torque, F, p: QuadParams = QuadParams()):
    """Angular acceleration [pitch, roll, yaw] in rad/s^2."""
    h = p.arm / 2.0
    moment = np.array([
        (F[2] + F[3]) * h - (F[0] + F[1]) * h,       # pitch
        (F[2] + F[1]) * h - (F[0] + F[3]) * h,       # roll
        torque[3] - torque[0] + torque[1] - torque[2],  # yaw
    ])
    return moment / p.I


# ----------------------------------------------------------------------------
# Gusts
# ----------------------------------------------------------------------------
def gust_force(v_gust, p: QuadParams = QuadParams()):
    """Gust velocity [m/s] (3,) -> drag-disturbance force (3,).
    Same as the Simulink block: +0.5*rho*v^2*A*Cd for v>=0, negative for v<0."""
    v = np.asarray(v_gust, dtype=float)
    return np.sign(v) * 0.5 * p.rho * v**2 * p.A_gust * p.Cd


# ----------------------------------------------------------------------------
# Linear dynamics
# ----------------------------------------------------------------------------
def linear_accel(F, disturbance, theta, vel, p: QuadParams = QuadParams()):
    """Linear acceleration (3,) in m/s^2.

    NOTE: the thrust-vector construction below is copied exactly from the
    Stateflow code (it is not a standard ZYX rotation matrix).
    """
    th1, th2, th3 = theta
    total = float(np.sum(F))

    f1 = np.sin(th2) * np.cos(th1) * total
    f2 = np.sin(th1) * np.cos(th2) * total
    f3 = total * np.cos(th1) * np.cos(th2)

    theta_xy = -np.arctan2(f1, f2)
    xy2d = np.hypot(f1, f2)
    sgn = 1.0 if f3 >= 0 else -1.0
    fp = np.array([
        xy2d * np.sin(sgn * th3 + theta_xy),
        xy2d * np.cos(sgn * th3 + theta_xy),
        f3,
    ])

    vel = np.asarray(vel, dtype=float)
    # Body drag always opposes velocity: v<0 -> +, v>=0 -> -
    drag = -np.sign(vel) * 0.5 * p.rho * vel**2 * p.A_body * p.Cd

    force = fp - np.asarray(disturbance) + drag
    force[2] -= p.m * p.g
    return force / p.m