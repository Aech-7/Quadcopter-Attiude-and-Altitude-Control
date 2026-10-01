"""
Reference closed-loop simulation = Python equivalent of ControlAlgorithm.slx,
extended with the XY position controller.

State vector x (24):
    0:3   position            [x, y, z]
    3:6   linear velocity     [vx, vy, vz]
    6:9   theta               [pitch, roll, yaw]
    9:12  omega               [pitch_rate, roll_rate, yaw_rate]
    12:24 PID states (12)     [alt(2), pitch(2), roll(2), yaw(2), x_pos(2), y_pos(2)]

Simulink settings reproduced: stop time 100 s, stiff variable-step solver (ode15s -> BDF),
rtol 1e-3 in Simulink; tighter here (explicit RK45 here, see run()) so this is a good "truth".
"""
import numpy as np
from scipy.integrate import solve_ivp

from quad_model import QuadParams, motor_model, rotational_accel, linear_accel, gust_force
from quad_controller import QuadController

P = QuadParams()
CTRL = QuadController(z_ref=20.0, rate_ref=(0.0, 0.0, 0.0))

# Total ODE state length: 12 plant states + 12 PID states
_N_PLANT = 12
_N_STATE = _N_PLANT + QuadController.N_STATES   # 12 + 12 = 24


def wind_gusts(t):
    """'Wind Gusts' constant block = [0,0,0] in the model. Make it a function of t if you like."""
    return np.array([0.0, 0.0, 0.0])


def rhs(t, x):
    pos, vel, th, om = x[0:3], x[3:6], x[6:9], x[9:12]
    s = x[12:_N_STATE]
    V, ds = CTRL.command(z=pos[2], omega=om, s=s, theta=th, pos_xy=pos[0:2])
    torque, F, _ = motor_model(V, P)
    ang_acc = rotational_accel(torque, F, P)
    lin_acc = linear_accel(F, gust_force(wind_gusts(t), P), th, vel, P)
    return np.concatenate([vel, lin_acc, om, ang_acc, ds])


def run(t_end=100.0, dt_out=0.01, method="RK45"):
    """RK45 (explicit) is used on purpose: the controller has hard saturations (throttle clip,
    voltage clip, PID clamping, mixer 'if's). Those kinks make the *numerical Jacobian* of the
    stiff solvers (BDF/Radau) overflow when the motors are saturated at t=0 (e.g. z_ref=20).
    max_step=0.01 keeps the fast PID filter (N=100 rad/s) stable."""
    t_eval = np.arange(0.0, t_end + dt_out / 2, dt_out)
    sol = solve_ivp(rhs, (0.0, t_end), np.zeros(_N_STATE), method=method,
                    t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.01)
    return sol


if __name__ == "__main__":
    import sys
    # python run.py [z_ref] [--xy=x,y] [--att=pitch,roll,yaw] [--legacy]
    #   --xy  in metres, e.g. --xy=5,3
    #   --att in degrees, e.g. --att=0,10,0  (ignored when --xy is set)
    #   --legacy = original, unstable Simulink behaviour
    args  = [a for a in sys.argv[1:] if not a.startswith("--")]
    att   = next((tuple(float(v) for v in a[6:].split(","))  for a in sys.argv if a.startswith("--att=")),  None)
    xy    = next((tuple(float(v) for v in a[5:].split(","))  for a in sys.argv if a.startswith("--xy=")),   None)

    CTRL = QuadController(z_ref=float(args[0]) if args else 20.0,
                          tuned="--legacy" not in sys.argv,
                          att_ref_deg=att,
                          xy_ref=xy)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sol = run()
    t, x = sol.t, sol.y.T
    pos, th = x[:, 0:3], np.degrees(x[:, 6:9])

    # recompute voltages for plotting
    volt = np.array([CTRL.command(xi[2], xi[9:12], xi[12:_N_STATE], xi[6:9], xi[0:2])[0]
                     for xi in x])

    np.savetxt("position_vector.csv", np.column_stack([t, pos]),
               delimiter=",", header="t,x,y,z", comments="")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(t, pos[:, 2]); ax[0, 0].axhline(20, ls="--", c="gray")
    ax[0, 0].set(title="Altitude z [m]", xlabel="t [s]")
    ax[0, 1].plot(t, pos[:, 0], label="x"); ax[0, 1].plot(t, pos[:, 1], label="y")
    if xy is not None:
        ax[0, 1].axhline(xy[0], ls="--", c="C0", alpha=0.5, label="x_ref")
        ax[0, 1].axhline(xy[1], ls="--", c="C1", alpha=0.5, label="y_ref")
    ax[0, 1].set(title="Horizontal position [m]", xlabel="t [s]"); ax[0, 1].legend()
    ax[1, 0].plot(t, th); ax[1, 0].set(title="Angles [deg] (pitch, roll, yaw)", xlabel="t [s]")
    ax[1, 1].plot(t, volt); ax[1, 1].set(title="Motor voltages [V]", xlabel="t [s]")
    fig.tight_layout(); fig.savefig("reference_run.png", dpi=130)

    z = pos[:, 2]
    print(f"solver ok={sol.success}  nfev={sol.nfev}")
    print(f"z final={z[-1]:.3f} m  max={z.max():.3f} m  min={z.min():.3f} m")
    print(f"max |angle| = {np.abs(th).max():.4f} deg   max |xy| = {np.abs(pos[:, :2]).max():.4f} m")
    print(f"voltage range = [{volt.min():.2f}, {volt.max():.2f}] V")