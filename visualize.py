"""
Simple 3D visualizer for the quadcopter simulation.

Usage (from the folder with the other files):
    python visualize.py                  # setpoint 20 m, opens a window
    python visualize.py --zref 5         # different altitude setpoint
    python visualize.py --tend 30 --speed 2
    python visualize.py --save run.gif   # write a GIF instead of opening a window

Needs only numpy / scipy / matplotlib. Imports the simulation from run_reference.py
(or run.py, if you renamed it).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    import run as rr
except ImportError:          # you renamed it
    import run as rr
from quad_controller import QuadController


def rot(theta):
    """Body->world rotation for the model's angle ordering [pitch, roll, yaw].
    Chosen so the body z-axis tilts the same way the model's thrust vector does
    (roll tilts toward +x, pitch toward +y); yaw is about world z."""
    t1, t2, t3 = theta
    Rx = np.array([[1, 0, 0], [0, np.cos(-t1), -np.sin(-t1)], [0, np.sin(-t1), np.cos(-t1)]])
    Ry = np.array([[np.cos(t2), 0, np.sin(t2)], [0, 1, 0], [-np.sin(t2), 0, np.cos(t2)]])
    Rz = np.array([[np.cos(t3), -np.sin(t3), 0], [np.sin(t3), np.cos(t3), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zref", type=float, default=20.0, help="altitude setpoint [m]")
    ap.add_argument("--tend", type=float, default=100.0, help="simulation length [s]")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--att", type=float, nargs=3, default=(1, 0, 0), metavar=("PITCH", "ROLL", "YAW"),
                    help="attitude target in degrees, e.g. --att 0 10 0")
    ap.add_argument("--legacy", action="store_true", help="original (unstable) Simulink gains/limits")
    ap.add_argument("--save", type=str, default=None, help="output .gif instead of a window")
    a = ap.parse_args()

    rr.CTRL = QuadController(z_ref=a.zref, tuned=not a.legacy, att_ref_deg=a.att)
    sol = rr.run(t_end=a.tend, dt_out=0.01)
    t, x = sol.t, sol.y.T
    pos, th = x[:, 0:3], x[:, 6:9]

    # pick frames at the requested playback rate
    step = max(1, int(round(a.speed / a.fps / 0.01)))
    idx = np.arange(0, len(t), step)

    # axis limits (equal scale) and an exaggerated drawing size for the quad
    lo = np.minimum(pos.min(0), [-1, -1, 0])
    hi = np.maximum(pos.max(0), [1, 1, 1])
    c, span = (lo + hi) / 2, max((hi - lo).max(), 2.0) * 1.1
    L = span * 0.06                                   # drawn arm length
    motors_body = L / np.sqrt(2) * np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]   # motors 1..4

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(c[0] - span / 2, c[0] + span / 2)
    ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
    ax.set_zlim(max(0, c[2] - span / 2), max(0, c[2] - span / 2) + span)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_box_aspect((1, 1, 1))

    # setpoint marker + ground shadow
    ax.plot([0], [0], [a.zref], "k+", ms=12, label=f"z_ref = {a.zref:g} m")
    trail, = ax.plot([], [], [], "-", c="gray", lw=1)
    shadow, = ax.plot([], [], [], "o", c="lightgray", ms=6)
    arm1, = ax.plot([], [], [], "-", c="k", lw=2)
    arm2, = ax.plot([], [], [], "-", c="k", lw=2)
    dots = [ax.plot([], [], [], "o", c=col, ms=8)[0] for col in colors]
    nose, = ax.plot([], [], [], "-", c="m", lw=3)     # short line along body +x ("front")
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, family="monospace")
    ax.legend(loc="upper right")

    def update(k):
        i = idx[k]
        R, p = rot(th[i]), pos[i]
        w = (R @ motors_body.T).T + p                 # world positions of the 4 motors
        arm1.set_data_3d(*zip(w[0], w[2]))
        arm2.set_data_3d(*zip(w[1], w[3]))
        for d, q in zip(dots, w):
            d.set_data_3d([q[0]], [q[1]], [q[2]])
        front = np.array([p, p + R @ np.array([L * 0.9, 0, 0])])
        nose.set_data_3d(front[:, 0], front[:, 1], front[:, 2])
        trail.set_data_3d(pos[: i + 1, 0], pos[: i + 1, 1], pos[: i + 1, 2])
        shadow.set_data_3d([p[0]], [p[1]], [0])
        d3 = np.degrees(th[i])
        txt.set_text(f"t = {t[i]:6.2f} s\nz = {p[2]:6.2f} m\n"
                     f"pitch/roll/yaw = {d3[0]:.0f}/{d3[1]:.0f}/{d3[2]:.0f} deg")
        return []

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / a.fps, blit=False)
    if a.save:
        anim.save(a.save, writer=PillowWriter(fps=a.fps))
        print("saved", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()