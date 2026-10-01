"""
Simple 3D visualizer for the quadcopter simulation.

Usage (from the folder with the other files):
    python visualize.py                       # hover at z=10 m
    python visualize.py --zref 5              # different altitude setpoint
    python visualize.py --att 0 15 0          # hold 15 deg roll (drifts sideways)
    python visualize.py --xy 5 3              # fly to x=5, y=3 while holding z=10
    python visualize.py --xy 5 3 --att 0 0 30 # fly to (5,3) and rotate yaw 30 deg
    python visualize.py --tend 30 --speed 2
    python visualize.py --save run.gif        # write a GIF instead of opening a window

The camera always follows the quad (fixed-size window centred on current position),
so the quad is always visible regardless of how far it drifts.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    import run as rr
except ImportError:
    import run as rr
from quad_controller import QuadController


def rot(theta):
    """Body->world rotation for the model's angle ordering [pitch, roll, yaw].
    Chosen so the drawn body z-axis tilts the way the model's thrust vector pushes the quad:
    positive pitch -> +y, positive roll -> -x, positive yaw = clockwise seen from above
    (that is what the Stateflow thrust-vector code does)."""
    t1, t2, t3 = theta
    a, b, c = -t1, -t2, -t3
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zref",  type=float, default=10.0, help="altitude setpoint [m]")
    ap.add_argument("--xy",    type=float, nargs=2, default=(10, 10),
                    metavar=("X", "Y"),
                    help="XY position setpoint [m], e.g. --xy 5 3. "
                         "Enables the XY position controller.")
    ap.add_argument("--tend",  type=float, default=30.0, help="simulation length [s]")
    ap.add_argument("--speed", type=float, default=1.0,  help="playback speed (1 = real time)")
    ap.add_argument("--fps",   type=int,   default=30)
    ap.add_argument("--att",   type=float, nargs=3, default=(0,0,0),
                    metavar=("PITCH", "ROLL", "YAW"),
                    help="attitude target [deg]. When --xy is also given, only yaw is used "
                         "(XY controller commands pitch/roll). Default: (0,0,0).")
    ap.add_argument("--legacy", action="store_true", help="original (unstable) Simulink gains")
    ap.add_argument("--save",  type=str, default=None, help="output .gif instead of a window")
    a = ap.parse_args()

    # When --xy is active: XY PID drives pitch/roll, att_ref only contributes yaw.
    # When --xy is absent: att_ref controls all three axes.
    rr.CTRL = QuadController(
        z_ref=a.zref,
        tuned=not a.legacy,
        att_ref_deg=a.att,
        xy_ref=a.xy,
    )
    sol = rr.run(t_end=a.tend, dt_out=0.01)
    t, x = sol.t, sol.y.T
    pos, th = x[:, 0:3], x[:, 6:9]

    # pick frames at the requested playback rate
    step = max(1, int(round(a.speed / a.fps / 0.01)))
    idx = np.arange(0, len(t), step)

    # --- Follow-cam window ---
    # Fixed-size view centred on the quad's current position each frame.
    # Span is set just large enough to comfortably show the quad and target.
    if a.xy is not None:
        dist = np.hypot(a.xy[0], a.xy[1])          # distance to XY target
        span = max(dist * 1.4, a.zref * 1.5, 6.0)  # wide enough to include target
    else:
        span = max(a.zref * 1.5, 6.0)              # altitude-relative window

    L = span * 0.07          # drawn arm length (scales with view)
    motors_body = L / np.sqrt(2) * np.array([[-1,-1,0],[-1,1,0],[1,1,0],[1,-1,0]])
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_box_aspect((1, 1, 1))

    # Setpoint marker
    if a.xy is not None:
        ax.plot([a.xy[0]], [a.xy[1]], [a.zref], "k+", ms=14, zorder=5,
                label=f"target ({a.xy[0]:g}, {a.xy[1]:g}, {a.zref:g}) m")
        ax.plot([a.xy[0], a.xy[0]], [a.xy[1], a.xy[1]], [0, a.zref],
                "--", c="gray", lw=1, alpha=0.5)
    else:
        ax.plot([0], [0], [a.zref], "k+", ms=12, label=f"z_ref = {a.zref:g} m")

    trail,  = ax.plot([], [], [], "-",  c="gray",     lw=1)
    shadow, = ax.plot([], [], [], "o",  c="lightgray", ms=6)
    arm1,   = ax.plot([], [], [], "-",  c="k",         lw=2)
    arm2,   = ax.plot([], [], [], "-",  c="k",         lw=2)
    dots = [ax.plot([], [], [], "o", c=col, ms=8)[0] for col in colors]
    nose,   = ax.plot([], [], [], "-",  c="m",         lw=3)
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, family="monospace")
    ax.legend(loc="upper right")

    def update(k):
        i   = idx[k]
        R, p = rot(th[i]), pos[i]
        w   = (R @ motors_body.T).T + p          # world positions of the 4 motors

        # --- Follow-cam: re-centre axes on current quad position every frame ---
        cx, cy, cz = p[0], p[1], max(p[2], span / 2)
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy - span / 2, cy + span / 2)
        ax.set_zlim(max(0, cz - span / 2), max(0, cz - span / 2) + span)

        # Draw quad body
        arm1.set_data_3d(*zip(w[0], w[2]))
        arm2.set_data_3d(*zip(w[1], w[3]))
        for d, q in zip(dots, w):
            d.set_data_3d([q[0]], [q[1]], [q[2]])
        front = np.array([p, p + R @ np.array([L * 0.9, 0, 0])])
        nose.set_data_3d(front[:, 0], front[:, 1], front[:, 2])
        trail.set_data_3d(pos[: i + 1, 0], pos[: i + 1, 1], pos[: i + 1, 2])
        shadow.set_data_3d([p[0]], [p[1]], [0])

        # HUD text
        d3 = np.degrees(th[i])
        xy_str = (f"\nx = {p[0]:6.2f} m  y = {p[1]:6.2f} m" if a.xy is not None else "")
        txt.set_text(f"t = {t[i]:6.2f} s\nz = {p[2]:6.2f} m{xy_str}\n"
                     f"pitch/roll/yaw = {d3[0]:.1f}/{d3[1]:.1f}/{d3[2]:.1f} deg")
        return []

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / a.fps, blit=False)
    if a.save:
        anim.save(a.save, writer=PillowWriter(fps=a.fps))
        print("saved", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()