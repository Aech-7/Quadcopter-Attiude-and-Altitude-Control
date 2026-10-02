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
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

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
    ap.add_argument("--save",  type=str, default=None, help="output .mp4 or .gif instead of a window")
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

    goal = np.array([*(a.xy if a.xy is not None else (0.0, 0.0)), a.zref])
    initial_span = max(np.max(np.abs(pos[0] - goal)) * 1.45, 5.0)
    route_points = np.vstack((pos, goal))
    route_center = (route_points.min(axis=0) + route_points.max(axis=0)) / 2
    route_span = max(np.ptp(route_points, axis=0).max() * 1.25, 5.0)
    zoom_frames = max(1, round(a.fps * 1.25))
    overview_frames = max(1, round(a.fps * 1.0))
    closing_frames = zoom_frames + overview_frames

    L = initial_span * 0.07
    motors_body = L / np.sqrt(2) * np.array([[-1,-1,0],[-1,1,0],[1,1,0],[1,-1,0]])

    bg = "#252a2d"
    pane = "#343a3e"
    fig = plt.figure(figsize=(6.75, 12), dpi=160, facecolor=bg)
    ax  = fig.add_subplot(111, projection="3d", facecolor=bg)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane)
        axis.pane.set_edgecolor("#697278")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.tick_params(colors="#d1d6d9", labelsize=9)
    ax.xaxis.label.set_color("#d1d6d9")
    ax.yaxis.label.set_color("#d1d6d9")
    ax.zaxis.label.set_color("#d1d6d9")
    ax.set_box_aspect((1, 1, 1.35))
    ax.view_init(elev=28, azim=-55)

    ax.plot([goal[0], goal[0]], [goal[1], goal[1]], [0, goal[2]],
            "--", c="#f4bd58", lw=1.8, alpha=0.8)
    ax.scatter(*goal, marker="*", s=220, c="#ffd166", edgecolors="white",
               linewidths=0.9, depthshade=False)
    ax.text(goal[0], goal[1], goal[2] + 0.35, "TARGET", color="#ffd166",
            fontsize=10, fontweight="bold", ha="center")

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "--", c="#778187", lw=1.5, alpha=0.7)
    trail,  = ax.plot([], [], [], "-", c="#55dbc6", lw=3.2)
    shadow, = ax.plot([], [], [], "o", c="#aeb7bc", ms=6)
    arm1,   = ax.plot([], [], [], "-", c="#d9dfe2", lw=2.6)
    arm2,   = ax.plot([], [], [], "-", c="#d9dfe2", lw=2.6)
    motor_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    dots = [ax.plot([], [], [], "o", c=color, ms=8)[0] for color in motor_colors]
    nose,   = ax.plot([], [], [], "-", c="#55dbc6", lw=3.2)
    txt = ax.text2D(0.04, 0.96, "", transform=ax.transAxes, family="monospace",
                    color="#f1f3f4", fontsize=11, va="top",
                    bbox={"facecolor": pane, "edgecolor": "#697278", "alpha": 0.9,
                          "boxstyle": "round,pad=0.5"})

    def update(k):
        closing = k >= len(idx)
        if closing:
            i = idx[-1]
            close_k = k - len(idx)
        else:
            i = idx[k]
        R, p = rot(th[i]), pos[i]
        w   = (R @ motors_body.T).T + p          # world positions of the 4 motors

        # Keep the quad prominent while framing the goal; the view tightens on approach.
        focus = 0.55 * p + 0.45 * goal
        span = max(np.max(np.abs(p - goal)) * 1.45, 5.0)
        if closing:
            progress = min(close_k / zoom_frames, 1.0)
            eased = progress * progress * (3 - 2 * progress)
            focus = focus * (1 - eased) + route_center * eased
            span = span * (1 - eased) + route_span * eased
        ax.set_xlim(focus[0] - span / 2, focus[0] + span / 2)
        ax.set_ylim(focus[1] - span / 2, focus[1] + span / 2)
        ax.set_zlim(max(0, focus[2] - span / 2), max(0, focus[2] - span / 2) + span)

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

    anim = FuncAnimation(fig, update, frames=len(idx) + closing_frames,
                         interval=1000 / a.fps, blit=False)
    if a.save:
        if a.save.lower().endswith(".mp4"):
            writer = FFMpegWriter(fps=a.fps, codec="libx264", bitrate=5000,
                                  extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        elif a.save.lower().endswith(".gif"):
            writer = PillowWriter(fps=a.fps)
        else:
            ap.error("--save output must end in .mp4 or .gif")
        anim.save(a.save, writer=writer)
        print("saved", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()