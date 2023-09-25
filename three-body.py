import rebound
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from matplotlib import animation

plt.style.use('dark_background')


labels = ["NS", "MS", "TTS"]
kms_to_AUyr = (u.km / u.s).to(u.AU / u.yr)


def simulate_exchange(particles, n_steps=10000):
    # start a REBOUND simulation
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    for p in particles:
        sim.add(**p)

    tracks = np.zeros((len(particles), 2, n_steps))

    timesteps = np.linspace(0, int(n_steps / 100), n_steps + 1)

    for i in range(n_steps):
        for j, p in enumerate(sim.particles):
            tracks[j, 0, i] = p.x
            tracks[j, 1, i] = p.y
        sim.integrate((i + 1) / 100)

    # success = (TTS closer to NS than MS is) & (TTS -- NS are close)
    MS_to_NS = np.sum((tracks[0, :, -1] - tracks[1, :, -1])**2, axis=0)**(0.5)
    TTS_to_NS = np.sum((tracks[0, :, -1] - tracks[2, :, -1])**2, axis=0)**(0.5)
    MS_to_TTS = np.sum((tracks[1, :, -1] - tracks[2, :, -1])**2, axis=0)**(0.5)

    status = "uh oh"
    if not any([MS_to_NS < 10, TTS_to_NS < 10, MS_to_TTS < 10]):
        status = "disruption"
    elif TTS_to_NS < MS_to_NS:
        status = "exchange"
    elif MS_to_TTS < MS_to_NS:
        status = "wrong exchange"
    else:
        status = "original setup"

    return status, tracks, timesteps


def add_stars(ax, starsurfacedensity=0.8, lw=1):
    starcolor = (1.,1.,1.)
    area = np.sqrt(np.sum(np.square(ax.transAxes.transform([1.,1.]) - ax.transAxes.transform([0.,0.]))))*1
    nstars = int(starsurfacedensity*area)

    #small stars
    xy = np.random.uniform(size=(nstars,2))
    ax.scatter(xy[:,0],xy[:,1], transform=ax.transAxes, alpha=0.05, s=8*lw, facecolor=starcolor, edgecolor=None, zorder=3, rasterized=True)
    ax.scatter(xy[:,0],xy[:,1], transform=ax.transAxes, alpha=0.1, s=4*lw, facecolor=starcolor, edgecolor=None, zorder=3, rasterized=True)
    ax.scatter(xy[:,0],xy[:,1], transform=ax.transAxes, alpha=0.2, s=0.5*lw, facecolor=starcolor, edgecolor=None, zorder=3, rasterized=True)

    #large stars
    xy = np.random.uniform(size=(nstars//4,2))
    ax.scatter(xy[:,0],xy[:,1], transform=ax.transAxes, alpha=0.1, s=15*lw, facecolor=starcolor, edgecolor=None, zorder=3, rasterized=True)
    ax.scatter(xy[:,0],xy[:,1], transform=ax.transAxes, alpha=0.1, s=5*lw, facecolor=starcolor, edgecolor=None, zorder=3, rasterized=True)
    ax.scatter(xy[:,0],xy[:,1], transform=ax.transAxes, alpha=0.5, s=2*lw, facecolor=starcolor, edgecolor=None, zorder=3, rasterized=True)

def plot_full_tracks(tracks, save=False, inset_lim=2000):
    fig, ax = plt.subplots(figsize=(6, 14))
    for i, label in enumerate(labels):
        ax.plot(tracks[i, 0], tracks[i, 1], label=label)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    ax.set(xlabel=r'$x \, [\rm AU]$', ylabel=r'$y \, [\rm AU]$')

    add_stars(ax)

    # add inset zoom in for the star
    inset = ax.inset_axes([0.6, 0.05, 0.35, 0.4])
    inset.set_xticks([])
    inset.set_yticks([])
    for i, label in enumerate(labels):
        inset.plot(tracks[i, 0, :inset_lim], tracks[i, 1, :inset_lim])
    for i, label in enumerate(labels):
        inset.scatter(tracks[i, 0, 0], tracks[i, 1, 0])
    
    add_stars(inset, starsurfacedensity=0.4)

    # plot a box around where the inset covers
    ix_min, ix_max = inset.get_xlim()
    iy_min, iy_max = inset.get_ylim()

    lw = 0.7
    ax.plot([ix_min, ix_max], [iy_min, iy_min], color="white", lw=lw)
    ax.plot([ix_max, ix_max], [iy_min, iy_max], color="white", lw=lw)
    ax.plot([ix_max, ix_min], [iy_max, iy_max], color="white", lw=lw)
    ax.plot([ix_min, ix_min], [iy_min, iy_max], color="white", lw=lw)

    # connect the boxes
    ax.annotate("", xy=(0.22, 0.06), xytext=(0.6, 0.15), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<|-", color="white", lw=1))

    if save is not None:
        plt.savefig('paper/figures/rat_formation.pdf' if isinstance(save, bool) else save,
                    format="pdf", bbox_inches="tight", dpi=600)

    plt.show()


def animate_tracks(tracks, timesteps, save=None):
    distances = np.maximum(np.maximum(np.sum((tracks[0, :, :] - tracks[1, :, :])**2, axis=0)**(0.5),
                    np.sum((tracks[0, :, :] - tracks[2, :, :])**2, axis=0)**(0.5)),
                np.sum((tracks[1, :, :] - tracks[2, :, :])**2, axis=0)**(0.5))

    cursor = 0
    plot_timesteps = []

    while cursor < len(timesteps) - 1:
        plot_timesteps.append(cursor)
        cursor += max(int(5 * distances[cursor] / 10), 1)


    fig, ax = plt.subplots()

    scatters = [ax.scatter(tracks[i, 0, 0], tracks[i, 1, 0], label=labels[i]) for i in range(len(labels))]
    lines = [ax.plot(tracks[i, 0, 0], tracks[i, 1, 0])[0] for i in range(len(labels))]
    timer = ax.annotate(rf"$t = {{{timesteps[0]:1.1f}}} \, {{\rm yr}}$", xy=(0.99, 1.04), xycoords="axes fraction",
                        ha="right", va="bottom")

    ax.set(xlabel=r'$x \, [\rm AU]$', ylabel=r'$y \, [\rm AU]$')
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 1.01), ncol=3)

    trail_length = 50

    def update(frame):

        trail_start = max(0, frame - trail_length)

        for i in range(len(scatters)):
            scatters[i].set_offsets([[tracks[i, 0, frame], tracks[i, 1, frame]]])
            lines[i].set_xdata(tracks[i, 0, trail_start:frame + 1])
            lines[i].set_ydata(tracks[i, 1, trail_start:frame + 1])

        timer.set_text(rf"$t = {{{timesteps[frame]:1.1f}}} \, {{\rm yr}}$")

        x_min, x_max = tracks[:, 0, trail_start:frame + 1].min(), tracks[:, 0, trail_start:frame + 1].max()
        y_min, y_max = tracks[:, 1, trail_start:frame + 1].min(), tracks[:, 1, trail_start:frame + 1].max()

        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)

        return (scatters, lines)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=plot_timesteps, interval=1, blit=False, repeat=False)

    if save is not None:
        FFwriter = animation.FFMpegWriter(fps=30)
        ani.save('rat_formation.mp4' if isinstance(save, bool) else save, writer = FFwriter, dpi=300)


    plt.show()

## ** this one looks great **
# status, tracks, timesteps = simulate_exchange(particles=[
#     dict(hash="NS", m=1.4),
#     dict(hash="MS", m=0.8, a=2.0),
#     dict(hash="TTS", m=2, y=10, vy=-1, vz=1)
# ], n_steps=20000)

# status, tracks, timesteps = simulate_exchange(particles=[
#     dict(hash="NS", m=1.4, y=-25, vy=30 * kms_to_AUyr),
#     dict(hash="MS", m=0.3, a=2.5),
#     dict(hash="TTS", m=0.9, y=-45, z=-1, vy=45 * kms_to_AUyr, vz=1 * kms_to_AUyr)
# ], n_steps=15000)

# status, tracks, timesteps = simulate_exchange(particles=[
#     dict(hash="NS", m=1.4, y=-25, vy=30 * kms_to_AUyr),
#     dict(hash="MS", m=0.3, a=2.5),
#     dict(hash="TTS", m=0.9, y=-45, z=-1, vx=2.5 * kms_to_AUyr, vy=45 * kms_to_AUyr, vz=1 * kms_to_AUyr)
# ], n_steps=60000)

# np.save("tracks.npy", tracks)
# np.save("timesteps.npy", timesteps)

tracks = np.load("tracks.npy")
timesteps = np.load("timesteps.npy")
status = "exchange"

PLOT_ANYWAY = False

if status == "exchange" or PLOT_ANYWAY:
    plot_full_tracks(tracks, save=True)
    animate_tracks(tracks, timesteps, save=False)
else:
    print(f"No exchange occurred\n\t**{status.upper()}**")
