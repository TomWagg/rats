import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)

kstar_translator = [
    None,
    {"long": "Main Sequence", "short": "MS", "colour": None},
    {"long": "Hertzsprung Gap", "short": "HG", "colour": None},
    {"long": "First Giant Branch", "short": "FGB", "colour": None},
    {"long": "Core Helium Burning", "short": "CHeB", "colour": None},
    {"long": "Early Asymptotic Giant Branch", "short": "EAGB", "colour": None},
    {"long": "Thermally Pulsing Asymptotic Giant Branch", "short": "TPAGB", "colour": None},
    {"long": "Helium Main Sequence", "short": "HeMS", "colour": None},
    {"long": "Helium Hertsprung Gap", "short": "HeHG", "colour": None},
    {"long": "Helium Giant Branch", "short": "HeGB", "colour": None},
    {"long": "Helium White Dwarf", "short": "HeWD", "colour": None},
    {"long": "Carbon/Oxygen White Dwarf", "short": "COWD", "colour": None},
    {"long": "Oxygen/Neon White Dwarf", "short": "ONeWD", "colour": None},
    {"long": "Neutron Star", "short": "NS", "colour": mpl.colors.to_rgba("grey")},
    {"long": "Black Hole", "short": "BH", "colour": mpl.colors.to_rgba("black")},
    {"long": "Massless Remnant", "short": "MR", "colour": mpl.colors.to_rgba("yellow")},
    {"long": "Chemically Homogeneous", "short": "CHE", "colour": mpl.colors.to_rgba("brown")}
]

for i in [1, 2]:
    kstar_translator[i]["colour"] = plt.get_cmap("YlOrBr")(0.3 * i)

for i in [3, 4]:
    kstar_translator[i]["colour"] = plt.get_cmap("Blues")(0.3 * (i - 2))

for i in [5, 6]:
    kstar_translator[i]["colour"] = plt.get_cmap("Greens")(0.3 * (3 - (i - 4)))

for i in [7, 8, 9]:
    kstar_translator[i]["colour"] = plt.get_cmap("plasma")(0.1 + 0.2 * (i - 7))

for i in [10, 11, 12]:
    kstar_translator[i]["colour"] = plt.get_cmap("copper")(0.1 + 0.2 * (i - 9))


evol_type_translator = [
    None,
    {"sentence": "Initial state", "short": "Init", "long": "Initial state"},
    {"sentence": "a star changed stellar type", "short": "Kstar change", "long": "Stellar type changed"},
    {"sentence": "Roche lobe overflow started", "short": "RLOF start", "long": "Roche lobe overflow started"},
    {"sentence": "Roche lobe overflow ended", "short": "RLOF end", "long": "Roche lobe overflow ended"},
    {"sentence": "the binary entered a contact phase", "short": "Contact", "long": "Binary entered contact phase"},
    {"sentence": "the binary coalesced", "short": "Coalescence", "long": "Binary coalesced"},
    {"sentence": "a common envelope phase started", "short": "CE start", "long": "Common-envelope started"},
    {"sentence": "the common envelope phase ended", "short": "CE end", "long": "Common-envelope ended"},
    {"sentence": "no remnant leftover", "short": "No remnant", "long": "No remnant"},
    {"sentence": "the maximum evolution time was reached", "short": "Max evolution time", "long": "Maximum evolution time reached"},
    {"sentence": "the binary was disrupted", "short": "Disruption", "long": "Binary disrupted"},
    {"sentence": "a symbiotic phase started", "short": "Begin symbiotic phase", "long": "Begin symbiotic phase"},
    {"sentence": "a symbiotic phase ended", "short": "End symbiotic phase", "long": "End symbiotic phase"},
    {"sentence": "Blue straggler", "short": "Blue straggler", "long": "Blue straggler"},
    {"sentence": "the primary went supernova", "short": "SN1", "long": "Supernova of primary"},
    {"sentence": "the secondary went supernova", "short": "SN2", "long": "Supernova of secondary"},
]

def _use_white_text(rgba):
    r, g, b, _ = rgba
    return (r * 0.299 + g * 0.587 + b * 0.114) < 186 / 255

def _supernova_marker(ax, x, y, s):
    ax.scatter(x, y, marker=(15, 1, 0), s=s * 6, zorder=-1,
               facecolor="#ebd510", edgecolor="#ebb810", linewidth=2)
    ax.scatter(x, y, marker=(10, 1, 0), s=s * 4, zorder=-1,
               facecolor="orange", edgecolor="#eb7e10", linewidth=2)

def _rlof_path(centre, width, height, m=1.5, flip=False):

    t = np.linspace(0, 2 * np.pi, 1000)
    x = 0.5 * width * np.cos(t) * (-1 if flip else 1) + centre[0]
    y = height * np.sin(t) * np.sin(0.5 * t)**(m) + centre[1]

    return x, y

def plot_cartoon_evolution(bpp, bin_num, label_type="long", plot_title="Cartoon Binary Evolution",
                           y_sep_mult=1.5, offset=0.2, s_base=1000, fig=None, ax=None, show=True):
    """Plot COSMIC bpp output as a cartoon evolution

    Parameters
    ----------
    bpp : `pandas.DataFrame`
        COSMIC bpp table
    bin_num : `int`
        Binary number of the binary to plot
    label_type : `str`, optional
        What sort of annotated labels to use ["short", "long", "sentence"], by default "long"
    plot_title : `str`, optional
        Title to use for the plot, use "" for no title, by default "Cartoon Binary Evolution"
    y_sep_mult : `float`, optional
        Multiplier to use for the y separation (larger=more spread out steps, longer figure)
    offset : `float`, optional
        Offset from the centre for each of the stars (larger=wider binaries)
    s_base : `float`, optional
        Base scatter point size for the stars
    """
    # extract the pertinent information from the bpp table
    df = bpp.loc[bin_num][["tphys", "mass_1", "mass_2", "kstar_1", "kstar_2", "porb", "evol_type", "RRLO_1"]]
    
    # add some offset kstar columns to tell what type a star *previously* was
    df[["prev_kstar_1", "prev_kstar_2"]] = df.shift(1, fill_value=0)[["kstar_1", "kstar_2"]]

    # count the number of evolution steps and start figure with size based on that
    total = len(df)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, total * y_sep_mult))

    # instantiate some flags to track state of binary
    i = 0
    disrupted = False
    common_envelope = False
    rlof = False

    # go through each row of the evolution
    for _, row in df.iterrows():
        # use the translators to convert evol_type and kstars
        et_ind, k1, k2, pk1, pk2 = int(row["evol_type"]), kstar_translator[int(row["kstar_1"])],\
            kstar_translator[int(row["kstar_2"])], kstar_translator[int(row["prev_kstar_1"])],\
            kstar_translator[int(row["prev_kstar_2"])]
        et = evol_type_translator[et_ind]

        # set disrupted, rlof and common-envelope flags are necessary
        if et_ind == 11 or row["porb"] < 0.0:
            disrupted = True
        if et_ind == 3:
            rlof = True
        if et_ind == 4:
            rlof = False
        if et_ind == 7:
            common_envelope = True
        if et_ind == 8:
            common_envelope = False

        # check if either star is now a massless remnant
        mr_1 = k1["short"] == "MR"
        mr_2 = k2["short"] == "MR"
        ks_fontsize = 0.3 * fs

        # start an evolution label variable
        evol_label = et[label_type]

        # if the star just evolved then edit to label to explain what happened
        if et_ind == 2:
            which_star = "Primary" if k1 != pk1 else "Secondary"
            to_type = k1[label_type] if k1 != pk1 else k2[label_type]
            evol_label = f'{which_star} evolved to\n{to_type}'

        # annotate the evolution label and time either side of the binary
        ax.annotate(evol_label, xy=(0.5, total - i), va="center")
        ax.annotate(f'{row["tphys"]:1.2e} Myr' if row["tphys"] > 1e4 else f'{row["tphys"]:1.2f} Myr',
                    xy=(-offset - 0.3, total - i), ha="right", va="center",
                    fontsize=0.4*fs, fontweight="bold")
        
        # if we've got a common envelope then draw an ellipse behind the binary
        if common_envelope:
            envelope = mpl.patches.Ellipse(xy=(0, total - i), width=4 * offset, height=1.5,
                                           facecolor="orange", edgecolor="none", zorder=-1, alpha=0.5)
            envelope_edge = mpl.patches.Ellipse(xy=(0, total - i), width=4 * offset, height=1.5,
                                           facecolor="none", edgecolor="darkorange", lw=2)
            ax.add_artist(envelope)
            ax.add_artist(envelope_edge)

        # if either star is a massless remnant then we're just dealing with a single star now
        if mr_1 or mr_2:
            # plot the star centrally and a little larger
            ax.scatter(0, total - i, color=k1["colour"] if mr_2 else k2["colour"], s=s_base * 1.5)

            # label its stellar type if (a) it changed or (b) we're at the start/end of evolution
            if (k1 != pk1 and not mr_1) or (k2 != pk2 and not mr_2) or et_ind in [1, 10]:
                ax.annotate(k1["short"] if k1 != pk1 else k2["short"], xy=(0, total - i),
                            ha="center", va="center", zorder=10, fontsize=ks_fontsize, fontweight="bold",
                            color="white" if _use_white_text(k1["colour"]
                                                             if mr_2 else k2["colour"]) else "black")

            # annotate the correct mass
            ax.annotate(f'{row["mass_1"] if mr_2 else row["mass_2"]:1.2f} ' + r'$\rm M_{\odot}$',
                        xy=(0, total - i - 0.45), ha="center", va="top", fontsize=0.3*fs)
            
            # if a supernova just happened then add an explosion marker behind the star
            if et_ind in [15, 16]:
                _supernova_marker(ax, 0, total - i, s_base)

        # otherwise we've got two stars
        else:
            # plot stars offset from the centre
            ax.scatter(0 - offset, total - i, color=k1["colour"], s=s_base, zorder=10)
            ax.scatter(0 + offset, total - i, color=k2["colour"], s=s_base, zorder=10)

            # annotate the mass (with some extra padding if there's RLOF)
            mass_y_offset = 0.35 if not (rlof and not common_envelope) else 0.5
            ax.annotate(f'{row["mass_1"]:1.2f} ' + r'$\rm M_{\odot}$',
                        xy=(0 - offset, total - i - mass_y_offset), ha="center", va="top", fontsize=0.3*fs)
            ax.annotate(f'{row["mass_2"]:1.2f} ' + r'$\rm M_{\odot}$',
                        xy=(0 + offset, total - i - mass_y_offset), ha="center", va="top", fontsize=0.3*fs)

            # if the primary type changed or we're at the start/end then label it
            if k1 != pk1 or et_ind in [1, 10]:
                ax.annotate(k1["short"], xy=(0 - offset, total - i),
                            ha="center", va="center",
                            color="white" if _use_white_text(k1["colour"]) else "black",
                            zorder=10, fontsize=ks_fontsize, fontweight="bold")
                
            # if the secondary type changed or we're at the start/end then label it
            if k2 != pk2 or et_ind in [1, 10]:
                ax.annotate(k2["short"], xy=(0 + offset, total - i),
                            ha="center", va="center",
                            color="white" if _use_white_text(k2["colour"]) else "black",
                            zorder=10, fontsize=ks_fontsize, fontweight="bold")

            # for bound binaries plot a line connecting them
            if not disrupted:
                ax.plot([0 - offset, 0 + offset], [total - i, total - i],
                        linestyle="--", zorder=-1, color="black")
                
                # annotate the line with period, offset to one side if there's RLOF
                x = 0 if not (rlof and not common_envelope) else (-offset / 4 if row["RRLO_1"] >= 1.0 else offset / 4)
                ax.annotate(f'{row["porb"]:1.2e} days' if row["porb"] > 10000 else f'{row["porb"]:1.0f} days',
                            xy=(x, total - i + 0.05), ha="center", va="bottom", fontsize=0.3*fs)

            # for non-common-envelope RLOF, plot a RLOF teardrop in the background
            if rlof and not common_envelope:
                # flip the shape depending on the direction
                if row["RRLO_1"] >= 1.0:
                    x, y = _rlof_path((0 - offset / 2.6, total - i), 2 * offset, 0.6, flip=False)
                else:
                    x, y = _rlof_path((0 + offset / 2.6, total - i), 2 * offset, 0.6, flip=True)
                ax.plot(x, y, color="darkorange", lw=2)
                ax.fill_between(x, y, color="orange", alpha=0.5, edgecolor="none", zorder=-2)

            # add supernova explosion markers as necessary
            if et_ind == 15:
                _supernova_marker(ax, 0 - offset, total - i, s_base / 1.5)
            if et_ind == 16:
                _supernova_marker(ax, 0 + offset, total - i, s_base / 1.5)

        # increment by multiplier
        i += y_sep_mult

    # clear off any x-ticks and axes
    ax.set_xlim(-1.5, 1.5)
    ax.set_xticks([])
    ax.axis("off")

    # annotate a title as the top
    ax.annotate(plot_title, xy=(0, total + 0.75), ha="center", va="center", fontsize=fs * 1.2)

    if show:
        plt.show()
    return fig, ax
