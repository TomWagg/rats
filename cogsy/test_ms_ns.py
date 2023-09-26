import matplotlib.pyplot as plt
import pandas as pd
from cartoon import plot_cartoon_evolution

bpp = pd.read_hdf("options_1.h5", key="df")
fig, ax = plot_cartoon_evolution(bpp, 159, plot_title="", show=False, fig=fig, ax=ax)
plt.savefig("cartoon.png", format="png", dpi=600, bbox_inches="tight")