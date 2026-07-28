import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --------------------------------------------------------
# Data
# --------------------------------------------------------

domains = ["Magnet", "Sponge"]

ohcam = [1.0, 1.0]
naive = [0.3, 0.2]
csam = [0.0, 0.4]
ci = [0.0, 0.4]

colors = {
    "OHCAM": "#1f77b4",
    "Naive": "#9467bd",
    "Conditional-SAM": "#ff7f0e",
    "Cluster&Intersect": "#2ca02c",
}

fig = plt.figure(figsize=(11, 5))

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1, 2],
    wspace=0.01,
)

ax = fig.add_subplot(gs[0])

x = np.arange(len(domains))
width = 0.18

methods = {
    "Naive": naive,
    "OHCAM": ohcam,
    "Conditional-SAM": csam,
    "Cluster&Intersect": ci,
}

# Center the four bars around each domain
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

for offset, (method, values) in zip(offsets, methods.items()):
    ax.bar(
        x + offset,
        values,
        width,
        label=method,
        color=colors[method],
        edgecolor="black",
        linewidth=0.8,
    )

# --------------------------------------------------------
# Labels
# --------------------------------------------------------

fs = 18

ax.set_ylabel("Completion Rate (%)", fontsize=fs)

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=fs)

ax.set_ylim(0, 1.03)
ax.tick_params(axis="y", labelsize=fs)

ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)


# --------------------------------------------------------
# Legend
# --------------------------------------------------------

legend_handles = [
    Patch(facecolor=colors["Naive"], label="Naive"),
    Patch(facecolor=colors["OHCAM"], label="OHCAM"),
    Patch(facecolor=colors["Conditional-SAM"], label="Conditional-SAM"),
    Patch(facecolor=colors["Cluster&Intersect"], label="Cluster&Intersect"),
]


ax.legend(
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(1.48, 0.98),
    frameon=True,
    fontsize=16,
    columnspacing=1.5,
    handletextpad=0.5,
)

# --------------------------------------------------------
# Image collage placeholder
# --------------------------------------------------------

photo_gs = gs[1].subgridspec(2, 2, wspace=0.00, hspace=0.01)

img_names = ["blocks_first.png", "sponge_first.png", "blocks_last.png", "sponge_last.png"]

for i in range(4):
    ax_img = fig.add_subplot(photo_gs[i // 2, i % 2])

    # Replace this with:
    img = plt.imread(img_names[i])
    ax_img.imshow(img)

    ax_img.set_xticks([])
    ax_img.set_yticks([])

plt.annotate(
    "",
    xy=(0.47, 0.42),  # arrow head
    xytext=(0.47, 0.51),  # arrow tail
    xycoords="figure fraction",
    textcoords="figure fraction",
    arrowprops=dict(arrowstyle="->", lw=2.5, color="red", mutation_scale=15),
)

# Arrow between the bottom two images
plt.annotate(
    "",
    xy=(0.73, 0.42),
    xytext=(0.73, 0.51),
    xycoords="figure fraction",
    textcoords="figure fraction",
    arrowprops=dict(arrowstyle="->", lw=2.5, color="red", mutation_scale=15),
)
fig.text(
    0.52,  # x position (center)
    0.049,  # y position (near bottom)
    "MagnetBlocks",  # label
    ha="center",
    va="bottom",
    fontsize=fs,
)

fig.text(
    0.77,  # x position (center)
    0.049,  # y position (near bottom)
    "SpongeWorld",  # label
    ha="center",
    va="bottom",
    fontsize=fs,
)
plt.tight_layout()
plt.savefig("hardware_solve_rate.png", bbox_inches="tight")
# plt.show()
