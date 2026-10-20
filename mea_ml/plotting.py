"""Save individual analysis plots."""

import matplotlib.pyplot as plt
from .io import ROOT


def save_png(fig, folder, stem):
    path = folder / f"{stem}.png"
    fig.savefig(path, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    print("Saved:", path.relative_to(ROOT))
