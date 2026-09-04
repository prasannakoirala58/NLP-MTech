"""Generate the PCA and t-SNE figures for C2."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUT = ("/Users/prasa/Desktop/M.Tech/Sem-II/NLP/nlp-assignments/"
       "assignments/a3-embeddings/c2-embedding-space-exploration/outputs")

g = api.load("glove-wiki-gigaword-100")

GROUPS = {
    "Animals":   ["cat", "dog", "horse", "cow", "tiger", "elephant"],
    "Royalty":   ["king", "queen", "prince", "princess", "throne", "crown"],
    "Countries": ["nepal", "india", "france", "japan", "brazil", "egypt"],
    "Technology":["computer", "software", "internet", "laptop", "keyboard", "server"],
    "Food":      ["rice", "bread", "apple", "banana", "cheese", "soup"],
}

# restrained, print-friendly palette
COLOURS = {
    "Animals":    "#b45309",
    "Royalty":    "#6d28d9",
    "Countries":  "#047857",
    "Technology": "#1d4ed8",
    "Food":       "#be123c",
}
MARKERS = {"Animals": "o", "Royalty": "s", "Countries": "^",
           "Technology": "D", "Food": "v"}

words, labels = [], []
for group, ws in GROUPS.items():
    for w in ws:
        words.append(w)
        labels.append(group)

X = np.array([g[w] for w in words])
print(f"{len(words)} words, each {X.shape[1]} dimensions")


def place_labels(ax, fig, coords, texts):
    """Put a label near each point without letting labels overlap.

    Words like queen / king / princess land almost on top of each other once
    100 dimensions are squashed into 2, so a single fixed offset guarantees
    unreadable overlaps. This tries positions at increasing distance from the
    point and draws a thin leader line whenever a label ends up far away.
    """
    import math

    # rings of candidate positions, nearest first
    candidates = []
    for radius in (8, 17, 27, 38):
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            candidates.append((radius * math.cos(rad), radius * math.sin(rad)))

    fig.canvas.draw()                       # bboxes are only valid after a draw
    renderer = fig.canvas.get_renderer()

    # start with the marker positions themselves as obstacles
    placed = []
    for x, y in coords:
        px, py = ax.transData.transform((x, y))
        placed.append(Bbox.from_bounds(px - 5, py - 5, 10, 10))

    # place isolated points first so crowded ones use the leftover space
    order = sorted(range(len(texts)), key=lambda i: -abs(coords[i, 1]))

    for i in order:
        x, y = coords[i]
        for dx, dy in candidates:
            ann = ax.annotate(texts[i], (x, y),
                              xytext=(dx, dy), textcoords="offset points",
                              fontsize=7.6, color="#2b3038", zorder=4,
                              ha="left" if dx >= 0 else "right",
                              va="bottom" if dy >= 0 else "top")
            bb = ann.get_window_extent(renderer=renderer).expanded(1.06, 1.18)
            if not any(bb.overlaps(other) for other in placed):
                placed.append(bb)
                # if the label sat far from its marker, join them with a line
                if math.hypot(dx, dy) > 20:
                    ann.set_arrowprops = None
                    ax.annotate("", xy=(x, y), xycoords="data",
                                xytext=(dx * 0.55, dy * 0.55),
                                textcoords="offset points",
                                arrowprops=dict(arrowstyle="-", lw=0.5,
                                                color="#9aa2b1",
                                                shrinkA=0, shrinkB=2),
                                zorder=2)
                break
            ann.remove()                    # collided, try the next position
        else:
            ann = ax.annotate(texts[i], (x, y), xytext=(6, 4),
                              textcoords="offset points",
                              fontsize=7.6, color="#2b3038", zorder=4)
            placed.append(ann.get_window_extent(renderer=renderer))


def draw(coords, title, subtitle, filename):
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)

    for group in GROUPS:
        idx = [i for i, l in enumerate(labels) if l == group]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=COLOURS[group], marker=MARKERS[group],
                   s=62, alpha=0.9, edgecolors="white", linewidths=1.1,
                   label=group, zorder=3)

    # give the data a little breathing room so labels are not clipped
    ax.margins(0.13)

    # title sits well above the axes; pad leaves room for the subtitle below it
    ax.set_title(title, fontsize=13, fontweight="bold", pad=30, loc="left",
                 color="#16181d")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9,
            color="#5b6270", va="bottom")

    ax.legend(frameon=False, fontsize=8.6, loc="best", ncol=2,
              handletextpad=0.4, columnspacing=1.1)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9cdd6")
    ax.tick_params(colors="#8b93a3", labelsize=7.5)
    ax.grid(True, linewidth=0.5, color="#eceef2", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    place_labels(ax, fig, coords, words)     # after layout, so bboxes are final

    path = f"{OUT}/{filename}"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ---- PCA: keeps the directions of greatest variance, straight linear projection
pca = PCA(n_components=2, random_state=0)
pca_xy = pca.fit_transform(X)
var = pca.explained_variance_ratio_
print(f"PCA variance explained: PC1 {var[0]:.1%}, PC2 {var[1]:.1%}, "
      f"total {var.sum():.1%}")
draw(pca_xy, "PCA of 30 word embeddings",
     f"100 dimensions reduced to 2 · captures {var.sum():.0%} of the total variance",
     "figure1_pca.png")

# ---- t-SNE: non-linear, optimised to keep NEIGHBOURS together
tsne = TSNE(n_components=2, random_state=42, perplexity=8,
            init="pca", max_iter=1500)
tsne_xy = tsne.fit_transform(X)
draw(tsne_xy, "t-SNE of the same 30 word embeddings",
     "non-linear · preserves near neighbours, not overall distances",
     "figure2_tsne.png")

print(f"\nPCA total variance retained: {var.sum():.1%} "
      f"(so {1-var.sum():.0%} of the information is lost in the picture)")
