"""Paste the two generated figures into the blank boxes of the a4 handwritten scan.

Pages 10 and 13 each carry a ruled rectangle with a placeholder note inside. We
locate each rectangle by ink density, white out its interior while keeping the
hand-drawn border, and drop the matching figure in, scaled to fit and centred.
"""
from PIL import Image, ImageDraw
import numpy as np
import subprocess, tempfile, glob, os

BASE = ("/Users/prasa/Desktop/M.Tech/Sem-II/NLP/nlp-assignments/"
        "assignments/a4-rnn-bptt")
SRC = f"{BASE}/Prasanna_Koirala_20016_RNN-BPTT.pdf"
OUT = f"{BASE}/PrasannaKoirala_RNN_BPTT.pdf"
SCR = ("/private/tmp/claude-501/-Users-prasa-Desktop-M-Tech-Sem-II-NLP-"
       "nlp-assignments/bfdf0e0a-48dc-4b32-ae43-bdd8c80305ae")

FIGURES = {10: f"{SCR}/fig1_final.png",      # gradient over 30 timesteps
           13: f"{SCR}/fig2_final.png"}      # printed .grad for every parameter

# rough region each box sits in, read off the rendered pages (fractions of the page)
PRIOR = {10: dict(y=(0.53, 0.96), x=(0.10, 0.95)),
         13: dict(y=(0.11, 0.60), x=(0.10, 0.95))}

DPI = 175


def find_box(gray, prior):
    """Locate the ruled rectangle inside the given region, by ink density."""
    H, W = gray.shape
    dark = (gray < 165).astype(float)
    py, px = prior["y"], prior["x"]

    y_lo, y_hi = int(H * py[0]), int(H * py[1])
    x_lo, x_hi = int(W * px[0]), int(W * px[1])

    # density of ink in each column, restricted to the box's vertical band
    col = dark[y_lo:y_hi, :].mean(axis=0)
    row = dark[:, x_lo:x_hi].mean(axis=1)

    def peak(sig, lo, hi, take):
        seg = sig[lo:hi]
        hits = [i for i, v in enumerate(seg) if v >= seg.max() * 0.55]
        return lo + (hits[0] if take == "first" else hits[-1])

    x_mid, y_mid = (x_lo + x_hi) // 2, (y_lo + y_hi) // 2
    return (peak(col, x_lo, x_mid, "first"), peak(col, x_mid, x_hi, "last"),
            peak(row, y_lo, y_mid, "first"), peak(row, y_mid, y_hi, "last"))


tmp = tempfile.mkdtemp()
subprocess.run(["pdftoppm", "-r", str(DPI), "-png", SRC, f"{tmp}/pg"], check=True)

pages = []
for path in sorted(glob.glob(f"{tmp}/pg-*.png")):
    n = int(path.split("-")[-1].split(".")[0])
    page = Image.open(path).convert("RGB")

    if n in FIGURES:
        gray = np.array(page.convert("L"))
        x0, x1, y0, y1 = find_box(gray, PRIOR[n])
        print(f"  page {n}: box {x1-x0} x {y1-y0} px")

        pad = 5
        ImageDraw.Draw(page).rectangle(
            [x0 + pad, y0 + pad, x1 - pad, y1 - pad], fill="white")

        fig = Image.open(FIGURES[n]).convert("RGB")
        inner_w = (x1 - x0) - 2 * pad - 10
        inner_h = (y1 - y0) - 2 * pad - 10
        scale = min(inner_w / fig.width, inner_h / fig.height)
        fig = fig.resize((max(1, int(fig.width * scale)),
                          max(1, int(fig.height * scale))), Image.LANCZOS)

        page.paste(fig, (x0 + (x1 - x0 - fig.width) // 2,
                         y0 + (y1 - y0 - fig.height) // 2))
        print(f"           figure placed at {fig.width} x {fig.height} px")

    # handwriting is black ink, so greyscale halves the size at no cost.
    # the two figure pages stay in colour - the plot lines are colour-coded.
    pages.append(page if n in FIGURES else page.convert("L"))

pages[0].save(OUT, save_all=True, append_images=pages[1:],
              resolution=float(DPI), quality=72, optimize=True)

print(f"\nwrote {OUT}")
print(f"  pages : {len(pages)}")
print(f"  size  : {os.path.getsize(OUT)/1024/1024:.2f} MB "
      f"(source {os.path.getsize(SRC)/1024/1024:.2f} MB)")
