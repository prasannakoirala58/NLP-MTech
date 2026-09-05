"""Paste the generated plots into the blank boxes of the handwritten C2 scan.

The student ruled an empty rectangle on pages 7 and 8 and wrote a placeholder
note inside each. We white out the interior of the box (keeping the hand-drawn
border) and drop the matching figure in, scaled to fit and centred.
"""

from PIL import Image, ImageDraw
import subprocess, tempfile, glob, json, os

BASE = ("/Users/prasa/Desktop/M.Tech/Sem-II/NLP/nlp-assignments/"
        "assignments/a3-embeddings/c2-embedding-space-exploration")
SRC = f"{BASE}/PrasannaKoirala_Embeddings_20016_MTechAI.pdf"
OUT = f"{BASE}/PrasannaKoirala_Embeddings_C2.pdf"
SCR = ("/private/tmp/claude-501/-Users-prasa-Desktop-M-Tech-Sem-II-NLP-"
       "nlp-assignments/bfdf0e0a-48dc-4b32-ae43-bdd8c80305ae/scratchpad")

boxes = json.load(open(f"{SCR}/boxes.json"))
FIGURES = {7: f"{BASE}/outputs/figure1_pca.png",
           8: f"{BASE}/outputs/figure2_tsne.png"}

DPI = 175                       # matches the C1 compression, keeps size sane
tmp = tempfile.mkdtemp()
subprocess.run(["pdftoppm", "-r", str(DPI), "-png", SRC, f"{tmp}/pg"], check=True)

pages = []
for path in sorted(glob.glob(f"{tmp}/pg-*.png")):
    n = int(path.split("-")[-1].split(".")[0])
    page = Image.open(path).convert("RGB")
    W, H = page.size

    if n in FIGURES:
        b = boxes[str(n)]
        x0, x1 = int(b["L"] * W), int(b["R"] * W)
        y0, y1 = int(b["T"] * H), int(b["B"] * H)

        # wipe the placeholder handwriting, but stay inside the ruled border
        pad = 4
        ImageDraw.Draw(page).rectangle(
            [x0 + pad, y0 + pad, x1 - pad, y1 - pad], fill="white")

        # scale the figure to fit the box, preserving its aspect ratio
        fig = Image.open(FIGURES[n]).convert("RGB")
        inner_w, inner_h = (x1 - x0) - 2 * pad - 8, (y1 - y0) - 2 * pad - 8
        scale = min(inner_w / fig.width, inner_h / fig.height)
        new = (max(1, int(fig.width * scale)), max(1, int(fig.height * scale)))
        fig = fig.resize(new, Image.LANCZOS)

        # centre it in the box
        px = x0 + (x1 - x0 - fig.width) // 2
        py = y0 + (y1 - y0 - fig.height) // 2
        page.paste(fig, (px, py))
        print(f"  page {n}: box {x1-x0}x{y1-y0}px -> figure placed at "
              f"{fig.width}x{fig.height}px")

    # Handwriting is black ink, so greyscale costs nothing and halves the size.
    # The two figure pages stay in colour, because the clusters are colour-coded.
    pages.append(page if n in FIGURES else page.convert("L"))

pages[0].save(OUT, save_all=True, append_images=pages[1:],
              resolution=float(DPI), quality=72, optimize=True)

print(f"\nwrote {OUT}")
print(f"  pages : {len(pages)}")
print(f"  size  : {os.path.getsize(OUT)/1024/1024:.2f} MB "
      f"(source was {os.path.getsize(SRC)/1024/1024:.2f} MB)")
