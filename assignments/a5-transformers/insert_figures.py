"""Build the final handwritten-notes PDF from the phone scan.

Run from anywhere:
    uv run --no-sync python assignments/a5-transformers/insert_figures.py

What it does, page by page:
  1. Takes the original page photo out of the CamScanner PDF. Each page in that PDF is one
     big photo plus a small "CS" logo stamped on top. We only take the photo, so the logo
     disappears without touching the handwriting.
  2. Whitens the bits of table cloth that show at a few page edges.
  3. Pastes the question snapshots (question-images/q01..q10.png) and the three notebook
     plots (outputs/*.png) into the blank gaps left for them while copying the notes.
  4. Shrinks the file from ~25 MB to ~7 MB: greyscale (the ink is black anyway), 150 dpi
     (a normal document-scan resolution), and the faint grey speckle of the paper flattened
     to clean white. Ink, pencil shading and the ruled lines are all kept.

Output: PrasannaKoirala_Transformer.pdf, next to the notebook.

The raw phone scan ("handwritten assignment.pdf", 25 MB) is not kept in the repo; only the
finished PDF is. To rebuild, put a scan back under that name and run the script again.
"""
import glob
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).parent
SOURCE = HERE / "handwritten assignment.pdf"           # the raw phone scan (not committed, 25 MB)
OUTPUT = HERE / "PrasannaKoirala_Transformer.pdf"

# Final page: A4 at 150 dpi.
DPI = 150
PAGE_W, PAGE_H = 1240, 1754
QUALITY_TEXT = 70        # JPEG quality for pages that are only handwriting
QUALITY_PASTED = 80      # pages with a pasted question or plot: keep the printed text crisp
PX = DPI / 200           # the pixel sizes below were chosen at 200 dpi; this rescales them

# What goes where. For each image: the page it goes on, the vertical band (as a fraction of
# the page photo's height) where the blank gap is, and its width as a fraction of the page
# width. 16 cm on a ~21 cm page = 0.76; the temperature plot's box was 13 cm = 0.62.
QUESTIONS = HERE / "question-images"
PLOTS = HERE / "outputs"
SLOTS = [
    # page, band (top, bottom), image,                                     width
    (1,  (0.54, 0.99), QUESTIONS / "q01.png",                              0.76),
    (5,  (0.48, 0.85), PLOTS / "q1c_positional_encoding.png",              0.76),
    (7,  (0.15, 0.63), QUESTIONS / "q02.png",                              0.76),
    (11, (0.14, 0.55), QUESTIONS / "q03.png",                              0.76),
    (16, (0.53, 0.99), QUESTIONS / "q04.png",                              0.76),
    (20, (0.60, 0.99), QUESTIONS / "q05.png",                              0.76),
    (24, (0.12, 0.64), QUESTIONS / "q06.png",                              0.76),
    (28, (0.48, 0.99), QUESTIONS / "q07.png",                              0.76),
    (30, (0.18, 0.66), PLOTS / "q7c_temperature.png",                      0.62),
    (32, (0.55, 0.99), QUESTIONS / "q08.png",                              0.76),
    (35, (0.50, 0.99), QUESTIONS / "q09.png",                              0.76),
    (39, (0.13, 0.56), QUESTIONS / "q10.png",                              0.76),
    (41, (0.19, 0.57), PLOTS / "q10b_benchmark.png",                       0.76),
]


def extract_page_photos(pdf, folder):
    """Pull the original JPEG of every page out of the PDF, without re-compressing it.

    `pdfimages -p` puts the page number in each file name. On every page the biggest image
    is the photo of the notebook page; the small one is the CamScanner logo, which we drop.
    """
    subprocess.run(["pdfimages", "-j", "-p", str(pdf), f"{folder}/img"], check=True)
    photos = {}
    for path in sorted(glob.glob(f"{folder}/img-*")):
        page = int(Path(path).name.split("-")[1])
        size = os.path.getsize(path)
        if page not in photos or size > os.path.getsize(photos[page]):
            photos[page] = path
    return photos


def keep_thick(mask, axis, size):
    """Keep only the parts of a mask that are at least `size` pixels thick along `axis`.

    This is a one-direction "opening": first keep the pixels whose whole window of `size`
    pixels (along the axis) is set, then grow those windows back to full size. A thin
    line across that direction disappears; a thick patch keeps its exact shape.
    """
    m = np.moveaxis(mask, axis, 0).astype(int)
    zeros = np.zeros((1,) + m.shape[1:], dtype=int)
    running = np.cumsum(np.concatenate([zeros, m]), axis=0)
    full_window = (running[size:] - running[:-size]) == size        # window starting at row r is all set
    # A pixel survives if at least one full window covers it.
    starts = np.cumsum(np.concatenate([zeros, full_window.astype(int)]), axis=0)
    covered = np.zeros_like(m)
    for row in range(m.shape[0]):
        first, last = max(0, row - size + 1), min(row, full_window.shape[0] - 1)
        if last >= first:
            covered[row] = (starts[last + 1] - starts[first]) > 0
    return np.moveaxis(covered.astype(bool), 0, axis)


def whiten_cloth_at_edges(photo):
    """Turn the table cloth and shadow that peek in at some page edges into white paper.

    Paper is bright and nearly colourless, so anything else is "not paper": ink, ruled lines,
    the red margin line, and the cloth (dark red and black, measured on this scan). What makes
    the cloth different is that it is joined to the edge of the photo and is a thick patch.
    So we walk in from each edge and whiten the unbroken run of not-paper that starts at the
    edge, at most 6% deep. Walking in from the left/right we ignore anything flat (the ruled
    lines); walking in from the top/bottom we ignore anything narrow (the red margin line).
    """
    # Work on a quarter-size copy: cloth patches are big, so this loses nothing and is 16x faster.
    small = photo.resize((photo.width // 4, photo.height // 4), Image.BOX)
    rgb = np.asarray(small).astype(int)
    grey = rgb.mean(axis=2)
    colourfulness = rgb.max(axis=2) - rgb.min(axis=2)
    not_paper = (grey < 170) | (colourfulness > 60)

    tall_parts = keep_thick(not_paper, axis=0, size=7)     # ≥ 7 px tall (28 px at full size)
    wide_parts = keep_thick(not_paper, axis=1, size=7)     # ≥ 7 px wide

    height, width = not_paper.shape
    depth_y, depth_x = int(0.06 * height), int(0.06 * width)
    touches_edge = np.zeros_like(not_paper)

    # Walk in from each edge and keep the unbroken run that starts at the edge.
    # cumprod turns [1,1,1,0,1] into [1,1,1,0,0]: everything after the first gap is dropped.
    touches_edge[:depth_y, :] |= np.cumprod(wide_parts[:depth_y, :], axis=0).astype(bool)                   # top
    touches_edge[-depth_y:, :] |= np.cumprod(wide_parts[::-1][:depth_y], axis=0)[::-1].astype(bool)          # bottom
    touches_edge[:, :depth_x] |= np.cumprod(tall_parts[:, :depth_x], axis=1).astype(bool)                    # left
    touches_edge[:, -depth_x:] |= np.cumprod(tall_parts[:, ::-1][:, :depth_x], axis=1)[:, ::-1].astype(bool)  # right

    # Grow the patch a little to catch its soft, blurred border, then scale the mask back up.
    grown = Image.fromarray((touches_edge * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    full_mask = np.asarray(grown.resize(photo.size, Image.NEAREST)) > 0
    cleaned = np.asarray(photo).copy()
    cleaned[full_mask] = 255
    return Image.fromarray(cleaned), int(full_mask.sum())


def fit_on_a4(photo):
    """Scale the page photo to fit an A4 page and centre it on white. Works for colour or grey."""
    scale = min(PAGE_W / photo.width, PAGE_H / photo.height)
    resized = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.LANCZOS)
    page = Image.new(photo.mode, (PAGE_W, PAGE_H), "white")
    top = (PAGE_H - resized.height) // 2
    left = (PAGE_W - resized.width) // 2
    page.paste(resized, (left, top))
    return page, top, resized.height


def clean_paper(grey_page):
    """Flatten the paper's faint grey speckle (show-through from the back, JPEG noise) to white.

    Everything darker than 200 (ink, pencil, ruled lines) is left exactly as it is. Between
    200 and 245 the tones are gently lifted, so light pencil shading (~215-235) stays visible.
    245 and up becomes pure white. Flat white also compresses far better.
    """
    knee, white = 200, 245
    values = np.asarray(grey_page).astype(float)
    lifted = knee + (values - knee) * (255 - knee) / (white - knee)
    out = np.where(values < knee, values, lifted)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def find_margin_line(colour_page):
    """x position of the red margin line: the reddest column in the left fifth of the page."""
    rgb = np.asarray(colour_page).astype(int)
    left_part = rgb[:, : PAGE_W // 5]
    redness = (left_part[..., 0] - (left_part[..., 1] + left_part[..., 2]) / 2).clip(min=0)
    return int(redness.sum(axis=0).argmax())


def find_blank_gap(colour_page, y_from, y_to, margin_x):
    """The tallest run of empty rows between y_from and y_to.

    Two things look like ink but aren't: the printed ruled lines (dark navy on some pages) and
    stray specks. Both are flat or tiny. Handwriting always has upright strokes (the stems of
    letters and digits), while a ruled line is only a few pixels tall. So: a pixel is ink if
    it is dark and colourless, and it is part of a STROKE if the pixels stacked on it in its
    column (about 1.4 mm) are all ink. A row is "written on" if enough strokes are near it.
    """
    rgb = np.asarray(colour_page).astype(int)[:, margin_x + 15 : PAGE_W - 15]
    grey = rgb.mean(axis=2)
    colourfulness = rgb.max(axis=2) - rgb.min(axis=2)
    ink = (grey < 120) & (colourfulness < 25)

    # Count ink in every `tall`-pixel window of each column with a running total (cumsum).
    # A window that is all ink is a piece of an upright stroke.
    tall = round(11 * PX)
    running = np.cumsum(np.vstack([np.zeros((1, ink.shape[1])), ink]), axis=0)
    upright = (running[tall:] - running[:-tall]) == tall          # row r covers rows r .. r+tall-1
    strokes_per_row = np.zeros(ink.shape[0])
    strokes_per_row[tall // 2 : tall // 2 + upright.shape[0]] = upright.sum(axis=1)

    # Add up the strokes within ~3.5 mm above and below. A line of text gives hundreds;
    # a stray dot gives a dozen or so.
    window = round(21 * PX)
    strokes_nearby = np.convolve(strokes_per_row, np.ones(window), mode="same")
    written = strokes_nearby > 60

    best_start, best_length, run_start = y_from, 0, None
    for y in range(y_from, y_to + 1):
        empty = y < y_to and not written[y]
        if empty and run_start is None:
            run_start = y
        if not empty and run_start is not None:
            if y - run_start > best_length:
                best_start, best_length = run_start, y - run_start
            run_start = None
    return best_start, best_start + best_length


def paste_into_gap(page, image_path, gap_top, gap_bottom, width_fraction, margin_x):
    """Scale the image to its planned width (or smaller, if the gap is short), give it a thin
    frame, and centre it in the gap over the writing area (margin line to right edge)."""
    art = Image.open(image_path)
    if art.mode in ("RGBA", "LA", "P"):                     # plots have a transparent background
        art = art.convert("RGBA")
        white = Image.new("RGBA", art.size, "white")
        art = Image.alpha_composite(white, art)
    art = art.convert("RGB")

    border = round(12 * PX)                                 # white space between image and frame
    clear = round(20 * PX)                                  # keep this much paper above and below
    room_h = (gap_bottom - gap_top) - 2 * clear - 2 * border
    target_w = width_fraction * PAGE_W
    scale = min(target_w / art.width, room_h / art.height)
    art = art.resize((round(art.width * scale), round(art.height * scale)), Image.LANCZOS)

    framed_w, framed_h = art.width + 2 * border, art.height + 2 * border
    centre_x = (margin_x + PAGE_W) // 2
    x0 = centre_x - framed_w // 2
    y0 = (gap_top + gap_bottom) // 2 - framed_h // 2

    draw = ImageDraw.Draw(page)
    draw.rectangle([x0, y0, x0 + framed_w, y0 + framed_h], fill="white", outline=(70, 70, 70), width=2)
    page.paste(art, (x0 + border, y0 + border))
    return scale, (x0, y0, framed_w, framed_h)


def build_page(number, photo, slot):
    """Turn one page photo into a finished page. Returns (page image, report line)."""
    photo, cloth_pixels = whiten_cloth_at_edges(photo)

    # The finished page is greyscale. A 3x3 median filter at full resolution removes single-
    # pixel speckle before shrinking; at 300 dpi it is far smaller than any pen stroke.
    grey_photo = photo.convert("L").filter(ImageFilter.MedianFilter(3))
    grey_page, _, _ = fit_on_a4(grey_photo)
    grey_page = clean_paper(grey_page)

    if slot is None:
        report = f"cleaned {cloth_pixels} edge pixels" if cloth_pixels else ""
        return grey_page, report

    # Finding the margin and the gap needs colour (red margin line, purple ruled lines).
    band, image, width = slot
    colour_page, photo_top, photo_h = fit_on_a4(photo)
    margin_x = find_margin_line(colour_page)
    y_from = photo_top + int(band[0] * photo_h)
    y_to = photo_top + int(band[1] * photo_h)
    gap_top, gap_bottom = find_blank_gap(colour_page, y_from, y_to, margin_x)

    # Paste the colour question/plot onto the grey page.
    page = grey_page.convert("RGB")
    scale, box = paste_into_gap(page, image, gap_top, gap_bottom, width, margin_x)
    report = (f"{image.name:<30} gap {gap_bottom - gap_top:4d} px, "
              f"box {box[2]} x {box[3]} px at y={box[1]}, scale {scale:.2f}")
    return page, report


def main():
    with tempfile.TemporaryDirectory() as tmp:
        photos = extract_page_photos(SOURCE, tmp)
        slots_by_page = {page: (band, image, width) for page, band, image, width in SLOTS}

        pdf_parts = []                                       # one single-page PDF per page
        for number in sorted(photos):
            photo = Image.open(photos[number]).convert("RGB")
            page, report = build_page(number, photo, slots_by_page.get(number))
            if report:
                print(f"page {number:2d}: {report}")

            # Each page gets its own JPEG quality, so save it as its own one-page PDF first.
            quality = QUALITY_PASTED if page.mode == "RGB" else QUALITY_TEXT
            part = Path(tmp) / f"part-{number:03d}.pdf"
            page.save(part, resolution=float(DPI), quality=quality, optimize=True)
            pdf_parts.append(str(part))

        # Join the one-page PDFs into one file (pdfunite ships with poppler, like pdfimages).
        subprocess.run(["pdfunite", *pdf_parts, str(OUTPUT)], check=True)

    print(f"\nwrote {OUTPUT.name}: {len(pdf_parts)} pages, "
          f"{os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB (scan was {os.path.getsize(SOURCE) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
