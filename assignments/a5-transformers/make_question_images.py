"""Cut each question out of the assignment PDF as a clean image, for the top of each question in the notes.

Run from anywhere:
    uv run --no-sync python assignments/a5-transformers/make_question_images.py

Writes question-images/q01.png ... q10.png and prints the box size (16 cm wide) each one needs.

The PDF's running header ("Transformer Assignment • ...") sits in the middle of the text on
pages 2-4, and questions 2, 5 and 8 run over a page break. So each question is described as a
list of vertical slices (page, top, bottom) in PDF points, which skip the header band; the
slices are stacked into one image.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
PDF = HERE.parent.parent / "questions" / "a5-transformers" / "Transformer_Mathematical_Python_Assignment.pdf"
OUT = HERE / "question-images"

DPI = 220                     # sharp enough to stay crisp when printed 16 cm wide
SCALE = DPI / 72              # PDF points -> pixels
LEFT, RIGHT = 84, 528         # horizontal crop in points (the text spans about 90-522)
BOX_WIDTH_CM = 16.0           # every question box in the notes is 16 cm wide

# (page, top, bottom) in PDF points, measured from the top of the page with `pdftotext -bbox-layout`
QUESTIONS = {
    1:  [(1, 343, 619)],
    2:  [(1, 619, 741), (2, 75, 176)],                  # page break; skip the header at 63-74 on page 2
    3:  [(2, 176, 387)],
    4:  [(2, 387, 600)],
    5:  [(2, 600, 765), (3, 49, 64), (3, 96, 147)],     # page break; skip the header at 84-95 on page 3
    6:  [(3, 147, 425)],
    7:  [(3, 425, 622)],
    8:  [(3, 622, 754), (4, 45, 74), (4, 111, 226)],    # page break; skip the header at 99-110 on page 4
    9:  [(4, 226, 466)],
    10: [(4, 466, 672)],
}


def render_pages(tmp):
    """Render pages 1-4 of the PDF to PNG files and return them as PIL images, keyed by page number."""
    subprocess.run(["pdftoppm", "-r", str(DPI), "-f", "1", "-l", "4", "-png", str(PDF), f"{tmp}/pg"], check=True)
    pages = {}
    for path in sorted(Path(tmp).glob("pg-*.png")):
        page_number = int(path.stem.split("-")[1])
        pages[page_number] = Image.open(path).convert("RGB")
    return pages


def build_question_image(pages, slices):
    """Crop every slice and stack them top to bottom on a white background, with a little padding."""
    pieces = []
    for page, top, bottom in slices:
        box = (int(LEFT * SCALE), int(top * SCALE), int(RIGHT * SCALE), int(bottom * SCALE))
        pieces.append(pages[page].crop(box))

    pad = 20                                                # white border, in pixels
    width = max(piece.width for piece in pieces) + 2 * pad
    height = sum(piece.height for piece in pieces) + 2 * pad
    canvas = Image.new("RGB", (width, height), "white")
    y = pad
    for piece in pieces:
        canvas.paste(piece, (pad, y))
        y += piece.height
    return canvas


OUT.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    pages = render_pages(tmp)
    print("question   pixels        box to rule (cm)")
    for number, slices in QUESTIONS.items():
        image = build_question_image(pages, slices)
        image.save(OUT / f"q{number:02d}.png", dpi=(DPI, DPI))
        box_height_cm = BOX_WIDTH_CM * image.height / image.width
        print(f"   Q{number:<4}  {image.width} x {image.height:<5}   {BOX_WIDTH_CM:.0f} x {box_height_cm:.1f}")
