"""Write a printable PDF of the table board, the object markers and a calibration checkerboard.

Sizes come from configs/capture.yaml (markers) and e2l_perception.calibration (checkerboard).
Print at 100% / "actual size" and check the 100 mm scale bar on each page with a ruler.

    uv run python scripts/make_markers.py --out outputs/markers.pdf
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.backends.backend_pdf import PdfPages

from e2l_common.config import CaptureConfig, load_config
from e2l_perception.calibration import BOARD_SIZE, SQUARE_M
from e2l_perception.markers import get_dictionary, make_board

MM = 1 / 25.4  # inches per mm
PX_PER_MM = 20


def page(pdf: PdfPages, size_mm: tuple[float, float], draw) -> None:
    fig = plt.figure(figsize=(size_mm[0] * MM, size_mm[1] * MM))
    draw(fig, size_mm)
    pdf.savefig(fig)
    plt.close(fig)


def place(fig, size_mm, img: np.ndarray, x_mm: float, y_mm: float, w_mm: float, h_mm: float):
    """Place an image with its lower-left corner at (x_mm, y_mm) from the page's lower-left."""
    W, H = size_mm
    ax = fig.add_axes((x_mm / W, y_mm / H, w_mm / W, h_mm / H))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
    ax.set_axis_off()


def text(fig, size_mm, x_mm: float, y_mm: float, s: str, **kw):
    fig.text(x_mm / size_mm[0], y_mm / size_mm[1], s, fontsize=kw.pop("fontsize", 9), **kw)


def scale_bar(fig, size_mm, x_mm: float, y_mm: float):
    bar = np.zeros((int(2 * PX_PER_MM), int(100 * PX_PER_MM)), np.uint8)
    place(fig, size_mm, bar, x_mm, y_mm, 100, 2)
    text(fig, size_mm, x_mm, y_mm + 3, "100 mm: check with a ruler after printing")


def main(
    capture_config: Path = typer.Option(Path("configs/capture.yaml")),
    out: Path = typer.Option(Path("outputs/markers.pdf")),
) -> None:
    cfg: CaptureConfig = load_config(capture_config, model=CaptureConfig)
    dictionary = get_dictionary(cfg.aruco_dictionary)
    tb = cfg.table_board
    board = make_board(tb, dictionary)
    bw = (tb.markers_x * tb.marker_length + (tb.markers_x - 1) * tb.marker_separation) * 1000
    bh = (tb.markers_y * tb.marker_length + (tb.markers_y - 1) * tb.marker_separation) * 1000
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        a4 = (210.0, 297.0)
        if bw > a4[0] - 20 or bh > a4[1] - 60:
            raise typer.BadParameter(f"table board {bw:.0f}x{bh:.0f} mm does not fit on A4")

        def board_page(fig, size):
            img = board.generateImage((round(bw * PX_PER_MM), round(bh * PX_PER_MM)), marginSize=0)
            x0, y0 = (size[0] - bw) / 2, size[1] - 40 - bh
            place(fig, size, img, x0, y0, bw, bh)
            text(fig, size, 15, size[1] - 20, f"Table board ({cfg.aruco_dictionary})", fontsize=14)
            last_id = tb.first_id + tb.markers_x * tb.markers_y - 1
            text(
                fig,
                size,
                15,
                size[1] - 28,
                f"{tb.markers_x}x{tb.markers_y} markers, ids {tb.first_id}..{last_id}, "
                f"marker {tb.marker_length * 1000:.0f} mm, "
                f"gap {tb.marker_separation * 1000:.0f} mm.",
            )
            text(
                fig,
                size,
                15,
                size[1] - 34,
                "Table frame: board centre, x right, y up the page, z out of the paper.",
            )
            scale_bar(fig, size, 15, 20)

        page(pdf, a4, board_page)

        def objects_page(fig, size):
            text(fig, size, 15, size[1] - 20, "Object markers", fontsize=14)
            x, y = 15.0, size[1] - 40
            for name, obj in cfg.objects.items():
                L = obj.marker_length * 1000
                quiet = L / 4  # white border so the detector finds the edge
                if x + L + 2 * quiet > size[0] - 10:
                    x, y = 15.0, y - L - 2 * quiet - 15
                cell = round((L + 2 * quiet) * PX_PER_MM)
                img = np.full((cell, cell), 255, np.uint8)
                m = cv2.aruco.generateImageMarker(dictionary, obj.marker_id, round(L * PX_PER_MM))
                q = round(quiet * PX_PER_MM)
                img[q : q + m.shape[0], q : q + m.shape[1]] = m
                place(fig, size, img, x, y - L - 2 * quiet, L + 2 * quiet, L + 2 * quiet)
                text(fig, size, x, y + 2, f"{name}: id {obj.marker_id}, {L:.0f} mm")
                x += L + 2 * quiet + 15
            scale_bar(fig, size, 15, 20)

        page(pdf, a4, objects_page)

        def checker_page(fig, size):
            cols, rows = BOARD_SIZE[0] + 1, BOARD_SIZE[1] + 1  # squares
            sq = SQUARE_M * 1000
            img = (np.indices((rows, cols)).sum(0) % 2 == 0).astype(np.uint8) * 255
            img = np.kron(img, np.ones((int(sq * 4), int(sq * 4)), np.uint8))
            w, h = cols * sq, rows * sq
            place(fig, size, img, (size[0] - w) / 2, 14, w, h)
            text(fig, size, 10, size[1] - 10 + 3,
                 f"Calibration checkerboard: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} inner corners, "
                 f"{sq:.0f} mm squares. Tape flat to a rigid board.")  # fmt: skip
            scale_bar(fig, size, 10, 4)

        page(pdf, (297.0, 210.0), checker_page)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
