#!/usr/bin/env python3
"""Draw an implementation-grounded, editable two-row storage-link schematic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
from matplotlib import get_data_path
from reportlab import rl_config
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WIDTH, HEIGHT = 510.0, 222.0
INK = "#17202A"
BLUE = "#2C6088"
PALE_BLUE = "#EFF5F8"
GRAY = "#64717C"
FONT = "SystemLinkSans"
BOLD = "SystemLinkSansBold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    font_dir = Path(get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont(FONT, str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(BOLD, str(font_dir / "DejaVuSans-Bold.ttf")))
    # Plain Flate streams support independent PDF text-size inspection.
    rl_config.useA85 = False
    pdf_path = args.out / "fig_system_link.pdf"
    pdf = canvas.Canvas(
        str(pdf_path),
        pagesize=(WIDTH, HEIGHT),
        invariant=1,
        initialFontName=FONT,
        initialFontSize=7.2,
    )
    pdf.setTitle("Implemented storage-link simulation")
    pdf.setAuthor("")
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="510pt" height="222pt" '
        'viewBox="0 0 510 222" role="img" aria-labelledby="title desc">',
        '<title id="title">Implemented storage-link simulation</title>',
        '<desc id="desc">Eight implemented stages in two rows; only public '
        "parameters and observed reads enter the receiver interface.</desc>",
        '<rect id="background" x="0" y="0" width="510" height="222" fill="white"/>',
    ]
    text_bounds = []
    node_bounds = {}
    edges = []

    def rect(identity, x, y, w, h, *, color=INK, fill="white", dashed=False):
        svg.append(
            f'<rect id="{identity}" x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="{color}" stroke-width="0.75"'
            + (' stroke-dasharray="3 2"' if dashed else "")
            + "/>"
        )
        pdf.setStrokeColor(color)
        pdf.setFillColor(fill)
        pdf.setLineWidth(0.75)
        pdf.setDash(3, 2) if dashed else pdf.setDash()
        pdf.rect(x, HEIGHT - y - h, w, h, stroke=1, fill=1)
        pdf.setDash()

    def text(identity, value, x, y, size=7.2, anchor="middle", bold=False, box=None):
        face = BOLD if bold else FONT
        width = pdfmetrics.stringWidth(value, face, size)
        left = (
            x - width / 2 if anchor == "middle" else x - width if anchor == "end" else x
        )
        assert 0 <= left <= left + width <= WIDTH, (identity, value, left, width)
        if box:
            bx, by, bw, bh = box
            assert bx + 4 <= left and left + width <= bx + bw - 4, (identity, width)
            assert by + 5 < y < by + bh - 2
        svg.append(
            f'<text id="{identity}" x="{x}" y="{y}" fill="{INK}" '
            f'font-family="DejaVu Sans" font-size="{size}pt" '
            f'font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}">'
            f"{escape(value)}</text>"
        )
        pdf.setFillColor(INK)
        pdf.setFont(face, size)
        pdf.drawString(left, HEIGHT - y, value)
        text_bounds.append(
            {
                "id": identity,
                "text": value,
                "x": left,
                "y_baseline": y,
                "width": width,
                "size_pt": size,
            }
        )

    def arrow(identity, x1, y1, x2, y2):
        assert x1 == x2 or y1 == y2
        if x2 > x1:
            triangle = [(x2, y2), (x2 - 4, y2 - 2), (x2 - 4, y2 + 2)]
        elif x2 < x1:
            triangle = [(x2, y2), (x2 + 4, y2 - 2), (x2 + 4, y2 + 2)]
        else:
            assert y2 > y1
            triangle = [(x2, y2), (x2 - 2, y2 - 4), (x2 + 2, y2 - 4)]
        points = " ".join(f"{x},{y}" for x, y in triangle)
        svg.append(
            f'<g id="{identity}"><path d="M{x1} {y1} L{x2} {y2}" '
            f'stroke="{INK}" stroke-width="0.9" fill="none"/>'
            f'<polygon points="{points}" fill="{INK}"/></g>'
        )
        pdf.setStrokeColor(INK)
        pdf.setLineWidth(0.9)
        pdf.line(x1, HEIGHT - y1, x2, HEIGHT - y2)
        path = pdf.beginPath()
        path.moveTo(triangle[0][0], HEIGHT - triangle[0][1])
        for x, y in triangle[1:]:
            path.lineTo(x, HEIGHT - y)
        path.close()
        pdf.setFillColor(INK)
        pdf.drawPath(path, stroke=0, fill=1)
        edges.append({"id": identity, "start": [x1, y1], "end": [x2, y2]})

    text("t:sender", "SENDER", 9, 16, 8.5, anchor="start", bold=True)
    text("t:channel", "CHANNEL MODEL", 393, 16, 8.5, anchor="start", bold=True)
    rect("c:receiver-boundary", 4, 111, 502, 92, color=GRAY, dashed=True)
    text(
        "t:receiver",
        "RECEIVER — public protocol + observed reads only",
        14,
        126,
        8.2,
        anchor="start",
        bold=True,
    )

    nodes = [
        (
            "file-framing",
            9,
            32,
            "1  File framing",
            "length + file CRC",
            "1202 useful bytes",
        ),
        (
            "outer-encode",
            137,
            32,
            "2  Outer RS protection",
            "RS(63,55), GF(256)",
            "63 × 22-byte frames",
        ),
        (
            "dna-mapping",
            265,
            32,
            "3  Framing + DNA map",
            "address / mode / CRC",
            "150-nt body",
        ),
        (
            "readout",
            393,
            32,
            "4  Readout simulation",
            "iid IDS or source profile",
            "unordered read sequences",
        ),
        (
            "assignment",
            393,
            147,
            "5  Address assignment",
            "public address collection",
            "reject ambiguity",
        ),
        (
            "reconstruction",
            265,
            147,
            "6  Read reconstruction",
            "medoid / consensus",
            "no true-origin clustering",
        ),
        (
            "inverse-crc",
            137,
            147,
            "7  Inverse + frame CRC",
            "bounded candidate testing",
            "payload or erasure",
        ),
        (
            "outer-decode",
            9,
            147,
            "8  RS + file check",
            "recovery across strands",
            "bytes or declared failure",
        ),
    ]
    for identity, x, y, title, detail1, detail2 in nodes:
        box = (x, y, 108, 44)
        svg.append(f'<g id="n:{identity}">')
        rect(
            f"s:{identity}",
            *box,
            color=BLUE if identity == "dna-mapping" else INK,
            fill=PALE_BLUE if identity == "dna-mapping" else "white",
        )
        text(f"t:{identity}:title", title, x + 54, y + 13, 7.6, box=box)
        text(f"t:{identity}:detail1", detail1, x + 54, y + 26, 7.2, box=box)
        text(f"t:{identity}:detail2", detail2, x + 54, y + 37, 7.2, box=box)
        svg.append("</g>")
        node_bounds[identity] = box

    arrow("e:1-2", 117, 54, 137, 54)
    arrow("e:2-3", 245, 54, 265, 54)
    arrow("e:3-4", 373, 54, 393, 54)
    arrow("e:4-5", 447, 76, 447, 147)
    text("t:observed-bases", "observed bases", 439, 92, 7.2, anchor="end")
    text("t:opaque-ids", "opaque read IDs", 439, 102, 7.2, anchor="end")
    arrow("e:5-6", 393, 169, 373, 169)
    arrow("e:6-7", 265, 169, 245, 169)
    arrow("e:7-8", 137, 169, 117, 169)
    text(
        "t:truth-boundary",
        "Evaluation truth is excluded from the receiver interface.",
        9,
        216,
        7.2,
        anchor="start",
    )
    pdf.showPage()
    pdf.save()
    svg.append("</svg>")
    (args.out / "fig_system_link.svg").write_text(
        "\n".join(svg) + "\n", encoding="utf-8"
    )
    with fitz.open(pdf_path) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        doc[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(
            args.out / "fig_system_link.png"
        )
        doc[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).save(
            args.out / "fig_system_link_final_size.png"
        )
        fonts = doc[0].get_fonts(full=True)
        assert all(doc.extract_font(f[0])[3] for f in fonts), fonts
        actual_text = doc[0].get_text()
        assert all(
            row[3].replace("  ", " ") in actual_text.replace("  ", " ") for row in nodes
        )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.iterdir()
        if p.is_file()
    }
    result = {
        "status": "PASS_SOURCE_GEOMETRY_AND_EMBEDDED_LIVE_TEXT_NOT_VISUAL_ACCEPTANCE",
        "canvas_pt": [WIDTH, HEIGHT],
        "font": "DejaVu Sans",
        "text_count": len(text_bounds),
        "node_bounds": node_bounds,
        "edges": edges,
        "text_geometry": text_bounds,
        "sha256": hashes,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "geometry_and_exports.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.out), "sha256": hashes},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
