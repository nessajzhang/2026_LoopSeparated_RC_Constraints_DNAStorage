"""Distributable rectangle-based figure gate; no personal Nature-skill dependency.

The checked scope is physical panel geometry, not scientific correctness or
text overlap. Visual QA of every compiled manuscript page remains separate.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


class PanelAlignmentError(RuntimeError):
    pass


def require_matplotlib_panel_alignment(fig, *, json_out=None, overlay_svg=None,
                                      tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                      strict=False, panel_ids=None, **options):
    if options:
        raise ValueError(f"Unsupported portable QA options: {sorted(options)}")
    fig.canvas.draw()
    width, height = (float(x) * 72 for x in fig.get_size_inches())
    axes = list(fig.axes)
    panel_ids = panel_ids or [str(i) for i in range(len(axes))]
    if len(panel_ids) != len(axes) or len(set(panel_ids)) != len(panel_ids):
        raise ValueError("One unique panel ID per axis is required")
    rows, columns, panels, failures = defaultdict(list), defaultdict(list), [], []
    for name, axis in zip(panel_ids, axes):
        box = axis.get_position()
        b = [box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height]
        if b[0] < 0 or b[1] < 0 or b[2] > width or b[3] > height:
            failures.append(f"{name}: panel outside figure")
        spec = axis.get_subplotspec()
        if spec is None:
            raise ValueError("Portable gate requires SubplotSpec for each axis")
        panel = {"id": name, "bbox_pt": b, "width_pt": b[2] - b[0],
                 "height_pt": b[3] - b[1]}
        panels.append(panel)
        grid = id(spec.get_gridspec())
        rows[grid, spec.rowspan.start, spec.rowspan.stop].append(panel)
        columns[grid, spec.colspan.start, spec.colspan.stop].append(panel)
    compared = 0
    for direction, groups in (("row", rows), ("column", columns)):
        for group in groups.values():
            if len(group) < 2:
                continue
            compared += 1
            edges = (1, 3) if direction == "row" else (0, 2)
            for edge in edges:
                values = [p["bbox_pt"][edge] for p in group]
                if max(values) - min(values) > tolerance_pt:
                    failures.append(f"{direction} edge {edge} misaligned")
            if direction == "row" and len(group) > 2:
                ordered = sorted(group, key=lambda p: p["bbox_pt"][0])
                gaps = [right["bbox_pt"][0] - left["bbox_pt"][2]
                        for left, right in zip(ordered, ordered[1:])]
                if max(gaps) - min(gaps) > gutter_tolerance_pt:
                    failures.append("Unequal row gutters")
    if strict and not compared:
        failures.append("No comparable multi-panel groups")
    report = {"schema": "portable-panel-alignment-v1", "status": "FAIL" if failures else "PASS",
              "scope": "Panel rectangles only; not text overlap or scientific validation",
              "width_pt": width, "height_pt": height, "panels": panels,
              "groups_checked": compared, "errors": failures,
              "tolerance_pt": tolerance_pt, "gutter_tolerance_pt": gutter_tolerance_pt}
    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if overlay_svg:
        rectangles = []
        for panel in panels:
            left, bottom, right, top = panel["bbox_pt"]
            rectangles.append(f'<rect x="{left}" y="{height-top}" width="{right-left}" '
                              f'height="{top-bottom}" fill="none" stroke="black"/>')
        Path(overlay_svg).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" '
                                    f'viewBox="0 0 {width} {height}">' + "".join(rectangles) + "</svg>\n")
    if failures:
        raise PanelAlignmentError("; ".join(failures))
    return report
