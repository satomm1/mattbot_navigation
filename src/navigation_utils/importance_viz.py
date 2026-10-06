"""PNG of obstacle importance on the roadmap (matplotlib, headless; no ROS).

Draws the static map, the roadmap, and each obstacle filled by its I_o (one-hue sequential
ramp, shared colorbar) with its blocked edges in the same colour. Obstacles that block nothing
are hollow outlines. Labels: "id: I_o / I_o_disc m, frac%".
"""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Light theme (reference dataviz palette): surface, ink, neutrals, sequential blue 250..700
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
WALL = "#52514e"
UNKNOWN = "#e2e1dc"
CSPACE = "#f0efec"
ROADMAP = "#b4b2ab"
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
IMPORTANCE_CMAP = LinearSegmentedColormap.from_list("importance", SEQ_BLUE)


def _background(ax, roadmap, occupancy):
    """Map image: free = surface, C-space margin, unknown, walls. occupancy = fine map or None."""
    if occupancy is not None:
        data, width, height, res, origin = occupancy
        grid = np.asarray(data, dtype=np.int16).reshape(height, width)
        img = np.zeros(grid.shape, dtype=np.uint8)  # 0 free
        img[grid < 0] = 1
        img[grid >= 50] = 3
        extent = (origin[0], origin[0] + width * res, origin[1], origin[1] + height * res)
    else:
        img = np.where(roadmap.occ, 3, 0).astype(np.uint8)
        h, w = roadmap.shape
        res, origin = roadmap.res, roadmap.origin
        extent = (origin[0], origin[0] + w * roadmap.res, origin[1], origin[1] + h * roadmap.res)
    cmap = ListedColormap([SURFACE, UNKNOWN, CSPACE, WALL])
    ax.imshow(img, cmap=cmap, vmin=0, vmax=3, origin="lower", extent=extent, interpolation="nearest", zorder=0)
    # Lightly shade free cells the robot centre cannot reach (C-space obstacles)
    h, w = roadmap.shape
    margin = np.ma.masked_where(roadmap.cfree | roadmap.occ, np.ones(roadmap.shape))
    ax.imshow(
        margin, cmap=ListedColormap([CSPACE]), origin="lower", interpolation="nearest", zorder=1,
        extent=(roadmap.origin[0], roadmap.origin[0] + w * roadmap.res,
                roadmap.origin[1], roadmap.origin[1] + h * roadmap.res),
    )


def _polyline_xy(roadmap, poly):
    return np.stack(
        [roadmap.origin[0] + (poly[:, 1] + 0.5) * roadmap.res, roadmap.origin[1] + (poly[:, 0] + 0.5) * roadmap.res],
        axis=1,
    )


_LABEL_OFFSETS_PT = [(6, 6), (6, -14), (6, 18), (6, -26), (6, 30), (6, -38), (-6, 6), (-6, -14)]


class _LabelPlacer:
    """Greedy: pick the first offset whose (approximate) label box overlaps no placed label."""

    def __init__(self, ax, fontsize):
        self.ax = ax
        self.fontsize = fontsize
        self.boxes = []

    def offset(self, x, y, text):
        px_per_pt = self.ax.figure.dpi / 72.0
        ax_x, ax_y = self.ax.transData.transform((x, y))
        w, h = 0.6 * self.fontsize * len(text) * px_per_pt, 1.2 * self.fontsize * px_per_pt
        frame = self.ax.get_window_extent()
        for dx, dy in _LABEL_OFFSETS_PT:
            x0 = ax_x + dx * px_per_pt - (w if dx < 0 else 0.0)
            y0 = ax_y + dy * px_per_pt
            box = (x0, y0, x0 + w, y0 + h)
            if box[0] < frame.x0 or box[2] > frame.x1 or box[1] < frame.y0 or box[3] > frame.y1:
                continue
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in self.boxes):
                self.boxes.append(box)
                return (dx, dy), ("right" if dx < 0 else "left")
        self.boxes.append(box)
        return _LABEL_OFFSETS_PT[0], "left"


def render_importance_png(path, roadmap, results, objects, trips=None, title=None, occupancy=None, dpi=150):
    """Write a PNG of the roadmap and the importance of each object.

    results:   {object_id: ImportanceResult}
    objects:   {object_id: (x, y, width)} in the local map frame
    trips:     TripSet (landmarks are drawn when given)
    occupancy: (data, width, height, resolution, (origin_x, origin_y)) of the fine static map;
               the coarse roadmap grid is drawn when None.
    """
    g = roadmap.graph
    xy = roadmap.node_xy
    if len(xy):
        lo, hi = xy.min(axis=0) - 2.0, xy.max(axis=0) + 2.0
    else:
        lo, hi = np.array(roadmap.origin), np.array(roadmap.origin) + np.array(roadmap.shape[::-1]) * roadmap.res
    span = np.maximum(hi - lo, 1.0)
    fig_w = 12.0
    fig_h = float(np.clip(fig_w * span[1] / span[0], 4.0, 14.0)) + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    _background(ax, roadmap, occupancy)

    segs = [_polyline_xy(roadmap, d["polyline"]) for _u, _v, d in g.edges(data=True)]
    ax.add_collection(LineCollection(segs, colors=ROADMAP, linewidths=0.8, zorder=2))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=ROADMAP, linewidths=0, zorder=3)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    placer = _LabelPlacer(ax, fontsize=7)

    vmax = max([r.I_o for r in results.values()] + [1e-6])
    norm = Normalize(vmin=0.0, vmax=vmax)
    halo = [patheffects.withStroke(linewidth=2.5, foreground=SURFACE)]

    # Most important first for label placement; zorder keeps important patches on top
    for oid in sorted(results, key=lambda k: -results[k].I_o):
        r = results[oid]
        if oid not in objects:
            continue
        x, y, width = objects[oid]
        side = max(float(width), roadmap.res)
        color = IMPORTANCE_CMAP(norm(r.I_o))
        if r.blocked_edges:
            bsegs = [_polyline_xy(roadmap, g.edges[u, v]["polyline"]) for u, v in r.blocked_edges if g.has_edge(u, v)]
            ax.add_collection(LineCollection(bsegs, colors=[color], linewidths=3.0, capstyle="round",
                                             zorder=4 + r.I_o / vmax))
        if r.I_o > 0:
            patch = Rectangle((x - side / 2, y - side / 2), side, side, facecolor=color, edgecolor=SURFACE,
                              linewidth=1.0, zorder=5 + r.I_o / vmax)
        else:
            patch = Rectangle((x - side / 2, y - side / 2), side, side, facecolor="none", edgecolor=TEXT_SECONDARY,
                              linewidth=1.2, zorder=5)
        ax.add_patch(patch)
        label = "%s: %.1f / %.1f m, %.0f%%" % (oid, r.I_o, r.I_o_disc, 100.0 * r.frac_affected)
        off, ha = placer.offset(x, y, label)
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=7, ha=ha,
                    color=TEXT_PRIMARY if r.I_o > 0 else TEXT_SECONDARY, path_effects=halo, zorder=7)

    if trips is not None and len(trips.landmark_nodes):
        lxy = xy[trips.landmark_nodes]
        ax.scatter(lxy[:, 0], lxy[:, 1], marker="^", s=22, facecolor=SURFACE, edgecolor=TEXT_SECONDARY,
                   linewidths=0.8, zorder=6)
        for (lx, ly), name in zip(lxy, trips.landmark_names):
            if name:
                ax.annotate(name, (lx, ly), xytext=(4, -9), textcoords="offset points", fontsize=6,
                            color=TEXT_SECONDARY, path_effects=halo, zorder=7)

    ax.set_xlabel("x (m)", color=TEXT_SECONDARY, fontsize=8)
    ax.set_ylabel("y (m)", color=TEXT_SECONDARY, fontsize=8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(ROADMAP)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=IMPORTANCE_CMAP)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("I_o (m extra per trip)", color=TEXT_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    cbar.outline.set_edgecolor(ROADMAP)

    n_block = sum(1 for r in results.values() if r.blocked_edges)
    subtitle = "%d obstacle(s), %d blocking roadmap edges; label = id: I_o / I_o_disc, %% of trips affected" % (
        len(results), n_block)
    if trips is not None:
        subtitle += "; %d trips (%s)" % (len(trips), trips.mode)
    ax.set_title(subtitle, fontsize=8, color=TEXT_SECONDARY, loc="left")
    fig.suptitle(title or "Obstacle importance", x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".png")
    os.close(fd)
    try:
        fig.savefig(tmp, dpi=dpi, facecolor=SURFACE)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path
