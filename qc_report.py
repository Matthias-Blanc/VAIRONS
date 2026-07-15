"""Independent batch QC report generator for Vairons.

Reads the per-sample artifacts produced by a batch run --
``<sample>/Analysis/Feature_summary.xlsx``, ``<sample>/Analysis/qc_meta.json``,
and the raw channel TIFFs -- and writes a single ``Batch_QC_report.pdf`` at the
batch root.

This script is deliberately standalone: it does NOT import the analysis modules
(``processing`` / ``io_manager``) and therefore never loads Cellpose or torch.

Usage:
    python qc_report.py <batch_root> [-o OUTPUT.pdf]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None
from skimage import io as skio

A4 = (8.27, 11.69)  # inches, portrait
SUMMARY_IDS = {"MEAN", "SEM"}


# =============================================================================
# DATA LOADING
# =============================================================================

def find_samples(root):
    """Find every <sample>/Analysis/Feature_summary.xlsx under root."""
    samples = []
    for dirpath, _, files in os.walk(root):
        p = Path(dirpath)
        if p.name == "Analysis" and "Feature_summary.xlsx" in files:
            samples.append(p.parent)
    return sorted(samples, key=lambda x: str(x).lower())


def _strip_summary(df, id_col="cell_id"):
    if id_col in df.columns:
        df = df[~df[id_col].astype(str).isin(SUMMARY_IDS)]
    return df.reset_index(drop=True)


def load_sample(sample_dir):
    analysis = sample_dir / "Analysis"
    xls = pd.ExcelFile(analysis / "Feature_summary.xlsx")
    sl = pd.read_excel(xls, "Sample_level") if "Sample_level" in xls.sheet_names else pd.DataFrame()
    sample_level = dict(zip(sl["metric"], sl["value"])) if not sl.empty else {}
    all_cells = (_strip_summary(pd.read_excel(xls, "All_cells"))
                 if "All_cells" in xls.sheet_names else pd.DataFrame())
    meta = {}
    meta_path = analysis / "qc_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return {
        "name": sample_dir.name,
        "dir": sample_dir,
        "sample_level": sample_level,
        "all_cells": all_cells,
        "meta": meta,
    }


def discover_channel_tiffs(sample_dir):
    """stem -> path for top-level TIFFs (the raw channels; Analysis/ is skipped).

    Matches .tif/.tiff case-insensitively; .tif wins over .tiff for the same stem.
    """
    found = {}
    for f in sorted(sample_dir.glob("*"), key=lambda p: p.name):
        if f.is_file() and f.suffix.lower() in (".tif", ".tiff"):
            found.setdefault(f.stem, f)
    return found


def read_image_dims(path):
    """Return (height, width, n_z) of a TIFF, best-effort; n_z is None for 2D."""
    h = w = nz = None
    if tifffile is not None:
        try:
            with tifffile.TiffFile(path) as tf:
                s = tf.series[0]
                axes = (s.axes or "").upper()
                shp = s.shape
                dim = {a: int(shp[i]) for i, a in enumerate(axes) if i < len(shp)}
                h, w = dim.get("Y"), dim.get("X")
                z = dim.get("Z")
                nz = z if (z and z > 1) else None
                if h is None or w is None:
                    p = tf.pages[0].shape
                    h, w = int(p[0]), int(p[1])
        except Exception:
            pass
    if h is None:
        try:
            arr = skio.imread(path)
            if arr.ndim == 3:
                nz, h, w = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
            else:
                h, w = int(arr.shape[0]), int(arr.shape[1])
        except Exception:
            pass
    return h, w, nz


# =============================================================================
# PLOT HELPERS
# =============================================================================

def _new_page(pdf, title):
    fig = plt.figure(figsize=A4)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    return fig


def _finish(pdf, fig, tight=True):
    if tight:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def bar_across_samples(ax, names, values, title, ylabel):
    x = np.arange(len(names))
    vals = [np.nan if v is None else v for v in values]
    ax.bar(x, vals, color="#4682b4")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)


def box_across_samples(ax, names, arrays, title, ylabel):
    data = [np.asarray(a, dtype=float) for a in arrays]
    data = [a[~np.isnan(a)] if a.size else np.array([np.nan]) for a in data]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)


def _col(samples, getter):
    return [getter(s) for s in samples]


def _cells_col(s, *cands):
    """Values of the first present per-cell column among candidates.

    Lets a single panel handle 2D and 3D samples (e.g. N_area_um2 vs N_volume_um3)
    and tolerate mixed-dimensionality batches.
    """
    ac = s["all_cells"]
    for c in cands:
        if c in ac:
            return ac[c].values
    return np.array([np.nan])


# =============================================================================
# REPORT PAGES
# =============================================================================

def _fmt_cellpose(params):
    if not params:
        return ""
    model = params.get("model_type", "cpsam")
    keys = ("diameter", "cellprob_threshold", "flow_threshold", "niter",
            "min_size", "max_size_fraction", "gpu", "invert")
    parts = [f"{k}={params[k]}" for k in keys if k in params and params[k] is not None]
    return f"{model} (" + ", ".join(parts) + ")"


def page_summary(pdf, root, samples):
    import importlib.metadata as im
    from collections import Counter
    fig = _new_page(pdf, "Vairons Batch QC Report")
    lines = [
        f"Batch root: {root}",
        f"Samples found: {len(samples)}",
        f"Total cells: {sum(int(s['sample_level'].get('cell_count', 0) or 0) for s in samples)}",
        "",
        "Configuration (from first sample's qc_meta.json):",
    ]
    meta0 = next((s["meta"] for s in samples if s["meta"]), {})
    for k in ("dimensionality", "pixel_size_um", "z_size_um", "nuclear_method",
              "cytoplasm_source", "cytoplasm_method", "filter_stain",
              "filter_condition", "remove_outliers", "outlier_mad_threshold",
              "channel_roles"):
        if k in meta0:
            lines.append(f"   {k}: {meta0[k]}")

    screen = meta0.get("outlier_screen")
    if screen:
        lines += ["", "Outlier screen:"]
        lines += [f"   {k}: {v}" for k, v in screen.items()]

    dist = meta0.get("outlier_size_distribution_by_channel")
    if dist:
        lines += ["", "Object size distribution per channel (first sample) — the "
                      "distribution the screen actually saw:",
                  f"   {'channel':16s} {'n':>7s} {'median':>10s} {'MAD(log10)':>11s} "
                  f"{'z range':>15s} {'flagged':>9s}"]
        for ch, st in dist.items():
            med = st.get("median_size"); mad = st.get("mad_log10")
            zlo, zhi = st.get("z_min"), st.get("z_max")
            zr = (f"[{zlo:+.1f},{zhi:+.1f}]" if zlo is not None and zhi is not None else "-")
            lines.append(
                f"   {ch:16s} {st.get('n_objects', 0):7d} "
                f"{(f'{med:.4g}' if med else '-'):>10s} "
                f"{(f'{mad:.3f}' if mad else '-'):>11s} {zr:>15s} "
                f"{st.get('n_objects_flagged', 0):5d} ({st.get('flag_rate', 0):.2%})")

    cp = meta0.get("cellpose")
    if cp:
        lines += ["", "Cellpose model & parameters:"]
        for detection, params in cp.items():
            lines.append(f"   {detection}: {_fmt_cellpose(params)}")

    # Image dimensions across all channel images (W x H [x Z] -> images/total)
    dim_counter = Counter()
    total_imgs = 0
    for s in samples:
        for _stem, path in discover_channel_tiffs(s["dir"]).items():
            h, w, nz = read_image_dims(path)
            if h:
                dim_counter[f"{w}x{h}x{nz}" if nz else f"{w}x{h}"] += 1
                total_imgs += 1
    lines += ["", "Image dimensions (W x H [x Z] -> images/total):"]
    for dim, cnt in dim_counter.most_common():
        lines.append(f"   {dim} -> {cnt}/{total_imgs}")

    lines += ["", "Environment:"]
    for pkg in ("numpy", "pandas", "scikit-image", "matplotlib", "cellpose", "tifffile"):
        try:
            lines.append(f"   {pkg}: {im.version(pkg)}")
        except Exception:
            pass
    ax = fig.add_axes([0.06, 0.04, 0.9, 0.88]); ax.axis("off")
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace")
    _finish(pdf, fig, tight=False)


def _channel_roles(samples):
    roles = {}
    for s in samples:
        for stem, role in (s["meta"].get("channel_roles") or {}).items():
            roles.setdefault(stem, role)
    return roles


def page_acquisition(pdf, samples):
    names = [s["name"] for s in samples]
    roles = _channel_roles(samples)
    # union of channels referenced by qc_*_intensity_mean across samples
    channels = set()
    for s in samples:
        for m in s["sample_level"]:
            if str(m).startswith("qc_") and str(m).endswith("_intensity_mean"):
                channels.add(str(m)[3:-len("_intensity_mean")])
    for ch in sorted(channels):
        fig = _new_page(pdf, f"Acquisition QC — channel '{ch}'"
                             + (f" ({roles[ch]})" if ch in roles else ""))
        means = _col(samples, lambda s: s["sample_level"].get(f"qc_{ch}_intensity_mean"))
        sat = _col(samples, lambda s: s["sample_level"].get(f"qc_{ch}_pct_saturated"))
        band = _col(samples, lambda s: s["sample_level"].get(f"qc_{ch}_pct_in_threshold_band"))
        # Top row (full width): mean intensity per sample.
        ax1 = fig.add_subplot(2, 1, 1); bar_across_samples(ax1, names, means, "Mean intensity", "intensity")
        # Bottom row: % saturated, [% threshold band], and a smaller intensity histogram.
        has_band = any(b is not None and not (isinstance(b, float) and np.isnan(b)) for b in band)
        if has_band:
            ax2 = fig.add_subplot(2, 3, 4); bar_across_samples(ax2, names, sat, "% saturated pixels", "%")
            ax3 = fig.add_subplot(2, 3, 5); bar_across_samples(ax3, names, band, "% in threshold band", "%")
            ax_hist = fig.add_subplot(2, 3, 6)
        else:
            ax2 = fig.add_subplot(2, 2, 3); bar_across_samples(ax2, names, sat, "% saturated pixels", "%")
            ax_hist = fig.add_subplot(2, 2, 4)
        # Intensity histograms overlaid across samples (smaller panel).
        plotted = 0
        for s in samples:
            path = discover_channel_tiffs(s["dir"]).get(ch)
            if path is None:
                continue
            try:
                img = skio.imread(path).ravel()
            except Exception:
                continue
            hi = 255 if img.max() <= 255 else int(img.max())
            counts, edges = np.histogram(img, bins=128, range=(0, hi))
            centers = (edges[:-1] + edges[1:]) / 2
            ax_hist.plot(centers, counts / counts.sum(), lw=0.6, alpha=0.6)
            plotted += 1
        ax_hist.set_title(f"Intensity histogram ({plotted} samples)", fontsize=9)
        ax_hist.set_xlabel("intensity", fontsize=8); ax_hist.set_ylabel("density", fontsize=8)
        ax_hist.tick_params(labelsize=7)
        _finish(pdf, fig)


def page_segmentation(pdf, samples):
    names = [s["name"] for s in samples]
    fig = _new_page(pdf, "Segmentation QC")
    counts = _col(samples, lambda s: s["sample_level"].get("cell_count"))
    ax1 = fig.add_subplot(3, 2, 1); bar_across_samples(ax1, names, counts, "Cell count", "cells")

    areas = _col(samples, lambda s: _cells_col(s, "N_volume_um3", "N_area_um2"))
    ax2 = fig.add_subplot(3, 2, 2)
    box_across_samples(ax2, names, areas, "Nuclear size", "um^3 / um^2")

    circ = _col(samples, lambda s: _cells_col(s, "N_sphericity", "N_circularity"))
    ax3 = fig.add_subplot(3, 2, 3)
    box_across_samples(ax3, names, circ, "Nuclear sphericity / circularity", "")

    soli = _col(samples, lambda s: _cells_col(s, "N_solidity"))
    ax4 = fig.add_subplot(3, 2, 4); box_across_samples(ax4, names, soli, "Nuclear solidity", "")

    # doublet / split counts from qc_meta
    ax5 = fig.add_subplot(3, 2, 5)
    initc, doub, splt = [], [], []
    for s in samples:
        nseg = ((s["meta"].get("segmentation") or {}).get("nuclear") or {})
        initc.append(nseg.get("initial_count", np.nan))
        doub.append(nseg.get("doublet_count", np.nan))
        splt.append(nseg.get("split_count", np.nan))
    x = np.arange(len(names))
    ax5.bar(x - 0.25, initc, 0.25, label="initial", color="#888")
    ax5.bar(x, doub, 0.25, label="doublets", color="#d98c00")
    ax5.bar(x + 0.25, splt, 0.25, label="split", color="#4682b4")
    ax5.set_xticks(x); ax5.set_xticklabels(names, rotation=90, fontsize=6)
    ax5.set_title("Doublet detection / splitting", fontsize=9)
    ax5.legend(fontsize=6); ax5.tick_params(axis="y", labelsize=7)

    # filtered fraction + cytoplasm-mapping completeness
    ax6 = fig.add_subplot(3, 2, 6)
    filt_frac = []
    for s in samples:
        cc = s["sample_level"].get("cell_count") or 0
        fc = (s["meta"].get("filtered_count") if s["meta"].get("filtered_count") is not None else np.nan)
        filt_frac.append((fc / cc) if cc else np.nan)
    bar_across_samples(ax6, names, filt_frac, "Filtered fraction (kept / total)", "fraction")
    _finish(pdf, fig)

    # cytoplasm mapping completeness + nuclear alignment (own page if cytoplasm present)
    has_cyto = any(("C_area_um2" in s["all_cells"]) or ("C_volume_um3" in s["all_cells"])
                   for s in samples)
    fig2 = _new_page(pdf, "Segmentation QC (continued)")
    ax7 = fig2.add_subplot(2, 1, 1)
    align = _col(samples, lambda s: s["sample_level"].get("nuclear_alignment"))
    bar_across_samples(ax7, names, align, "Nuclear alignment (0=random, 1=aligned)", "order param")
    if has_cyto:
        ax8 = fig2.add_subplot(2, 1, 2)
        compl = []
        for s in samples:
            ac = s["all_cells"]
            col = ("C_volume_um3" if "C_volume_um3" in ac
                   else "C_area_um2" if "C_area_um2" in ac else None)
            if col and len(ac):
                compl.append(float(ac[col].notna().mean()))
            else:
                compl.append(np.nan)
        bar_across_samples(ax8, names, compl, "Cytoplasm mapping completeness (cells with cytoplasm)", "fraction")
    _finish(pdf, fig2)


def page_outliers(pdf, samples):
    """Outlier-removal QC: how many cells each per-channel screen removed, the
    per-sample removal rate (with batch-robust flagging of anomalous samples), and a
    heatmap attributing stain-particle removals to the channel that misfired.

    Skipped entirely when no sample recorded outlier removal (older runs / disabled).
    """
    active = [s for s in samples
              if s["meta"].get("remove_outliers")
              and s["meta"].get("outlier_removed_count") is not None]
    if not active:
        return
    names = [s["name"] for s in samples]
    x = np.arange(len(names))

    thr_vals = {s["meta"].get("outlier_mad_threshold") for s in active
                if s["meta"].get("outlier_mad_threshold") is not None}
    thr_txt = (f"{next(iter(thr_vals)):g}" if len(thr_vals) == 1
               else "mixed" if thr_vals else "?")

    def _val(s, k):
        v = s["meta"].get(k)
        return np.nan if v is None else float(v)

    filt = np.array([_val(s, "filtered_count") for s in samples])
    kept = np.array([_val(s, "outlier_removed_count") for s in samples])
    removed = np.where(np.isfinite(filt) & np.isfinite(kept), filt - kept, np.nan)
    rate = np.where(np.isfinite(removed) & (filt > 0), removed / filt * 100.0, np.nan)

    metric = next((s["meta"].get("outlier_screen", {}).get("metric")
                   for s in active if s["meta"].get("outlier_screen")), "log10(size)")
    fig = _new_page(pdf, f"Outlier removal QC  —  two-sided robust z on {metric}, "
                         f"threshold = {thr_txt}")

    # Panel 1: cells removed per sample. Channels vote independently and overlap, so
    # a stacked by-screen breakdown would double-count; the per-channel votes are the
    # heatmap in panel 3.
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.bar(x, np.nan_to_num(removed), 0.7, color="#4682b4")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=90, fontsize=6)
    ax1.set_title("Cells removed as outliers (flagged by any channel)", fontsize=9)
    ax1.set_ylabel("cells removed", fontsize=8)
    ax1.tick_params(axis="y", labelsize=7)

    # Panel 2: removal rate; flag samples that are themselves robust-z outliers of
    # the batch rate distribution (same MAD logic the tool applies to cells).
    ax2 = fig.add_subplot(3, 1, 2)
    finite = rate[np.isfinite(rate)]
    colors = ["#4682b4"] * len(rate)
    hi = None
    if finite.size >= 3:
        med = np.median(finite); mad = np.median(np.abs(finite - med))
        if mad > 0:
            hi = med + 3 * 1.4826 * mad
            colors = ["#c0392b" if (np.isfinite(v) and v > hi) else "#4682b4" for v in rate]
    ax2.bar(x, np.nan_to_num(rate), color=colors)
    if hi is not None:
        ax2.axhline(hi, color="#c0392b", lw=0.7, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=90, fontsize=6)
    ax2.set_title("Outlier removal rate (% of filtered cells; red = high vs batch)", fontsize=9)
    ax2.set_ylabel("%", fontsize=8); ax2.tick_params(axis="y", labelsize=7)

    # Panel 3: how each channel voted. Every channel casts one binary vote per cell, so
    # an over-firing channel stands out immediately. Channels overlap; rows need not sum
    # to the removals in panel 1.
    chans = sorted({c for s in active
                    for c in (s["meta"].get("outlier_cells_flagged_by_channel") or {})})
    ax3 = fig.add_subplot(3, 1, 3)
    if chans:
        mat = np.full((len(chans), len(samples)), np.nan)
        for j, s in enumerate(samples):
            d = s["meta"].get("outlier_cells_flagged_by_channel") or {}
            for i, c in enumerate(chans):
                if c in d:
                    mat[i, j] = d[c]
        im_ = ax3.imshow(mat, aspect="auto", cmap="OrRd")
        ax3.set_yticks(np.arange(len(chans))); ax3.set_yticklabels(chans, fontsize=7)
        ax3.set_xticks(x); ax3.set_xticklabels(names, rotation=90, fontsize=6)
        ax3.set_title("Cells flagged by each channel (one vote per channel)", fontsize=9)
        cbar = fig.colorbar(im_, ax=ax3, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        vmax = np.nanmax(mat) if np.isfinite(mat).any() else 0
        for i in range(len(chans)):
            for j in range(len(samples)):
                if np.isfinite(mat[i, j]):
                    # White text on the dark (high) cells so counts stay legible.
                    tc = "white" if (vmax and mat[i, j] > 0.6 * vmax) else "#222"
                    ax3.text(j, i, int(mat[i, j]), ha="center", va="center",
                             fontsize=5, color=tc)
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "No outlier flags recorded",
                 ha="center", va="center", fontsize=8)
    _finish(pdf, fig)


def page_depth(pdf, samples):
    """Depth (z) QC for 3-D samples: intensity, stain coverage and nuclear density
    as a function of depth, plus the fraction of nuclei clipped by the z-border.

    These are the whole-mount bias checks — attenuation/bleaching, stain penetration
    and truncation. Skipped entirely when no sample carries depth profiles (2-D runs).
    """
    active = [s for s in samples if s["meta"].get("depth_profiles")]
    if not active:
        return
    fig = _new_page(pdf, "Depth (z) QC — whole-mount bias checks")

    chans = sorted({c for s in active
                    for c in s["meta"]["depth_profiles"].get("channel_intensity_by_z", {})})
    stains = sorted({c for s in active
                     for c in s["meta"]["depth_profiles"].get("stain_coverage_by_z", {})})

    def _cmap(names, cm):
        return {n: cm(i / max(1, len(names) - 1)) for i, n in enumerate(names)}
    ch_color = _cmap(chans, plt.cm.viridis)
    st_color = _cmap(stains, plt.cm.plasma)

    # A: per-channel intensity vs depth, normalized to each channel's own max so the
    #    attenuation/bleaching *shape* is comparable across channels and samples.
    axA = fig.add_subplot(2, 2, 1); seen = set()
    for s in active:
        dp = s["meta"]["depth_profiles"]; z = dp["z_um"]
        for c, prof in dp.get("channel_intensity_by_z", {}).items():
            a = np.asarray(prof, float); mx = a.max()
            lbl = c if c not in seen else None; seen.add(c)
            axA.plot(z, a / mx if mx > 0 else a, lw=0.9, alpha=0.65,
                     color=ch_color.get(c, "#888"), label=lbl)
    axA.set_title("Channel intensity vs depth (norm.)", fontsize=9)
    axA.set_xlabel("z (µm)", fontsize=8); axA.set_ylabel("mean / max", fontsize=8)
    axA.tick_params(labelsize=7); axA.legend(fontsize=6, loc="best")

    # B: stain coverage vs depth (penetration / attenuation of the marker).
    axB = fig.add_subplot(2, 2, 2)
    if stains:
        seen = set()
        for s in active:
            dp = s["meta"]["depth_profiles"]; z = dp["z_um"]
            for c, prof in dp.get("stain_coverage_by_z", {}).items():
                lbl = c if c not in seen else None; seen.add(c)
                axB.plot(z, np.asarray(prof, float) * 100, lw=0.9, alpha=0.65,
                         color=st_color.get(c, "#d98c00"), label=lbl)
        axB.legend(fontsize=6, loc="best")
    else:
        axB.axis("off"); axB.text(.5, .5, "No stain channels", ha="center", va="center", fontsize=8)
    axB.set_title("Stain coverage vs depth", fontsize=9)
    axB.set_xlabel("z (µm)", fontsize=8); axB.set_ylabel("% covered", fontsize=8); axB.tick_params(labelsize=7)

    # C: nuclear foreground fraction vs depth (object density falling off with depth).
    axC = fig.add_subplot(2, 2, 3)
    for s in active:
        dp = s["meta"]["depth_profiles"]; y = dp.get("nuclear_foreground_by_z")
        if y:
            axC.plot(dp["z_um"], np.asarray(y, float) * 100, lw=0.9, alpha=0.65, color="#3f7fb0")
    axC.set_title("Nuclear foreground vs depth", fontsize=9)
    axC.set_xlabel("z (µm)", fontsize=8); axC.set_ylabel("% of plane", fontsize=8); axC.tick_params(labelsize=7)

    # D: nuclei clipped by the first/last plane -> truncated volumes (see pitfall P-05).
    axD = fig.add_subplot(2, 2, 4)
    names, fracs = [], []
    for s in active:
        dp = s["meta"]["depth_profiles"]
        n = dp.get("n_nuclei") or 0; t = dp.get("n_nuclei_z_truncated") or 0
        names.append(s["name"]); fracs.append(100.0 * t / n if n else np.nan)
    x = np.arange(len(names))
    axD.bar(x, np.nan_to_num(fracs), color="#c0392b")
    axD.set_xticks(x); axD.set_xticklabels(names, rotation=90, fontsize=6)
    axD.set_title("Nuclei truncated by z-border (%)", fontsize=9)
    axD.set_ylabel("%", fontsize=8); axD.tick_params(axis="y", labelsize=7)
    _finish(pdf, fig)


def page_markers(pdf, samples):
    names = [s["name"] for s in samples]
    stems = set()
    for s in samples:
        for c in s["all_cells"].columns:
            if str(c).startswith("stain_") and str(c).endswith("_coverage_fraction_cell"):
                stems.add(str(c)[len("stain_"):-len("_coverage_fraction_cell")])
    for stem in sorted(stems):
        fig = _new_page(pdf, f"Marker QC — '{stem}'")
        def arr(s, col):
            c = f"stain_{stem}_{col}"
            return s["all_cells"][c].values if c in s["all_cells"] else np.array([np.nan])

        def measure_arr(s):  # stain volume (3D) or area (2D), whichever is present
            return _cells_col(s, f"stain_{stem}_volume_cell_um3",
                              f"stain_{stem}_area_cell_um2")
        ax1 = fig.add_subplot(3, 1, 1)
        box_across_samples(ax1, names, [arr(s, "coverage_fraction_cell") for s in samples],
                           "Whole-cell coverage fraction", "fraction")
        ax2 = fig.add_subplot(3, 1, 2)
        box_across_samples(ax2, names, [measure_arr(s) for s in samples],
                           "Stain volume / area per cell", "um^3 / um^2")
        ax3 = fig.add_subplot(3, 1, 3)
        box_across_samples(ax3, names, [arr(s, "particle_count_cell") for s in samples],
                           "Stain particle count per cell", "count")
        _finish(pdf, fig)


# =============================================================================
# MAIN
# =============================================================================

def generate(root, output=None):
    root = Path(root)
    sample_dirs = find_samples(root)
    if not sample_dirs:
        print(f"No Analysis/Feature_summary.xlsx found under {root}")
        return None
    print(f"Found {len(sample_dirs)} sample(s); loading...")
    samples = [load_sample(d) for d in sample_dirs]
    output = Path(output) if output else root / "Batch_QC_report.pdf"
    with PdfPages(output) as pdf:
        page_summary(pdf, root, samples)
        page_acquisition(pdf, samples)
        page_segmentation(pdf, samples)
        page_depth(pdf, samples)
        page_outliers(pdf, samples)
        page_markers(pdf, samples)
    print(f"QC report written to: {output}")
    return output


def main():
    ap = argparse.ArgumentParser(description="Generate a batch QC PDF for Vairons outputs.")
    ap.add_argument("root", help="Batch root folder (the one passed to Run Batch).")
    ap.add_argument("-o", "--output", default=None, help="Output PDF path.")
    args = ap.parse_args()
    generate(args.root, args.output)


if __name__ == "__main__":
    main()
